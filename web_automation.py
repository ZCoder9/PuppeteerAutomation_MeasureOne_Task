import asyncio
from pyppeteer import launch
import requests
import pytesseract
import os
from dotenv import load_dotenv
import cv2
import re
import numpy as np
from scipy.signal import find_peaks
import traceback

load_dotenv()

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = f"{os.getenv('TESSERACT_PATH')}"

CORRECTIONS = {"O": "9", "S": "5", "0": "9"}
WL = (
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    "-c load_system_dawg=0 -c load_freq_dawg=0"
)
SHEAR = -0.15

def _correct(text: str) -> str:
    return "".join(CORRECTIONS.get(c, c) for c in text)

def preprocess_and_ocr(image_bytes: bytes) -> str:
    if not image_bytes:
        raise ValueError("image_bytes is empty")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from bytes")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    red_mask = m1 | m2

    cleaned = np.full(img.shape[:2], 255, dtype=np.uint8)
    cleaned[red_mask > 0] = 0

    inv = cv2.bitwise_not(cleaned)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 15:
            cleaned[labels == i] = 255

    h, w = cleaned.shape
    upscaled = cv2.resize(cleaned, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    _, upscaled = cv2.threshold(upscaled, 128, 255, cv2.THRESH_BINARY)
    upscaled = cv2.copyMakeBorder(upscaled, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)

    deskew_inv = cv2.bitwise_not(upscaled)
    uh, uw = upscaled.shape
    msh = np.float32([[1, SHEAR, 0], [0, 1, 0]])
    new_uw = uw + int(uh * abs(SHEAR))
    deskewed_inv = cv2.warpAffine(deskew_inv, msh, (new_uw, uh), borderValue=0)
    deskewed = cv2.bitwise_not(deskewed_inv)
    padded = cv2.copyMakeBorder(deskewed, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

    raw_text = pytesseract.image_to_string(padded, config=f"--oem 3 --psm 8 {WL}")
    raw_text = re.sub(r"[^A-Z0-9]", "", raw_text)
    final_text = _correct(raw_text)

    if not final_text:
        return ""

    col_sum = np.sum(cv2.bitwise_not(padded) > 0, axis=0)
    smooth = np.convolve(col_sum, np.ones(10) / 10, mode="same")
    valleys, props = find_peaks(-smooth, distance=30, prominence=10)

    n_chars = len(final_text)
    if len(valleys) >= n_chars - 1:
        top_idx = np.argsort(props["prominences"])[-(n_chars - 1) :]
        split_cols = sorted(int(valleys[i]) for i in top_idx)
    else:
        split_cols = sorted(int(v) for v in valleys)

    content_cols = np.where(col_sum > 0)[0]
    if len(content_cols) == 0:
        return final_text

    x_start = int(content_cols[0])
    x_end = int(content_cols[-1]) + 1
    boundaries = [x_start] + split_cols + [x_end]

    colors = [
        (220, 60, 60),
        (60, 190, 60),
        (60, 60, 220),
        (0, 175, 225),
        (190, 60, 190),
    ]

    vis = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
    img_h = padded.shape[0]

    for idx in range(min(n_chars, len(boundaries) - 1)):
        x1, x2 = boundaries[idx], boundaries[idx + 1]
        strip = padded[:, x1:x2]
        rows_ok = np.any(cv2.bitwise_not(strip) > 0, axis=1)
        if not rows_ok.any():
            continue

        rmin, rmax = np.where(rows_ok)[0][[0, -1]]
        char = final_text[idx]
        color = colors[idx % len(colors)]

        cv2.rectangle(vis, (x1, rmin), (x2 - 1, rmax), color, 3)
        cv2.putText(
            vis,
            f"#{idx + 1}: {char}",
            (x1 + 4, max(rmin - 8, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis,
        f"Result: {final_text}",
        (20, img_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return final_text

async def main():
    try:
        browser = await launch(
            headless=False,
            executablePath=r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                    args=[
                '--no-sandbox',
                '--disable-setuid-sandbox'
            ]
        )
        page = await browser.newPage()

        await page.goto('https://2captcha.com/demo/normal', {'waitUntil': 'networkidle2'})

        image_src = await page.evaluate(
            '''() => document.querySelector("img[class='_captchaImage_rrn3u_9']").getAttribute('src')'''
        )

        image_url = f"https://2captcha.com{image_src}"

        response = requests.get(image_url)
        if response.status_code == 200:
            image_bytes = response.content
        else:
            print("Failed to fetch the image")
            await browser.close()
            return

        captcha_text = preprocess_and_ocr(image_bytes)
        print(f"Extracted Captcha Text: {captcha_text}")

        input_box = await page.waitForSelector("input[type='text']")
        await input_box.type(captcha_text)

        await page.click("button[type='submit']")

        await page.waitForSelector("p[class='_successMessage_rrn3u_1']")
        success_element = await page.querySelector("p[class='_successMessage_rrn3u_1']")
        if success_element:
            success_text = await page.evaluate(
                "(el) => el.innerText",
                success_element
            )
            print(success_text)

        else:
            print("Wrong Captcha")

        await browser.close()
    
    except Exception as e:
        print(f"Some Error occurred: {e} \n\n {traceback.format_exc()}")
        await browser.close()


asyncio.run(main())