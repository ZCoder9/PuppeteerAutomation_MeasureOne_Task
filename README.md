# PuppeteerAutomation_MeasureOne_Task

This repository contains a Python script (`web_automation.py`) that performs an automated web task. It uses Tesseract OCR, so you must provide the path to the Tesseract binary via an environment variable.

> ⚠️ The instructions below refer to `web_automation.py`.

## Setup

1. **Clone the repository** (if you haven't already):
   ```powershell
   git clone <repo-url>
   cd PuppeteerAutomation_MeasureOne_Task
   ```

2. **Install Tesseract OCR** on your system:
   - Download the Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
   - Run the installer and follow the installation wizard
   - Note the installation path (default is usually `C:\Program Files\Tesseract-OCR`)

3. **Create a `.env` file** in the project root and set the `TESSERACT_PATH` environment variable:
   ```env
   TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
   Replace the value with the actual installation path of Tesseract on your system.

4. **Create a virtual environment** and activate it:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1     # on PowerShell
   # or
   .\venv\Scripts\activate.bat    # on cmd.exe
   ```

5. **Install dependencies** from `requirements.txt`:
   ```powershell
   pip install -r requirements.txt
   ```

## Running the script

With the virtual environment activated and the `.env` file configured, execute the automation script:

```powershell
python web_automation.py
```

## Notes

- Ensure Tesseract is installed on your system and that the path in `.env` points to the executable.
- Keep the virtual environment activated whenever running the script to use the installed dependencies.

---
