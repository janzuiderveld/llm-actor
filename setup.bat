@echo off
setlocal enabledelayedexpansion

REM === SETTINGS ===
set REPO_URL=https://github.com/RVirmoors/llm-actor
set REPO_ZIP=https://github.com/RVirmoors/llm-actor/archive/refs/heads/main.zip
set TARGET_DIR=llm-actor
set PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=python-3.11.9-amd64.exe


echo Checking for Python...

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo Python not found on this system.
    echo Downloading Python 3.11.9 installer...

    powershell -Command ^
        "Invoke-WebRequest -Uri '%PYTHON_INSTALLER_URL%' -OutFile '%PYTHON_INSTALLER%'"

    if not exist "%PYTHON_INSTALLER%" (
        echo Failed to download Python installer.
        pause
        exit /b 1
    )

    echo Running Python installer...
    start "" "%PYTHON_INSTALLER%"

    echo.
    echo ================= IMPORTANT =================
    echo When the installer opens, make sure to enable:
    echo.
    echo       [x] Add python.exe to PATH
    echo.
    echo Without this, the installation will not work.
    echo ==============================================
    echo.
    pause

    echo Rechecking Python availability...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo Python is still not detected in PATH.
        echo Please confirm that you selected:
        echo     "Add python.exe to PATH"
        echo in the installer.
        echo.
        echo If you already installed Python, restart the script
        echo after opening a new Command Prompt window.
        pause
        exit /b 1
    )
)

echo Python detected.
echo.


REM === CLONE OR DOWNLOAD PROJECT ===
if not exist "%TARGET_DIR%" (
    echo Project directory not found. Preparing to fetch the repository...

    git --version >nul 2>&1
    if %ERRORLEVEL%==0 (
        echo Git detected. Cloning repository...
        git clone "%REPO_URL%" "%TARGET_DIR%"
    ) else (
        echo Git not found. Downloading project ZIP instead...

        powershell -Command ^
            "(New-Object System.Net.WebClient).DownloadFile('%REPO_ZIP%', 'project.zip')"

        if not exist project.zip (
            echo Failed to download repository ZIP.
            pause
            exit /b 1
        )

        echo Extracting ZIP...
        powershell -Command ^
            "Expand-Archive -LiteralPath 'project.zip' -DestinationPath '.' -Force"

        del project.zip

        for /d %%D in ("%TARGET_DIR%-main") do (
            if exist "%%D" (
                ren "%%D" "%TARGET_DIR%"
            )
        )
    )
)

echo.
echo Repository ready in "%TARGET_DIR%".
echo.

cd "%TARGET_DIR%"

REM === PYTHON ENV SETUP ===
python -m venv venv
call venv\Scripts\activate
pip install -e .

REM === EDIT .env FOR API KEYS ===
if not exist ".env" (
    copy ".env.example" ".env"
    notepad ".env"
)

REM === SOUNDDEVICE TEST ===
python -m sounddevice

REM === OPEN PROJECT SETTINGS FILE ===
notepad "BASIC_PROJECT\settings.ini"

pause