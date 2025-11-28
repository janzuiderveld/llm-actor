@echo off
setlocal enabledelayedexpansion

REM === SETTINGS ===
set REPO_URL=https://github.com/RVirmoors/llm-actor
set REPO_ZIP=https://github.com/RVirmoors/llm-actor/archive/refs/heads/main.zip
set TARGET_DIR=llm-actor
set PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=python-3.11.9-amd64.exe

echo Checking for Python...
setlocal enabledelayedexpansion

python --version >nul 2>&1
if errorlevel 1 (
    set NEED_PYTHON_INSTALL=1
) else (
    REM Extract version string safely
    for /f "tokens=2 delims= " %%v in ('python --version') do set PY_VER=%%v

    REM Split Major.Minor.Patch
    for /f "tokens=1,2,3 delims=." %%a in ("!PY_VER!") do (
        set PY_MAJOR=%%a
        set PY_MINOR=%%b
    )

    REM Ensure values exist before comparing
    if not defined PY_MAJOR set NEED_PYTHON_INSTALL=1
    if not defined PY_MINOR set NEED_PYTHON_INSTALL=1

    REM Compare version numbers
    if not defined NEED_PYTHON_INSTALL (
        if !PY_MAJOR! LSS 3 (
            set NEED_PYTHON_INSTALL=1
        ) else if !PY_MAJOR!==3 if !PY_MINOR! LSS 10 (
            set NEED_PYTHON_INSTALL=1
        )
    )
)


if defined NEED_PYTHON_INSTALL (
    echo.
    echo Python 3.10+ is required. Installing Python 3.11.9...

    powershell -Command ^
        "Invoke-WebRequest -Uri '%PYTHON_INSTALLER_URL%' -OutFile '%PYTHON_INSTALLER%'"

    if not exist "%PYTHON_INSTALLER%" (
        echo Failed to download Python installer.
        pause
        exit /b 1
    )

    start "" "%PYTHON_INSTALLER%"

    echo.
    echo ================= IMPORTANT =================
    echo Enable the option:
    echo.
    echo      [x] Add python.exe to PATH
    echo.
    echo The installer will not work without this.
    echo =============================================
    echo.
    pause

    echo Rechecking Python availability...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo Python is still not detected in PATH. You probably need to run this setup again.
        echo Please verify you enabled "Add python.exe to PATH".
        pause
        exit /b 1
    )
)

echo Python detected.
echo.



REM === CLONE OR DOWNLOAD PROJECT ===
if not exist "%TARGET_DIR%" (
    echo Project directory not found. Preparing to fetch the repository...

    where git >nul 2>&1
    if errorlevel 1 (
        echo Git not found. Using ZIP download...
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
    ) else (
        echo Git found. Cloning repository...
        git clone "%REPO_URL%" "%TARGET_DIR%"
    )
)

echo.
echo Repository ready in "%TARGET_DIR%".
echo.

cd "%TARGET_DIR%"

REM === PYTHON ENV SETUP ===
python -m venv venv
call venv\Scripts\activate
pip install .

REM === CHECK .env ===
if not exist ".env" (
    copy ".env.example" ".env"
)

REM === CHECK FOR DEFAULT DEEPGRAM API KEY ===
set NEED_DEEPGRAM_SETUP=

for /f "usebackq tokens=* delims=" %%L in (".env") do (
    echo %%L | findstr /C:"DEEPGRAM_API_KEY=your-deepgram-api-key" >nul
    if not errorlevel 1 (
        set NEED_DEEPGRAM_SETUP=1
    )
)

if defined NEED_DEEPGRAM_SETUP (
    echo.
    echo API key[s] not yet configured.
    echo Opening .env for editing and launching signup pages...
    echo.
    echo Please register or log in, create API keys,
    echo and paste them into *_API_KEY inside .env
    echo.
    notepad ".env"
    start "" "https://console.deepgram.com/"
    start "" "https://console.groq.com/"
)

echo.
echo Note the input and output device indices 
echo from the sounddevice below, and edit them into
echo BASIC_PROJECT\settings.ini as needed.
echo.

REM === SOUNDDEVICE TEST ===
python -m sounddevice
notepad "BASIC_PROJECT\settings.ini"

echo.
echo Setup complete. You can now run the project
echo by entering the llm_actor/ folder and executing:
echo.
echo     run.bat
echo.
echo To change any settings, run this setup again
echo or edit the .env and settings.ini files directly.
echo.

pause