@echo off
setlocal enabledelayedexpansion

REM === SETTINGS ===
set REPO_URL=https://github.com/RVirmoors/llm-actor
set REPO_ZIP=https://github.com/RVirmoors/llm-actor/archive/refs/heads/main.zip
set TARGET_DIR=llm-actor
set "ASSET_DIR=llm-actor\assets"
set PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=python-3.11.9-amd64.exe

REM === CHECK FOR MSVC++ REDISTRIBUTABLE (x64) ===
set "NEED_VCREDIST="

for /f "skip=2 tokens=3" %%B in (
    'reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed 2^>nul'
) do (
    if /i NOT "%%B"=="0x1" set "NEED_VCREDIST=1"
)

if defined NEED_VCREDIST (
    echo Downloading VC++ runtime...
    set "VCREDIST_URL=https://aka.ms/vc14/vc_redist.x64.exe"
    set "VCREDIST_FILE=vc_redist.x64.exe"
)

rem Perform PowerShell download outside the parentheses
if defined NEED_VCREDIST powershell -Command "Invoke-WebRequest -Uri '%VCREDIST_URL%' -OutFile '%VCREDIST_FILE%'"

if defined NEED_VCREDIST (
    if not exist "%VCREDIST_FILE%" (
        echo Failed to download VC++ redistributable.
        pause
        exit /b 1
    )
)



REM === CHECK FOR PYTHON 3.10+ ===
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


REM === DOWNLOAD KOKORO MODEL FILES ===

if not exist "%ASSET_DIR%" mkdir "%ASSET_DIR%"

echo Downloading kokoro-v1.0.onnx...
powershell -Command ^
  "Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx' -OutFile '%ASSET_DIR%\kokoro-v1.0.onnx'"

if not exist "%ASSET_DIR%\kokoro-v1.0.onnx" (
    echo Failed to download kokoro-v1.0.onnx
    pause
    exit /b 1
)

echo Downloading voices-v1.0.bin...
powershell -Command ^
  "Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin' -OutFile '%ASSET_DIR%\voices-v1.0.bin'"

if not exist "%ASSET_DIR%\voices-v1.0.bin" (
    echo Failed to download voices-v1.0.bin
    pause
    exit /b 1
)

echo Kokoro model files downloaded successfully.
echo.


cd "%TARGET_DIR%"

REM === PYTHON ENV SETUP ===
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
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
    echo Switch to your web browser to see two new tabs.
    echo Please register or log in, create API keys,
    echo and paste them into *_API_KEY inside .env
    echo.
    start "" "https://console.deepgram.com/"
    start "" "https://console.groq.com/"
    notepad ".env"
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
