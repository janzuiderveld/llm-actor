@echo off
setlocal enabledelayedexpansion

REM === SETTINGS ===
set REPO_URL=https://github.com/RVirmoors/llm-actor
set REPO_ZIP=https://github.com/RVirmoors/llm-actor/archive/refs/heads/main.zip
set TARGET_DIR=llm-actor
set "ASSET_DIR=llm-actor\assets"

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


set PYTHON_INSTALLER_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=python-3.11.9-amd64.exe

echo Checking for compatible Python (3.10 or 3.11)...
setlocal EnableExtensions EnableDelayedExpansion

set PYTHON_EXE=

REM === Try Python launcher (preferred method) ===
py -0p >nul 2>&1
if not errorlevel 1 (
for /f "tokens=1,*" %%a in ('py -0p 2^>nul') do (
    echo %%a | findstr /C:"-V:3.11" >nul
    if not errorlevel 1 set "PYTHON_EXE=%%b"

    if not defined PYTHON_EXE (
        echo %%a | findstr /C:"-V:3.10" >nul
        if not errorlevel 1 set "PYTHON_EXE=%%b"
    )
)
)

REM === Fallback: check default python in PATH ===
if not defined PYTHON_EXE (
    python --version >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=2 delims= " %%v in ('python --version') do set PY_VER=%%v

        for /f "tokens=1,2 delims=." %%a in ("!PY_VER!") do (
            set PY_MAJOR=%%a
            set PY_MINOR=%%b
        )

        if "!PY_MAJOR!"=="3" if "!PY_MINOR!"=="11" set PYTHON_EXE=python
        if "!PY_MAJOR!"=="3" if "!PY_MINOR!"=="10" set PYTHON_EXE=python
    )
)

REM === Install Python if nothing usable found ===
if not defined PYTHON_EXE (
    echo.
    echo No compatible Python found. Installing Python 3.11.9...

    powershell -Command ^
        "Invoke-WebRequest -Uri '%PYTHON_INSTALLER_URL%' -OutFile '%PYTHON_INSTALLER%'"

    if not exist "%PYTHON_INSTALLER%" (
        echo Failed to download Python installer.
        pause
        exit /b 1
    )

start "" /wait "%PYTHON_INSTALLER%" /quiet ^
    InstallAllUsers=0 ^
    PrependPath=1 ^
    Include_launcher=1 ^
    AssociateFiles=0 ^
    Shortcuts=0 ^
    Include_pip=1
    
echo Waiting for system to register Python...

REM Give Windows time to register PATH / launcher
timeout /t 5 /nobreak >nul

REM Force refresh environment variables for current session
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set "USER_PATH=%%B"
set "PATH=%PATH%;%USER_PATH%"

REM Re-check Python
    py -0p >nul 2>&1
    if errorlevel 1 (
        echo Python launcher not available yet.
        echo Please run this script again and it should work.
        pause
        exit /b 1
    )

    for /f "tokens=1,2,*" %%a in ('py -0p 2^>nul') do (
        echo %%a %%b %%c | findstr /C:"-V:3.11" >nul
        if not errorlevel 1 set PYTHON_EXE=%%c
    )

    if not defined PYTHON_EXE (
        echo Python 3.11 installation not detected yet.
        echo Please run this script again and it should work.
        pause
        exit /b 1
    )
)

echo.
echo Using Python: %PYTHON_EXE%
echo.

endlocal & set PYTHON_EXE=%PYTHON_EXE%



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

if exist "%ASSET_DIR%\kokoro-v1.0.onnx" (
    echo Kokoro ONNX model already exists, skipping download.
) else (
    echo This project uses Kokoro for local text-to-speech.
    echo You can choose between Kokoro or Deepgram TTS in settings.ini later.
    echo.
    echo Downloading kokoro-v1.0.onnx...
    powershell -Command ^
      "Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx' -OutFile '%ASSET_DIR%\kokoro-v1.0.onnx'"

    if not exist "%ASSET_DIR%\kokoro-v1.0.onnx" (
        echo Failed to download kokoro-v1.0.onnx
        pause
        exit /b 1
    )
)
if exist "%ASSET_DIR%\voices-v1.0.bin" (
    echo Kokoro voices file already exists, skipping download.
) else (
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
)

echo Setting up the project Python environment...
cd "%TARGET_DIR%"

REM === PYTHON ENV SETUP ===
"%PYTHON_EXE%" -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install .

REM === CHECK .env ===
if not exist ".env" (
    copy ".env.example" ".env"
)

REM === CHECK FOR DEEPGRAM API KEY ===
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
    echo.
    echo Switch to your web browser to see two new tabs.
    echo Please register or log in, create API keys and copy them,
    echo then switch to Notepad where the .env file is open,
    echo and paste them into the relevant *_API_KEY fields.
    echo.
    echo When you're done, save and close the notepad window.
    echo.
    start "" "https://console.deepgram.com/"
    start "" "https://console.groq.com/"
    notepad ".env"
)

echo.
echo Note the input and output device indices 
echo from the list below, and edit them into
echo BASIC_PROJECT\settings.ini as needed.
echo.
echo When you're done, save and close the notepad window.
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
