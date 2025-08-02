@echo off
chcp 65001 >nul

REM Firebase Demo Startup Script for Windows
REM This script sets up and starts the web application with Firebase integration

echo 🚀 Starting ICD-11 Mental Health Assistant with Firebase...

REM Check if we're in the correct directory
if not exist "package.json" (
    echo [ERROR] package.json not found. Please run this script from the web directory.
    pause
    exit /b 1
)

REM Check Node.js installation
echo [INFO] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed. Please install Node.js first.
    echo [INFO] Visit: https://nodejs.org/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('node --version') do set NODE_VERSION=%%i
echo [SUCCESS] Node.js version: %NODE_VERSION%

REM Check npm installation
echo [INFO] Checking npm installation...
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm is not installed. Please install npm first.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('npm --version') do set NPM_VERSION=%%i
echo [SUCCESS] npm version: %NPM_VERSION%

REM Check for .env.local file
echo [INFO] Checking Firebase configuration...
if not exist ".env.local" (
    echo [WARNING] .env.local file not found!
    echo [INFO] Please create .env.local file with your Firebase configuration:
    echo.
    echo NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key_here
    echo NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project_id.firebaseapp.com
    echo NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id
    echo NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project_id.appspot.com
    echo NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
    echo NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id
    echo.
    echo [INFO] See FIREBASE_SETUP_GUIDE.md for detailed instructions.
    echo.
    set /p CONTINUE="Do you want to continue without Firebase configuration? (y/N): "
    if /i not "%CONTINUE%"=="y" (
        echo [INFO] Setup cancelled. Please configure Firebase first.
        pause
        exit /b 1
    )
    echo [WARNING] Running without Firebase - authentication will not work!
) else (
    echo [SUCCESS] Firebase configuration found
)

REM Install dependencies
echo [INFO] Installing dependencies...
npm install
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [SUCCESS] Dependencies installed successfully

REM Run type check
echo [INFO] Running TypeScript type check...
npm run type-check
if errorlevel 1 (
    echo [WARNING] Type check failed - continuing anyway
) else (
    echo [SUCCESS] Type check passed
)

REM Start development server
echo [INFO] Starting development server...
echo.
echo [SUCCESS] 🎉 Application starting...
echo [INFO] Access the application at: http://localhost:3000
echo.
echo [INFO] Firebase Features Available:
echo [INFO] ✅ User registration and login
echo [INFO] ✅ Secure user data storage
echo [INFO] ✅ Conversation history persistence
echo [INFO] ✅ Real-time authentication state
echo.
echo [INFO] Testing the Firebase Integration:
echo [INFO] 1. Visit http://localhost:3000
echo [INFO] 2. Click 'Get Started' or 'Sign In'
echo [INFO] 3. Create a new account or sign in
echo [INFO] 4. Start a conversation in the chat
echo [INFO] 5. Check Firebase Console to see saved data
echo.
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Start the development server
npm run dev 