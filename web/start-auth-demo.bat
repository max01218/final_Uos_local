@echo off
chcp 65001 >nul

REM Authentication System Demo Startup Script for Windows
echo 🚀 Starting ICD-11 Mental Health Assistant with Authentication System
echo ================================================================

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed. Please install Node.js first.
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ npm is not installed. Please install npm first.
    pause
    exit /b 1
)

REM Check if we're in the web directory
if not exist "package.json" (
    echo ❌ Please run this script from the web directory
    pause
    exit /b 1
)

REM Install dependencies if node_modules doesn't exist
if not exist "node_modules" (
    echo 📦 Installing dependencies...
    npm install
    if %errorlevel% neq 0 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ✅ Dependencies already installed
)

REM Check if TypeScript is available
npx tsc --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing TypeScript...
    npm install -g typescript
)

REM Run type check
echo 🔍 Running type check...
npx tsc --noEmit
if %errorlevel% neq 0 (
    echo ⚠️  TypeScript errors found, but continuing...
) else (
    echo ✅ TypeScript check passed
)

REM Start the development server
echo 🌐 Starting development server...
echo ================================================================
echo 📱 The application will be available at: http://localhost:3000
echo 🔐 Authentication page: http://localhost:3000/auth
echo 💬 Chat page (protected): http://localhost:3000/chat
echo.
echo 🧪 To test the authentication system:
echo 1. Visit http://localhost:3000
echo 2. Click 'Get Started' to go to the auth page
echo 3. Register a new account
echo 4. Login with your credentials
echo 5. Access the protected chat functionality
echo.
echo 📚 For more information, see AUTH_README.md
echo ================================================================

REM Start the development server
npm run dev 