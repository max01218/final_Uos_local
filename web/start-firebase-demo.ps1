# Firebase Demo Startup Script for PowerShell
# This script sets up and starts the web application with Firebase integration

Write-Host "🚀 Starting ICD-11 Mental Health Assistant with Firebase..." -ForegroundColor Green

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if we're in the correct directory
if (-not (Test-Path "package.json")) {
    Write-Error "package.json not found. Please run this script from the web directory."
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Node.js installation
Write-Status "Checking Node.js installation..."
try {
    $nodeVersion = node --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Node.js version: $nodeVersion"
    } else {
        Write-Error "Node.js is not installed. Please install Node.js first."
        Write-Status "Visit: https://nodejs.org/"
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Error "Node.js is not installed. Please install Node.js first."
    Write-Status "Visit: https://nodejs.org/"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check npm installation
Write-Status "Checking npm installation..."
try {
    $npmVersion = npm --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "npm version: $npmVersion"
    } else {
        Write-Error "npm is not installed. Please install npm first."
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Error "npm is not installed. Please install npm first."
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for .env.local file
Write-Status "Checking Firebase configuration..."
if (-not (Test-Path ".env.local")) {
    Write-Warning ".env.local file not found!"
    Write-Status "Please create .env.local file with your Firebase configuration:"
    Write-Host ""
    Write-Host "NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key_here"
    Write-Host "NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project_id.firebaseapp.com"
    Write-Host "NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id"
    Write-Host "NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project_id.appspot.com"
    Write-Host "NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id"
    Write-Host "NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id"
    Write-Host ""
    Write-Status "See FIREBASE_SETUP_GUIDE.md for detailed instructions."
    Write-Host ""
    $continue = Read-Host "Do you want to continue without Firebase configuration? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Status "Setup cancelled. Please configure Firebase first."
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Warning "Running without Firebase - authentication will not work!"
} else {
    Write-Success "Firebase configuration found"
}

# Install dependencies
Write-Status "Installing dependencies..."
try {
    npm install
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Dependencies installed successfully"
    } else {
        Write-Error "Failed to install dependencies"
        Read-Host "Press Enter to exit"
        exit 1
    }
} catch {
    Write-Error "Failed to install dependencies"
    Read-Host "Press Enter to exit"
    exit 1
}

# Run type check
Write-Status "Running TypeScript type check..."
try {
    npm run type-check
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Type check passed"
    } else {
        Write-Warning "Type check failed - continuing anyway"
    }
} catch {
    Write-Warning "Type check failed - continuing anyway"
}

# Start development server
Write-Status "Starting development server..."
Write-Host ""
Write-Success "🎉 Application starting..."
Write-Status "Access the application at: http://localhost:3000"
Write-Host ""
Write-Status "Firebase Features Available:"
Write-Status "✅ User registration and login"
Write-Status "✅ Secure user data storage"
Write-Status "✅ Conversation history persistence"
Write-Status "✅ Real-time authentication state"
Write-Host ""
Write-Status "Testing the Firebase Integration:"
Write-Status "1. Visit http://localhost:3000"
Write-Status "2. Click 'Get Started' or 'Sign In'"
Write-Status "3. Create a new account or sign in"
Write-Status "4. Start a conversation in the chat"
Write-Status "5. Check Firebase Console to see saved data"
Write-Host ""
Write-Status "Press Ctrl+C to stop the server"
Write-Host ""

# Start the development server
npm run dev 