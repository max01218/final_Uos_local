#!/bin/bash

# Firebase Demo Startup Script
# This script sets up and starts the web application with Firebase integration

echo "🚀 Starting ICD-11 Mental Health Assistant with Firebase..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the web directory."
    exit 1
fi

# Check Node.js installation
print_status "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js first."
    print_status "Visit: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
print_success "Node.js version: $NODE_VERSION"

# Check npm installation
print_status "Checking npm installation..."
if ! command -v npm &> /dev/null; then
    print_error "npm is not installed. Please install npm first."
    exit 1
fi

NPM_VERSION=$(npm --version)
print_success "npm version: $NPM_VERSION"

# Check for .env.local file
print_status "Checking Firebase configuration..."
if [ ! -f ".env.local" ]; then
    print_warning ".env.local file not found!"
    print_status "Please create .env.local file with your Firebase configuration:"
    echo ""
    echo "NEXT_PUBLIC_FIREBASE_API_KEY=your_api_key_here"
    echo "NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your_project_id.firebaseapp.com"
    echo "NEXT_PUBLIC_FIREBASE_PROJECT_ID=your_project_id"
    echo "NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your_project_id.appspot.com"
    echo "NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id"
    echo "NEXT_PUBLIC_FIREBASE_APP_ID=your_app_id"
    echo ""
    print_status "See FIREBASE_SETUP_GUIDE.md for detailed instructions."
    echo ""
    read -p "Do you want to continue without Firebase configuration? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Setup cancelled. Please configure Firebase first."
        exit 1
    fi
    print_warning "Running without Firebase - authentication will not work!"
else
    print_success "Firebase configuration found"
fi

# Install dependencies
print_status "Installing dependencies..."
if npm install; then
    print_success "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Run type check
print_status "Running TypeScript type check..."
if npm run type-check; then
    print_success "Type check passed"
else
    print_warning "Type check failed - continuing anyway"
fi

# Start development server
print_status "Starting development server..."
echo ""
print_success "🎉 Application starting..."
print_status "Access the application at: http://localhost:3000"
echo ""
print_status "Firebase Features Available:"
print_status "✅ User registration and login"
print_status "✅ Secure user data storage"
print_status "✅ Conversation history persistence"
print_status "✅ Real-time authentication state"
echo ""
print_status "Testing the Firebase Integration:"
print_status "1. Visit http://localhost:3000"
print_status "2. Click 'Get Started' or 'Sign In'"
print_status "3. Create a new account or sign in"
print_status "4. Start a conversation in the chat"
print_status "5. Check Firebase Console to see saved data"
echo ""
print_status "Press Ctrl+C to stop the server"
echo ""

# Start the development server
npm run dev 