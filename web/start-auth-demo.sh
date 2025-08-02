#!/bin/bash

# Authentication System Demo Startup Script
echo "🚀 Starting ICD-11 Mental Health Assistant with Authentication System"
echo "================================================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check if we're in the web directory
if [ ! -f "package.json" ]; then
    echo "❌ Please run this script from the web directory"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed successfully"
else
    echo "✅ Dependencies already installed"
fi

# Check if TypeScript is available
if ! npx tsc --version &> /dev/null; then
    echo "📦 Installing TypeScript..."
    npm install -g typescript
fi

# Run type check
echo "🔍 Running type check..."
npx tsc --noEmit
if [ $? -ne 0 ]; then
    echo "⚠️  TypeScript errors found, but continuing..."
else
    echo "✅ TypeScript check passed"
fi

# Start the development server
echo "🌐 Starting development server..."
echo "================================================================"
echo "📱 The application will be available at: http://localhost:3000"
echo "🔐 Authentication page: http://localhost:3000/auth"
echo "💬 Chat page (protected): http://localhost:3000/chat"
echo ""
echo "🧪 To test the authentication system:"
echo "1. Visit http://localhost:3000"
echo "2. Click 'Get Started' to go to the auth page"
echo "3. Register a new account"
echo "4. Login with your credentials"
echo "5. Access the protected chat functionality"
echo ""
echo "📚 For more information, see AUTH_README.md"
echo "================================================================"

# Start the development server
npm run dev 