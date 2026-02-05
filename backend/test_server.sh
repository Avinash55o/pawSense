#!/bin/bash

# PawSense Backend Testing Script

echo "🚀 Starting PawSense Backend Testing..."
echo ""

# Step 1: Check Python version
echo "📋 Step 1: Checking Python version..."
python3 --version
echo ""

# Step 2: Create virtual environment (optional but recommended)
echo "📋 Step 2: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Step 3: Install dependencies
echo "📋 Step 3: Installing dependencies..."
echo "This may take a few minutes..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""

# Step 4: Start the server
echo "📋 Step 4: Starting FastAPI server..."
echo "Server will start on http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""
echo "⚠️  First prediction will be slow (model download)"
echo "⏱️  Subsequent predictions will be fast"
echo ""

# Start uvicorn
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
