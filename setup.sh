#!/bin/bash
# Setup script for Schwab Paper Trading Agent
# Run this once credentials arrive

set -e

echo "🛠️  Wheel Trader Setup"
echo "====================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Install Python 3.9+ first."
    exit 1
fi

PYTHON=$(python3 --version | awk '{print $2}')
echo "✅ Python $PYTHON"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✅ venv activated"

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip setuptools wheel
pip install -q requests python-dateutil mibian  # mibian for Black-Scholes

echo ""
echo "📋 Required environment variables:"
echo "   SCHWAB_PAPER_ACCOUNT_ID  — your paper account ID"
echo "   SCHWAB_API_KEY            — your API key/OAuth token"
echo ""
echo "Example:"
echo "   export SCHWAB_PAPER_ACCOUNT_ID='ACC123456'"
echo "   export SCHWAB_API_KEY='your-api-key-here'"
echo ""
echo "Once set, run:"
echo "   python3 schwab_paper_trader.py"
echo ""
echo "✅ Setup complete!"
