#!/bin/bash
# ============================================================
#  setup.sh — macOS / Linux one-click project setup
#  Run once from your project folder:
#    cd ~/Desktop/air_quality_project
#    chmod +x setup.sh && ./setup.sh
# ============================================================

set -e

echo ""
echo "======================================================"
echo " Nairobi Air Quality Project — macOS/Linux Setup"
echo "======================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi
echo "[OK] Python: $(python3 --version)"

echo "[1/5] Creating virtual environment (venv)..."
python3 -m venv venv

echo "[2/5] Activating virtual environment..."
source venv/bin/activate

echo "[3/5] Upgrading pip..."
pip install --upgrade pip --quiet

echo "[4/5] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "[5/5] Checking .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[WARNING] .env created from .env.example"
    echo "          DATA_URL is already set. Edit if needed."
else
    echo "[OK] .env already exists."
fi

echo ""
echo "======================================================"
echo " Setup complete!"
echo ""
echo " To start working:"
echo "   1. Open VS Code:"
echo "        code ."
echo "   2. Activate venv in terminal:"
echo "        source venv/bin/activate"
echo "   3. Run the notebook:"
echo "        jupyter notebook data_cleaning_eda.ipynb"
echo "   4. Launch the dashboard:"
echo "        streamlit run dashboard.py"
echo "======================================================"
echo ""
