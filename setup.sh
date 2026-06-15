#!/bin/bash
# Quick setup for the Predict Spec Assistant
set -e
echo "Setting up Predict Spec Assistant..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo ""
echo "Setup complete. Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. export ANTHROPIC_API_KEY='sk-ant-...'"
echo "  3. python -m src.agent        # interactive"
echo "  4. python -m evals.run_evals  # run the eval suite"
