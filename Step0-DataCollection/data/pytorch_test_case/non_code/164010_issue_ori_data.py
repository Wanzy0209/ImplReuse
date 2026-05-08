python3.10 -m venv
source venv/bin/activate
pip install -r docs/requirements.txt
python -c "import matplotlib.pyplot"  # fails with ImportError