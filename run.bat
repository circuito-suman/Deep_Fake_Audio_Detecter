@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Running Deepfake Audio Detection (Synthetic Demo)...
python main.py --mode synthetic
pause
