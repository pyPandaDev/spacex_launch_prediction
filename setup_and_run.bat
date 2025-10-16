@echo off
echo ========================================
echo SpaceX Launch Prediction - Setup
echo ========================================
echo.

echo [1/3] Activating virtual environment...
call venv\Scripts\activate
echo ✓ Virtual environment activated
echo.

echo [2/3] Installing dependencies...
pip install -r requirements.txt --quiet
echo ✓ Dependencies installed
echo.

echo [3/3] Running automated analysis pipeline...
python run_analysis.py
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Launch Streamlit app: streamlit run app/streamlit_app.py
echo   2. Explore notebooks in Jupyter
echo   3. Make predictions via CLI: python src/predict.py --help
echo.
pause
