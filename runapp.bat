@echo off
echo 🚀 Starting Hugging Face Streamlit App...

:: Activate virtual environment
call venv\Scripts\activate

:: Set Hugging Face token (replace with yours if needed)
set HF_API_TOKEN=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

:: Run seed script (comment out if you don’t need to reload data every time)
python scripts\seed.py

:: Run Streamlit app
streamlit run streamlit_app.py

pause

