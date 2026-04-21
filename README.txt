DESCRIPTION
------------------
Our project democratizes analytics through providing pitch outcome predictions based on pitch attributes (e.g., spin rate, speed, location) allowing players and coaches to optimize their training. We use a sequential machine learning pipeline and deliver outputs in an interactive, user-friendly web application. Users select a pitch type, provide pitch attribute inputs, and view the probability of pitch outcomes facing an MLB batter.


INSTALLATION
------------------

1. Create and activate a virtual environment

    `python3 -m venv .venv`

    macOS/Linux:
    `source .venv/bin/activate`

    Windows:
    `.venv\Scripts\activate`


2. Install dependencies

    `pip install -r requirements.txt`


3. Download the data. This will download and clean MLB Statcast data from 2021-2025 and save it to `data/data.csv`. Note: This may take several minutes to complete.

    `python download_mlb_data.py`


4. (Optional) Train the models. The XGBoost models (used by the UI) are included in the models folder, so you do not need to run this script to run the application. The KNN and Random Forest models are not used by the UI and were strictly used for evaluation.

    `python model-tests/xgb_tuning.py`
    `python model-tests/knn_model.py`
    `python model-tests/random_forest.py`


5. Run the app. The app will open in your browser at http://localhost:8501.

    `streamlit run app.py`

EXECUTION
------------------

Once the app is running:
- Use the "Pitch Controls" expander to adjust pitch parameters
- View the pitch visualization and outcome probabilities
- Toggle "Use Batted Ball Model" for detailed in-play predictions
