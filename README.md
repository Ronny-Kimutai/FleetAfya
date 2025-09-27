# FleetAfya

FleetAfya is a synthetic data and machine learning project focused on electric vehicle (EV) battery health modeling for African fleet conditions. It includes:

- Synthetic data generation for EV battery packs, reflecting African climate and usage patterns
- Machine learning pipeline using XGBoost to predict battery State of Health (SoH)
- Model evaluation and output saving
- Streamlit app for interactive exploration (if present)

## Project Structure

- `Model.ipynb`: Jupyter notebook with data generation, model training, and evaluation
- `requirements.txt`: Python dependencies
- `streamlit_app.py`: Streamlit web app
- `soh_xgb_africa.joblib`: Saved XGBoost model
- `scaler_africa.joblib`: Saved scaler
- `synthetic_africa_soh.csv`: Generated synthetic dataset

## Features

- Africa-specific synthetic EV battery data generator
- XGBoost regression model for SoH prediction
- Model evaluation (MAE, R2)
- Output saving for model, scaler, and dataset
- Streamlit app for visualization and user interaction

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook (`Model.ipynb`) to generate data and train the model
4. (Optional) Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- streamlit (optional)

## Usage

- Modify parameters in `Model.ipynb` to simulate different fleet scenarios
- Use the Streamlit app for interactive exploration

## License

This project is for educational and research purposes.
