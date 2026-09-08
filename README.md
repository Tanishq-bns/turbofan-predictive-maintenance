# ✈️ AeroPredict: Turbofan Engine Predictive Maintenance & Risk Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://turbofan-predictive-maintenance-sr3eazfywgn6e52sda5ris.streamlit.app/)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20(98%25%20Acc)-success.svg)](https://xgboost.readthedocs.io/)

🌐 **Live Web Application**: [https://turbofan-predictive-maintenance-sr3eazfywgn6e52sda5ris.streamlit.app/](https://turbofan-predictive-maintenance-sr3eazfywgn6e52sda5ris.streamlit.app/)

---

An end-to-end Machine Learning and Predictive Maintenance platform trained on the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset. 

The application classifies engine degradation stages in real time, forecasts remaining operational cycles until the next degradation transition, and issues automated maintenance alerts via an interactive **Streamlit** dashboard.

---

## 🚀 Key Highlights & Performance

* **Degradation Stage Classification**: Multi-class **XGBoost Classifier** achieving **98% overall test accuracy** across all 5 degradation stages:
  * Stage 0: Nominal Operation
  * Stage 1: Initial Degradation
  * Stage 2: Moderate Degradation
  * Stage 3: Critical Degradation
  * Stage 4: Severe Failure Imminent
* **Remaining Time Forecasting**: **XGBoost Regressor** predicting operational cycles remaining before transitioning to the next failure state.
* **Composite Risk Engine**: Dynamic health risk score computed by fusing class failure probability $P(\text{Stage 4})$ with remaining cycle estimates.
* **Interactive Web Dashboard**: Built with **Streamlit** & **Plotly** featuring fleet lifecycle scrubbing, live sensor injection simulation, and automated critical maintenance alerts.

---

## 📁 Repository Structure

```text
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore configuration
├── models/
│   ├── xgb_classifier.joblib   # Trained XGBoost Degradation Classifier (98% acc)
│   ├── xgb_regressor.joblib    # Trained XGBoost Regressor
│   ├── meta_info.joblib        # Sensor descriptions & risk normalization thresholds
│   └── sample_engines.csv      # Telemetry slice across sample test units
├── ML_project01.ipynb          # Comprehensive research & training notebook
├── ML_project01.py             # Script version of the ML pipeline
└── README.md                   # Project documentation
```

---

## 🛠️ Local Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tanishq-bns/turbofan-predictive-maintenance.git
   cd turbofan-predictive-maintenance
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit web dashboard**:
   ```bash
   streamlit run app.py
   ```
   *The app will automatically launch in your browser at `http://localhost:8501`.*

---

## 📊 Dataset Reference
* **Source**: NASA Prognostics Center of Excellence (PCoE)
* **Dataset**: C-MAPSS Turbofan Engine Degradation Simulation (FD001–FD004)
* **Sensory Channels**: 21 continuous sensor measurements + 3 operational settings.
