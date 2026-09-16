# Medical Appointment No-Show Risk Predictor

An end-to-end machine-learning project for estimating the probability that a patient will miss a scheduled medical appointment. The repository includes exploratory analysis, feature engineering, model comparison, threshold selection, explainability, and a Streamlit interface for interactive scoring.

> **Important:** This is an educational portfolio project, not a clinical decision-support system. Predictions must not be used as the sole basis for patient care, access, or scheduling decisions.

## Highlights

- Compares Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SMOTE + Random Forest, and XGBoost.
- Evaluates class-imbalanced performance with stratified cross-validation, ROC-AUC, PR-AUC, and F1.
- Uses a tuned operating threshold instead of assuming a 0.50 cutoff.
- Provides global and individual prediction explanations with SHAP.
- Includes a Streamlit application and serialized model artifacts for immediate demonstration.

## Results

| Model | ROC-AUC | PR-AUC | F1 (no-show) |
| --- | ---: | ---: | ---: |
| Random Forest | 0.776 | 0.333 | 0.387 |
| Random Forest + SMOTE | 0.775 | 0.343 | 0.374 |
| XGBoost (deployed) | 0.710 | 0.248 | 0.277 |

The Streamlit application serves the XGBoost classifier. A Random Forest model is retained for SHAP explanations.

## Demo

| Global importance | Feature impact distribution | Individual explanation |
| --- | --- | --- |
| ![SHAP bar chart](assets/shap_bar_rf.png) | ![SHAP beeswarm](assets/shap_beeswarm_rf.png) | ![SHAP waterfall](assets/shap_waterfall_patient.png) |

## Quick Start

```bash
git clone https://github.com/LikithaDudala/medical-appointment-noshow-predictor.git
cd medical-appointment-noshow-predictor
python -m venv .venv
```

Activate the environment, install dependencies, then start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproduce the Analysis

Open `Medical_Appointments_NoShow.ipynb` in Jupyter and run the notebook to reproduce the exploration, training, evaluation, and artifact generation steps.

```bash
jupyter notebook Medical_Appointments_NoShow.ipynb
```

## Data and Privacy

The repository includes `medical-appointments-no-show-en.csv` so the notebook can be reproduced. Treat it as sensitive healthcare-related data: do not use it to identify individuals, make clinical decisions, or redistribute it outside the permissions governing the source data. Remove or replace it before using this project in a setting with stricter privacy, legal, or institutional requirements.

## Repository Structure

```text
.
├── app.py                              # Streamlit prediction interface
├── Medical_Appointments_NoShow.ipynb   # EDA, modeling, and evaluation workflow
├── medical-appointments-no-show-en.csv # Source dataset
├── models/                              # Serialized models and feature metadata
├── assets/                              # SHAP visualizations used in this README
├── requirements.txt                     # Python dependencies
└── .github/workflows/ci.yml             # Syntax validation on pushes and pull requests
```

## Technical Stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SHAP, Matplotlib, Seaborn, Streamlit, and Jupyter.

## Limitations

- The data represents a specific historical population and may not generalize to another clinic or region.
- No model-monitoring, fairness assessment, or retraining process is implemented.
- Performance metrics are experimental estimates, not production guarantees.
- Serialized pickle artifacts should be loaded only from this trusted repository.
