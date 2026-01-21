# Model Card — AutoGluon Satisfaction Classifier (Draft)

## 1. Problem & Use-case
The goal of this project is to automatically classify airline customer satisfaction
("satisfied" vs "neutral or dissatisfied") using structured survey data.
This model is created for **educational purposes** within the ASI Machine Learning course and
should not be used for real-world decision-making without additional validation.

## 2. Data Description
Raw data source:
`data/01_raw/sample_raw.csv`

Data after preprocessing:
`data/03_primary/clean_data.parquet`

Key dataset information:
- ~795 rows after cleaning
- 24 input features
- Target: `satisfaction` → binary (1 = satisfied, 0 = neutral/dissatisfied)

Preprocessing steps:
- stripping column names
- categorical encoding (Gender, Customer Type, Travel Type, Class, Satisfaction)
- replacing zero-ratings with per-column means
- filling delay-related NaNs with 0
- outlier filtering using z-score
- 80/20 train-test split with `random_state=10`

Feature groups:
- **Demographics**: Age, Gender, Customer Type
- **Flight parameters**: Class, Flight Distance
- **Delays**: Arrival/Departure delay
- **Service ratings**: Wifi, Seat comfort, Food & drink, Online boarding, Cleanliness, etc.

## 3. Evaluation Metrics
Primary metric: **ROC AUC**
Rationale:
- suitable for binary classification
- robust metric independent of threshold
- well-supported by AutoGluon

Additional stored metrics:
- error rate
- ROC AUC on test set
- training time

## 4. Model Performance (Production Candidate)
After running 5 full AutoGluon experiments, the best model was selected based on test ROC AUC.

**Selected production model:**
`ag_model:v2`

**Best run:**
`decent-surf-3`

**Performance summary (v2):**
- **ROC AUC:** ~0.97
- **Test error rate:** 0
- **Training time:** ~200–250 seconds
- **Model type:** AutoGluon WeightedEnsemble_L2
  (ensemble of LightGBM, ExtraTrees, CatBoost)

**Why v2 was selected?**
- highest ROC AUC
- stable metric spread across folds
- lowest training instability among runs
- consistent feature importance distribution

## 5. Limitations
- Small dataset → limited generalization to real airline operations
- Ratings are subjective and may include individual bias
- Zero-value imputation may distort distributions
- Outlier removal may hide real-world extreme cases
- No hyperparameter sweeps or fairness checks included
- Not production-grade preprocessing (no pipelines for missing values, drift, etc.)

## 6. Ethical Considerations & Risks
- The model uses demographic features like Gender and Age → **risk of bias**
- Satisfaction prediction should *not* be used for:
  - customer prioritization
  - pricing decisions
  - operational automation affecting individuals
- Dataset origin and representativeness are unclear
- No fairness audit or bias analysis performed

## 7. Versioning Information
**Production model version:**
`ag_model:v2`

**Corresponding run ID:**
`decent-surf-3`

**Artifact path:**
`s27634-pjatk/asi-ml-satisfaction/ag_model:v2`

**Local model file:**
`tmp_kedro/satisfaction-prediction/data/06_models/ag_production.pkl`

**W&B project link:**
https://wandb.ai/s27634-pjatk/asi-ml-satisfaction

## 8. Next steps (recommended)
- Add hyperparameter tuning (W&B Sweeps)
- Perform fairness and bias analysis
- Add SHAP explanations and feature stability checks
- Train on significantly larger dataset
- Introduce CI/CD process for automated model promotion
- Deploy inference endpoint (optional)
