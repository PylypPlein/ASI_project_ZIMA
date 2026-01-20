# Model Card — AutoGluon Customer Satisfaction Classifier

## Problem & Intended Use
This model predicts airline customer satisfaction (“satisfied” vs. “neutral or dissatisfied”) based on flight-related features and service ratings.
The intended use is to support analytics teams in identifying factors that drive satisfaction and to help improve operational and service decisions.
The model is **not** intended for high-risk decisions or automated actions affecting customers.

## Data (source, license, size, PII = no)
- **Source:** Airline Passenger Satisfaction Dataset — Kaggle. https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
- **Size:** 1000 samples
- **License:** The dataset “Airline Passenger Satisfaction” by TJ Klein is available on Kaggle under the license type “Other (specified in description)”. Since the author did not specify a standard open license, the dataset is used in this project for educational and non-commercial purposes only, in accordance with Kaggle’s Terms of Service (featuring CC0: Public Domain specification). Users should credit the dataset’s author when sharing or reproducing related work. If the dataset author later defines an explicit license, those terms shall take precedence.
- **PII:** No personally identifiable information.
- **Features** include:
  - Passenger demographics: Gender, Age, Customer Type
  - Travel metadata: Type of Travel, Class, Flight Distance
  - Service ratings: wifi, boarding, entertainment, service, cleanliness, comfort
  - Operational metrics: departure & arrival delays
- **Target:** Binary class — `satisfied` / `neutral or dissatisfied`.

## 3. Metrics (Main + Secondary)

### **Main metric**
- **ROC AUC:** **0.973**

The model achieves high discriminative performance and reliably ranks satisfied vs. dissatisfied passengers.

### **Secondary metrics**
- **Test error rate:** 0.0
  *(AutoGluon sometimes logs 0 for missing external test set — interpret with caution)*
- **Training time:** 60.47 seconds
- **Feature importance (Top 5):**
  Available as a W&B artifact under `feature_importance_top5`.

## Limitations & Risks
- Dataset represents a single airline → limited generalization across carriers or seasons.
- Customer ratings are subjective and may contain noise or bias.
- Model may underperform on rare passenger groups or unusual flights.
- Class imbalance may affect F1 or recall for minority class.
- The model should not be used to evaluate individuals — it predicts satisfaction trends, not customer value.

## Ethics & Risk Mitigation
- Use the model only as a decision-support tool, not for automated customer impact.
- Regularly retrain and monitor for data drift.
- Validate model performance on new demographic or operational segments.
- Keep training data, code, and environment versioned to ensure reproducibility.
- Communicate limitations clearly to stakeholders.

## Versioning
- **W&B Run:** https://wandb.ai/s27634-pjatk/asi-ml-satisfaction?nw=nwusers27634
- **Model Artifact:** ag_model:v2
- **Code Commit:** `05f9a57` Sprint 3 – AutoGluon + W&B integration
- **Data Version:** `[sample.csv](../data/01_raw/sample.csv)`
- **Environment:** Python 3.11, AutoGluon 1.x, scikit-learn 1.5

## Links
- **W&B Comparison Dashboard:** https://wandb.ai/s27634-pjatk/asi-ml-satisfaction?nw=nwusers27634
