# asi-ml — Project

---

## Project Overview

This project analyzes airline passenger satisfaction based on demographic and service-related features.
The goal is to build a machine learning model that predicts whether a passenger will be **satisfied** with their flight experience.

---

## Data Source

**Dataset:**
[Airline Passenger Satisfaction Dataset — Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

**License:**
The dataset “Airline Passenger Satisfaction” by TJ Klein is available on Kaggle
 under the license type “Other (specified in description)”.
Since the author did not specify a standard open license, the dataset is used in this project for educational and non-commercial purposes only, in accordance with Kaggle’s Terms of Service (featuring CC0: Public Domain specification).
Users should credit the dataset’s author when sharing or reproducing related work.
If the dataset author later defines an explicit license, those terms shall take precedence.

**Date of Download:** October 16, 2025

**Description:**
The dataset contains information about airline passengers — including age, class, customer type, flight distance, service ratings, and overall satisfaction.
The target variable `satisfaction` has two possible values:
- `Satisfied`
- `Neutral or dissatisfied`

The goal of this project is to **predict passenger satisfaction** based on their demographic characteristics and flight experience.

**Sample size:**
A sample of 1000 rows was included in the repository
(`ASI_project_ZIMA/data/01_raw
/sample_dummy.csv
`).

---

## Selected Evaluation Metric

**Metric:** `F1-score`

**Justification:**
The dataset is **slightly imbalanced** (most passengers are not satisfied).
The **F1-score** metric balances **precision** and **recall**, providing a more reliable evaluation of model performance than simple accuracy.
It helps assess how well the classifier identifies both satisfied and dissatisfied passengers.

---

## Project Structure

## Kedro quickstart
kedro run
# lub
kedro run --pipeline second_sprint_pipeline