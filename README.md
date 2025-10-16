# asi-ml — Project

Template repository for the project

---

## Project Overview

This project analyzes airline passenger satisfaction based on demographic and service-related features.
The goal is to build a machine learning model that predicts whether a passenger will be **satisfied** with their flight experience.

---

## Data Source

**Dataset:**
[Airline Passenger Satisfaction Dataset — Kaggle](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

**License:**
CC BY-NC-SA 4.0 (Creative Commons Attribution–NonCommercial–ShareAlike 4.0)

**Date of Download:** October 16, 2025

**Description:**
The dataset contains information about airline passengers — including age, class, customer type, flight distance, service ratings, and overall satisfaction.
The target variable `satisfaction` has two possible values:
- `Satisfied`
- `Neutral or dissatisfied`

The goal of this project is to **predict passenger satisfaction** based on their demographic characteristics and flight experience.

**Sample size:**
A sample of 500 rows was included in the repository
(`data/01_raw/airline_satisfaction_sample.csv`).

---

## Selected Evaluation Metric

**Metric:** `F1-score`

**Justification:**
The dataset is **slightly imbalanced** (most passengers are not satisfied).
The **F1-score** metric balances **precision** and **recall**, providing a more reliable evaluation of model performance than simple accuracy.
It helps assess how well the classifier identifies both satisfied and dissatisfied passengers.

---

## Project Structure
