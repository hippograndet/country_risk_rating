
# OECD Country Risk Rating Prediction with Machine Learning

> End-to-end data science project using public macroeconomic APIs to predict OECD country risk ratings.

## 🔍 Overview

This project builds a **reproducible machine learning pipeline** that predicts **OECD country risk ratings** using **macroeconomic indicators** from World Bank public data.

It focuses on **real-world data engineering, feature design, and applied ML**.

---

## 🧠 What This Demonstrates

- API-driven data ingestion (World Bank, OECD)
- Feature engineering on macroeconomic indicators
- Efficient and Transparent Data Pipeline (sklearn)
- Training and evaluating ML models (xgboost, mlflow)
- Clean, modular, and reproducible project structure

---

## 📊 Data Sources (Public)

- **World Bank** – fiscal, monetary, balance-of-payments, growth, development, demographic indicators  
- **OECD** – country risk ratings (target variable)

<!-- All data is fetched programmatically via APIs. -->

---

## ⚙️ Pipeline (High-Level)

APIs → Cleaning & Alignment → Feature Engineering
→ Model Training → Evaluation

Multiple ML models are explored, including linear baselines and tree-based methods, with cross-validation throughout.

---

## 🗂️ Repository Structure

├── data/           # Raw and processed datasets
├── notebooks/      # Exploration, evaluation and visualization jupyter notebooks
├── src/            # Ingestion, preprocessing, training python scripts
├── models/         # Logged models, using mlflow
├── results/        # Figures and metrics
├── requirements.txt
└── README.md

Notebooks explain why decisions were made.
src/ contains the reproducible pipeline code.

---

## 🚀 Quick Start

git clone https://github.com/hippograndet/country_risk_rating.git
cd country_risk_rating
pip install -r requirements.txt
python src/train.py
Optional exploration:
jupyter notebook notebooks/

Virtual Environmentx
python -m venv .venv
source .venv/bin/activate

---

## 📈 Results (Summary)

Macroeconomic indicators capture meaningful structure in risk ratings
Tree-based models outperform linear baselines
Feature importance aligns with economic intuition
Results are exploratory, not financial advice.

---

## ⚠️ Notes

Country risk ratings include subjective expert judgment
Data availability varies across countries and years
Models capture correlation, not causation

---

## 👤 Author

Hippolyte Grandet
MSc Artificial Intelligence – University of Edinburgh
Interests: Applied ML, Data Science, Economic Modeling