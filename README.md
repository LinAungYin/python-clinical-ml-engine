# python-clinical-ml-engine
A pure Python implementation of Logistic Regression and K-Fold Cross-Validation from scratch. Built to demonstrate a deep understanding of core ML mathematics, gradient descent, and Z-score standardization for clinical risk prediction without relying on black-box libraries like scikit-learn.

# Clinical Risk Prediction Engine (Python) 🐍

**Project:** Computational Deep Phenotyping for Clinical Risk Prediction  
**Context:** PhD Application Technical Showcase (NUS Yong Loo Lin School of Medicine)  
**Author:** Lin Aung Yin

## 📌 Overview
This repository contains a pure Python implementation of a machine learning pipeline designed for clinical risk prediction. As part of a polyglot engineering showcase (featuring Python, Rust, and Java), this specific project demonstrates a deep, first-principles understanding of the mathematics underlying predictive modeling.

By deliberately avoiding black-box libraries like `scikit-learn` or `numpy`, this code proves a fundamental mastery of:
* **Batch Gradient Descent & Calculus:** Implementing partial derivatives and weight updates from scratch.
* **Linear Algebra & Z-Score Standardization:** Preventing feature dominance by scaling data to $\mu=0$ and $\sigma=1$.
* **Robust Evaluation:** Building a custom K-Fold Cross-Validation engine to test model stability.

## 🧬 Clinical Narrative
In translational medicine, interpretability and mathematical transparency are just as critical as predictive accuracy. This simulation generates synthetic "deep phenotypes" (Age, Glucose Level, Comorbidity Score, Systolic BP) and trains a Logistic Regression model to classify patients into High/Low risk categories, mimicking the rigorous data pipelines required for modern clinical research.

## 🚀 How to Run
Because this is written in pure Python, there are no external dependencies to install.
1. Clone the repository.
2. Run the script directly from your terminal:
   ```bash
   python main.py
