# 🧪 Superconductor Critical Temperature Prediction

## 📌 Project Overview
This notebook implements a comprehensive machine learning pipeline to predict the **critical temperature ($T_c$)** of superconducting materials based on their chemical properties. We utilize the **UCI Superconductor Dataset** and employ advanced techniques including **physics-informed feature engineering**, **Residual Neural Networks (ResMLP)**, and **Bayesian Hyperparameter Optimization (Optuna)**.

## 🚀 Key Features
- **Physics-Informed Feature Engineering**: Creation of 30+ new features based on domain knowledge (BCS theory, electron-phonon coupling, lattice strain).
- **Deep Learning Architecture**: Implementation of a **Residual MLP (ResMLP)** with skip connections to enable deeper network training without vanishing gradients.
- **Automated Tuning**: Use of **Optuna** for Bayesian optimization of hyperparameters (network depth, width, learning rate, dropout).
- **Model Interpretability**: Feature importance analysis using **Permutation Importance** to understand which chemical properties drive superconductivity.

## 📂 Dataset
- **Source**: [UCI Machine Learning Repository - Superconductivty Data](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data)
- **Size**: 21,263 samples with 81 initial features.
- **Target**: Critical Temperature (Kelvin).

## 🛠️ Workflow
1.  **Data Loading & Exploration**: Statistical analysis and visualization of the target variable and features.
2.  **Feature Engineering**:
    *   *Coefficient of Variation* features
    *   *Mismatch/Diversity Indices* (atomic radius, mass, etc.)
    *   *BCS Theory* proxies (Debye frequency, density of states)
3.  **Preprocessing**: Standardization (StandardScaler) and Train/Test splitting.
4.  **Baseline Model**: Training a custom Residual MLP in PyTorch.
5.  **Hyperparameter Optimization**: Tuning the ResMLP using Optuna to minimize validation loss.
6.  **Final Evaluation**: Comparing baseline vs. optimized models and analyzing prediction errors across temperature ranges.

## 📦 Requirements
The following Python libraries are required:
*   `pandas`, `numpy` (Data manipulation)
*   `matplotlib`, `seaborn` (Visualization)
*   `scikit-learn` (Preprocessing & Metrics)
*   `torch` (Deep Learning)
*   `optuna` (Hyperparameter Optimization)

---
*Run the cells sequentially to reproduce the analysis and results.*
