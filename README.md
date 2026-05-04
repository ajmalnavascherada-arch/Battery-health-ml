# 🔋 Machine Learning-Based Lithium-Ion Battery Health Classification

## 📌 Overview

This project presents a **machine learning pipeline for classifying lithium-ion battery health** using real-world cycling data. The goal is to simulate automated battery cell sorting for second-life applications, aligning with current research in sustainable battery recycling and diagnostics.

The model processes battery discharge data, extracts meaningful features, and classifies cells into health categories such as **Healthy, Aged, and Degraded**.

---

## 🎯 Objectives

* Develop a data-driven approach for **battery health classification**
* Perform **feature engineering** based on electrochemical behavior
* Train and evaluate machine learning models
* Analyze **feature importance** to understand degradation indicators

---

## 📂 Dataset

This project uses the **NASA Lithium-Ion Battery Dataset**, which includes:

* Charge/discharge cycles
* Voltage, current, and capacity measurements
* Degradation behavior over time

> Note: Dataset files are excluded from this repository due to size.
> You can download them from NASA's Prognostics Center of Excellence.

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Extract discharge cycle data from `.mat` files
* Compute statistical summaries (mean, min, max)

### 2. Feature Engineering

* Derived features:

  * Voltage range
  * Power proxy (voltage × current)
* These features approximate internal battery behavior and degradation trends

### 3. Labeling Strategy

Battery health is classified based on capacity:

* **Healthy**
* **Aged**
* **Degraded**

### 4. Model Development

* Random Forest Classifier
* Train-test split for evaluation

### 5. Evaluation Metrics

* Accuracy
* Confusion Matrix
* Feature Importance Analysis

---
## Results

- Random Split Accuracy: 97.06%
- Cross-Battery Accuracy: 92.86%
- Without Capacity: 76.79%




## 📈 Visual Outputs

The model generates:

* Feature importance plot
* Confusion matrix
* Performance metrics

All outputs are stored in the `results/` directory.

---

## Key Insights

- Capacity is the strongest predictor of degradation
- Model generalizes well across batteries
- Degradation signals exist beyond capacity

---

## 🚀 Future Work

* Integration of **Electrochemical Impedance Spectroscopy (EIS)** data
* Advanced models (e.g., Gradient Boosting, Neural Networks)
* Multi-battery dataset generalization
* Real-time battery sorting systems

---

## 🛠️ Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* Matplotlib, Seaborn
* SciPy

---

## 📁 Project Structure

```
battery-health-ml/
│
├── data/                # Dataset (not included)
├── src/                 # Source code
├── notebooks/           # Exploratory analysis
├── results/             # Output plots and metrics
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/main.py
```

---

## 📬 Contact

**Ajmal Navas Cherada**
📍 Magdeburg, Germany
🔗 LinkedIn: https://www.linkedin.com/in/ajmal-navas-cherada-224151171
💻 GitHub: https://github.com/ajmalnavascherada-arch

---

## ⭐ Acknowledgment

This project is inspired by research in **battery diagnostics, machine learning, and sustainable energy systems**, particularly in the context of second-life battery applications.

---
