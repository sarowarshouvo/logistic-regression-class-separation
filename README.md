# logistic-regression-class-separation
# Performance Analysis of Logistic Regression under Controlled Class Separation and Overlap

## 📘 Overview
This repository contains the implementation and analysis for **Mini Project 2** of **MATH 5384 – Advanced Machine Learning**.

The project investigates the behavior of Logistic Regression in controlled three-dimensional classification settings using synthetically generated data. Both non-overlapping and overlapping class configurations are studied to analyze how class separation and regularization influence classification performance.

Special attention is given to misclassification error, decision boundaries, class overlap, and the effect of L2 regularization on model generalization.

---

## 🎯 Objectives

- Study Logistic Regression performance under varying class separation
- Analyze the relationship between class overlap and misclassification error
- Evaluate training and testing performance
- Investigate the effect of regularization strength
- Visualize 3D decision boundaries
- Examine the bias-variance trade-off in overlapping datasets

---

## 📊 Dataset Description

Synthetic datasets were generated in a three-dimensional feature space.

### Non-Overlapping Configuration

Two Gaussian-distributed classes:

\[
X_0 \sim N(\mu_0, I), \quad X_1 \sim N(\mu_1, I)
\]

Class centers:

\[
\mu_0 = (0,0,0), \quad \mu_1 = (5,5,5)
\]

- Samples per class: 500
- Total samples: 1000
- Covariance matrix: Identity matrix
- Produces a linearly separable dataset

---

### Distance-Based Separation Study

The second class center was varied as:

\[
\mu_1 = (d,d,d)
\]

where:

\[
d \in \{0.5, 1.0, 1.5, ..., 6.0\}
\]

This experiment analyzes how increasing separation reduces misclassification error.

---

### Overlapping Configuration

To create overlap:

- Spread parameter:

\[
\sigma = 1.8
\]

- Center distance:

\[
d = 4
\]

Class centers:

\[
\mu_0 = (0,0,0), \quad \mu_1 = (4,4,4)
\]

This produces partially overlapping Gaussian distributions.

---

## 🧠 Logistic Regression Model

The probability of class membership is modeled as:

\[
P(Y=1|X)=\sigma(\beta_0+\beta^TX)
\]

where the sigmoid activation is:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The model is trained using Maximum Likelihood Estimation (MLE).

---

## ⚙️ Regularization

L2 regularization is introduced through a Bayesian interpretation:

\[
\beta \sim N(0,\tau^2 I)
\]

Objective function:

\[
L_{MAP}(\beta)=L_{MLE}(\beta)+\frac{\lambda}{2}\|\beta\|^2
\]

Regularization strength is controlled using:

\[
C=\frac{1}{\lambda}
\]

Values tested:

\[
C \in \{0.001,0.01,0.05,0.1,0.5,1,5,10,50,100\}
\]

---

## 📈 Key Findings

- Logistic Regression achieves perfect classification for well-separated data
- Misclassification error decreases as class separation increases
- Overlapping datasets introduce unavoidable classification error
- Moderate regularization improves generalization performance
- Strong regularization reduces overfitting but increases bias
- Training and testing errors remain closely aligned, indicating stable generalization

---

## 🛠️ Technologies Used

- Python
- NumPy
- Scikit-learn
- Matplotlib
- Plotly

---

## 📂 Repository Structure

```text
├── MiniProject2_LogisticRegression.ipynb
├── MiniProject2_Report.pdf
├── figures/
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Run the notebook:

```text
MiniProject2_LogisticRegression.ipynb
```

---

## 📎 Included Contents

- Synthetic dataset generation
- Logistic Regression implementation
- Distance-based separation experiments
- Regularization analysis
- 3D visualization of decision boundaries
- Training/testing error comparisons
- Misclassification error analysis

---

## 👤 Author

**Saroar Jahan Shuba**  
M.S. Computational & Quantitative Methods  
Lamar University  
