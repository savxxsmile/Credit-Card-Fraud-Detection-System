# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using a combination of **Logistic Regression** and an **Autoencoder (Neural Network)**.  
The project includes model training, data preprocessing, and a **Streamlit web app** for real-time fraud detection.

---

## 📂 Project Structure

```

.
├── data/                    # Dataset folder
│   └── creditcard.csv       # Credit card transactions dataset
│
├── models/                  # Pre-trained models & scalers
│   ├── autoencoder.h5
│   ├── logistic\_regression.pkl
│   └── scaler.pkl
│
├── notebooks/               # Jupyter notebooks for exploration
│   └── fraud\_detection.ipynb
│
├── src/                     # Training & utility scripts
│   ├── train\_autoencoder.py
│   ├── train.py
│   └── utils.py
│
├── app.py                   # Streamlit app for fraud detection
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md

````

---

## 🚀 Features
- **Logistic Regression model** trained on scaled features.
- **Autoencoder** trained to reconstruct normal transactions; anomalies are flagged as fraud.
- **Dynamic thresholding** based on reconstruction error of non-fraud data.
- **Ensemble prediction** combining Logistic Regression and Autoencoder results.
- **Interactive Streamlit app** with:
  - Dataset exploration
  - Fraud prediction for custom transactions
  - Dashboard metrics for quick insights

---

## ⚙️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# create virtual environment
python -m venv .venv
source .venv/bin/activate   # on Linux/Mac
.venv\Scripts\activate      # on Windows

# install dependencies
pip install -r requirements.txt
````

---

## ▶️ Usage

### 1. Run the Streamlit App

```bash
streamlit run app.py
```

### 2. Explore Dataset

* View class distribution
* Inspect first rows of dataset

### 3. Predict Transactions

* Enter transaction details (auto-filled with sample values)
* Get predictions from:

  * Logistic Regression
  * Autoencoder
  * Final Ensemble Decision

---

## 📊 Models

* **Autoencoder (Keras, TensorFlow)**: Trained to minimize reconstruction error on non-fraud transactions.
* **Logistic Regression (scikit-learn)**: Trained on scaled features.
* **Scaler**: StandardScaler used for feature normalization.

Models are stored inside `models/`.

---

## 📓 Notebook

The notebook in `notebooks/fraud_detection.ipynb` contains exploratory data analysis (EDA), preprocessing steps, and training process.

---

## ✅ Requirements

See [`requirements.txt`](requirements.txt).
Key dependencies include:

* TensorFlow / Keras
* Scikit-learn
* Pandas, NumPy
* Streamlit
* Joblib

---

## 📈 Future Improvements

* Try advanced models (XGBoost, Random Forest, Deep Learning).
* Implement model monitoring & drift detection.
* Add APIs for integration with financial systems.

---

## 🙌 Acknowledgements

Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

