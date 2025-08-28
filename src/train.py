import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import joblib

df = pd.read_csv("data/creditcard.csv")
X = df.drop("Class", axis=1).values
y = df["Class"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_res, y_res)

# Save Logistic Regression
joblib.dump(lr_model, "models/logistic_regression.pkl")

input_dim = X_scaled.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="linear")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

X_train_ae = X_scaled[y == 0]
autoencoder.fit(X_train_ae, X_train_ae, epochs=50, batch_size=256, validation_split=0.1, verbose=1)

autoencoder.save("models/autoencoder.h5")
