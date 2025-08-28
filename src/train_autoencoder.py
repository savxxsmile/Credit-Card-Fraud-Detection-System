import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Load data
data = pd.read_csv("data/creditcard.csv")
X = data.drop("Class", axis=1).values

input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

checkpoint = ModelCheckpoint("models/autoencoder.h5", save_best_only=True, monitor='loss')

autoencoder.fit(X, X, epochs=50, batch_size=256, shuffle=True, callbacks=[checkpoint])
