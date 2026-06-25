import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import keras
from keras import layers, Model
import torch

B, T, F = 32, 50, 10
inp = layers.Input(shape=(T, F))
x = layers.Dense(1, activation=None)(inp)
x = layers.Activation("sigmoid")(x)
model = Model(inp, x)
model.compile(optimizer="adam", loss="binary_crossentropy")

X = np.random.randn(B, T, F).astype(np.float32)
y = np.random.randint(0, 2, (B, T, 1)).astype(np.float32)

model.fit(X, y, epochs=1, batch_size=32, verbose=1)