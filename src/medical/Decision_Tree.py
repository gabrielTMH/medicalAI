import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tfidf


model = tfdf.keras.RandomForestModel()
model.fit(tfidf.create_dataset(tfidf.train_df))
model.compile(metrics=["accuracy"])
print(model.evaluate(tfidf.create_dataset(tfidf.test_df)))