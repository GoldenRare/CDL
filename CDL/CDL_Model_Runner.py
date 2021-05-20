import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

stats = pd.read_excel("C:/Users/Golden/Documents/CDL/Stats.xlsx", usecols="A:P", skiprows=38)
stats = stats.to_numpy()

MAX_MODELS = 10
models_imported = 0
model_predictions = []
while models_imported < MAX_MODELS:
    models_imported += 1
    model = load_model('CDL_Model' + str(models_imported) + '_ACCURACY_200_175')
    model_predictions.append(model.predict(stats))

counter = 0
score = 1
prediction_scores = [0, 0, 0, 0, 0, 0]
while counter < MAX_MODELS:
    index_sorted = np.argsort(model_predictions[counter])
    print(model_predictions[counter])
    print(index_sorted)
    for i in index_sorted:
        for j in i:
            prediction_scores[j] += score
            score += 1
            if score > 6:
                score = 1
    counter += 1

print(prediction_scores)
