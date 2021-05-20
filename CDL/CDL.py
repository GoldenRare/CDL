import tensorflow as tf
import pandas as pd
from keras import Input
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

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

MAX_MODELS = 10
counter = models_created = 0
while counter < MAX_MODELS:
    # Collecting and splitting training data
    training_data = pd.read_excel("C:/Users/Golden/Documents/CDL/Training_Data.xlsx", header=None, usecols="A:P")
    #   training_data =
    #   pd.read_excel("C:/Users/Golden/Documents/CDL/Training_Data_Wins_Only.xlsx", header=None, usecols="A:P")
    training_data = training_data.to_numpy()

    training_target = pd.read_excel("C:/Users/Golden/Documents/CDL/Training_Data.xlsx", header=None, usecols="Q")
    #   training_target =
    #   pd.read_excel("C:/Users/Golden/Documents/CDL/Training_Data_Wins_Only.xlsx", header=None, usecols="Q")
    training_target = training_target.to_numpy()
    label_encoder = LabelEncoder()
    training_target = label_encoder.fit_transform(training_target)
    training_target = to_categorical(training_target)

    training_data, testing_data, training_target, testing_target = train_test_split(training_data, training_target,
                                                                                    test_size=0.1)
    #############################################

    #   scalar = MinMaxScaler()
    #   scalar.fit(training_data)
    #   training_data = scalar.transform(training_data)
    #   print(training_data, training_target, testing_data.shape, testing_target.shape)

    # Callbacks and checkpoint path
    checkpoint_path = 'CDL_Model' + str(counter + 1) + '_ACCURACY_200_175'
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
    ############

    # Creating and training model
    models_created += 1
    model = Sequential()
    model.add(Input(shape=(16,)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(175, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x=training_data, y=training_target, epochs=100, verbose=0,
                        callbacks=[model_checkpoint, early_stopping], validation_data=(testing_data, testing_target))
    ##############################

    # Save only the good models
    past_accuracies = history.history['val_accuracy']
    if past_accuracies[len(past_accuracies) - 11] >= 0.50:
        counter += 1
        print('HIT', past_accuracies[len(past_accuracies) - 11], counter, 'EPOCH', len(past_accuracies) - 10,
              'LOSS', history.history['val_loss'][len(past_accuracies) - 11])
    else:
        print('MISSED', past_accuracies[len(past_accuracies) - 11], counter + 1)

print(MAX_MODELS / models_created)
#   print(model.predict(np.array([[1.12, 1.08, 1.04, 1.00, 1, 1, 1, 1, 0.98, 0.9, 0.89, 0.88, 0.4, 0.3, 0.6, 0.1]])))
#   print(model.predict(np.array([[1.199663016, 1.106365159, 1.076865672, 1.010492333, 0.842105263, 0.71875, 0.655172414,
#                               0.714285714, 1.080246914, 0.985388128, 0.932735426, 0.835538752, 0.25, 0.44, 0.428571429,
#                               0.1875]])))
#print(model.predict(np.array([[1.010403121, 0.987630208, 0.938596491, 0.932330827, 0.619047619, 0.6, 0.5, 0.545454545,
#                               1.0515625, 1.009836066, 0.979495268, 0.761506276, 0.388888889, 0.433333333, 0.473684211,
#                               0.444444444]])))
