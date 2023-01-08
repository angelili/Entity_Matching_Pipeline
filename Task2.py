from gc import callbacks
import pandas as pd
import numpy as np
import seaborn as sb
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks


data_prep = __import__('01_DataPrep')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)


##################################################################################################
#************************* Classification Base Model  **********************************************#
##################################################################################################

# model callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=0)  # val_loss


callbacks = [early_stopping]

# Prediction of Wine Type


def BaseClassificationModel(input_shape, layer1, layer2, epochs, batchsize, optimizer):
    # Import `Sequential` from `keras.models`
    from keras.models import Sequential

    # Import `Dense` from `keras.layers`
    from keras.layers import Dense
    X_train, X_test, y_train, y_test, X_val, y_val = data_prep.split_dataset_classification()

    # Initialize the constructor
    model = Sequential()

    # Add an input layer
    model.add(Dense(layer1, activation='relu', input_shape=input_shape))

    # Add one hidden layer
    model.add(Dense(layer2, activation='relu'))

    # Add an output layer
    model.add(Dense(1, activation='sigmoid'))

    # Model output shape
    model.output_shape

    # Model summary
    model.summary()

    # Model config
    model.get_config()

    # List all weight tensors
    model.get_weights()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    # Training Model
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batchsize, verbose=1, validation_data=(X_val, y_val), callbacks=callbacks, shuffle=True)
    history_frame = pd.DataFrame(history.history)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.ylim([0, max([max(loss), max(val_loss)])])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    # plt.show() Plot Commented for Checking runtime

    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))

    return history_frame, model


##################################################################################################
#************************* Regression Base Model **********************************************#
##################################################################################################


def BaseModelRegression():
    EPOCHS = 300
    BATCH_SIZE = 2 ** 8  # 256

    X_train, X_valid, y_train, y_valid = data_prep.regressionPreprocess()
    input_shape = [X_train.shape[1]]

    # Training Configuration

    # Define linear model
    model = keras.Sequential([
        layers.Dense(1, input_shape=input_shape),
    ])

    # Compile in the optimizer and loss function
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # Fit model (and save training history)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    # Convert the training history to a dataframe
    history_frame = pd.DataFrame(history.history)

    # Plot training history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.ylim([0, max([max(loss), max(val_loss)])])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Minimum Validation Loss: {:0.4f}".format(
        history_frame['val_loss'].min()))
    print("Minimum Validation MAE (mean absolute error): {:0.4f}".format(
        history_frame['val_mae'].min()))
