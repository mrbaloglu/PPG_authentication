import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import torch
import keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback

import mlflow
import mlflow.keras

from Processor import Processor
from models import CNN

def scheduler(epoch, lr):
    
    if epoch % 200 == 0 and epoch > 0:
        return lr * 0.5
    else:
        return lr


if __name__ == "__main__":
    trn_data = pd.read_excel("train8.xlsx", sheet_name="Sheet1")
    tst_data = pd.read_excel("test8.xlsx", sheet_name="Sheet1")

    print("Data is read.")

    NUM_SCALES = 64
    MOTHER_WAVELETS = ["mexh", "morl", "gaus5"]
    OLD_DATA = True
    mlflow.start_run()
    mlflow.log_param("num scales", NUM_SCALES)
    mlflow.log_param("Mother Wavelets", ", ".join(MOTHER_WAVELETS))
    mlflow.log_param("old data", OLD_DATA)

    processor = Processor(trn_data, tst_data, "ID", ["Serial", "ID", "Total"], 
                          mother_wavelets=MOTHER_WAVELETS,
                          num_scales=NUM_SCALES,
                          old_data=OLD_DATA)
    
    xtrain, ytrain = processor.prepare_training_data(return_tensors="np")
    xtest, ytest = processor.prepare_test_data(return_tensors="np")

    print("Data preparation and processing is completed.")

    xtrain = np.swapaxes(xtrain, 1, 3)
    xtrain = np.swapaxes(xtrain, 1, 2)

    xtest = np.swapaxes(xtest, 1, 3)
    xtest = np.swapaxes(xtest, 1, 2)

    print(f"xtrain: {xtrain.shape}, ytrain: {ytrain.shape}")

    N_CONV_FILTERS = [128, 64]
    CONV_WINDOW_SIZES = [5, 5]
    N_DENSE_UNITS = [128, 64]


    cnn = CNN(
        input_shape=xtrain.shape[1:],
        classes = ytrain.shape[1],
        num_conv_filters=N_CONV_FILTERS,
        conv_window_sizes=CONV_WINDOW_SIZES,
        num_dense_units=N_DENSE_UNITS
    )

    print(cnn.summary())
    adam = Adam(learning_rate=5e-4)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=20, min_lr=1e-6)
    es = EarlyStopping(monitor="val_loss", patience = 30, restore_best_weights=True)
    cb = keras.callbacks.LearningRateScheduler(scheduler)
    cnn.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = cnn.fit(xtrain, ytrain, validation_data=(xtest, ytest), callbacks=[rlr, cb, es, TqdmCallback(verbose=0)], batch_size=32, epochs=1000, verbose=0)
    print("Training of CNN is completed.")

    mlflow.log_params({f"num_conv_filters_{i}": N_CONV_FILTERS[i] for i in range(len(N_CONV_FILTERS))})
    mlflow.log_params({f"conv_window_size_{i}": CONV_WINDOW_SIZES[i] for i in range(len(CONV_WINDOW_SIZES))})
    mlflow.log_params({f"num_dense_units_{i}": N_DENSE_UNITS[i] for i in range(len(N_DENSE_UNITS))})

    plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical cross entropy")
    plt.yscale("log")
    plt.legend()
    plt.savefig("charts/Losses.png")
    plt.close("all")

    plt.plot(np.array(hist.history["val_accuracy"]) * 100, label="Validation Accuracy")
    plt.plot(np.array(hist.history["accuracy"]) * 100, label="Train Accuracy")
    plt.title("Accuracies (%)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("charts/Accuracies.png")
    plt.close("all")

    pred = cnn.predict(xtest)
    pred_ = pred.argmax(axis=1)

    print(classification_report(ytest.argmax(axis=1), pred_, digits=6))
    acc = accuracy_score(ytest.argmax(axis=1), pred_)
    f1_micro = f1_score(ytest.argmax(axis=1), pred_, average="micro")
    f1_macro = f1_score(ytest.argmax(axis=1), pred_, average="macro")
    mlflow.log_metric("Test Accuracy", acc)
    mlflow.log_metric("Test Micro F1", f1_micro)
    mlflow.log_metric("Test Macro F1", f1_macro)

    cm = confusion_matrix(ytest.argmax(axis=1), pred_, normalize="pred")
    sns.heatmap(cm)
    plt.title("Confusion Matrix")
    plt.savefig("charts/Confusion Matrix.png")
    plt.close("all")

    mlflow.log_artifact("charts")
    mlflow.keras.log_model(cnn, "cnn_model")
    mlflow.end_run()
