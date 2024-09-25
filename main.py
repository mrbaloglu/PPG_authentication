import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, roc_curve

import torch
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


from Processor import Processor
from models import CNN


if __name__ == "__main__":
    trn_data = pd.read_excel("total8.xlsx", sheet_name="Trainset")
    tst_data = pd.read_excel("total8.xlsx", sheet_name="Testset")

    processor = Processor(trn_data, tst_data, "ID", ["Serial", "ID", "Total"], mother_wavelets=["mexh", "morl", "gaus5"])
    xtrain, ytrain = processor.prepare_training_data(return_tensors="np")
    xtest, ytest = processor.prepare_test_data(return_tensors="np")

    xtrain = np.swapaxes(xtrain, 1, 3)
    xtrain = np.swapaxes(xtrain, 1, 2)

    xtest = np.swapaxes(xtest, 1, 3)
    xtest = np.swapaxes(xtest, 1, 2)

    cnn = CNN(input_shape=xtrain.shape[1:], classes = ytrain.shape[1])
    print(cnn.summary())
    adam = Adam(learning_rate=1e-3)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6)
    es = EarlyStopping(monitor="val_loss", patience = 20, restore_best_weights=True)
    cnn.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = cnn.fit(xtrain, ytrain, validation_data=(xtest, ytest), callbacks=[rlr, es], batch_size=128, epochs=200)
    print("Training of CNN is completed.")

    plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.plot(hist.history["loss"], label="Train Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Caterical cross entropy")
    plt.yscale("log")
    plt.legend()
    plt.savefig("Losses.png")
    plt.close("all")

    plt.plot(np.array(hist.history["val_accuracy"]) * 100, label="Validation Accuracy")
    plt.plot(np.array(hist.history["accuracy"]) * 100, label="Train Accuracy")
    plt.title("Accuracies (%)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("Accuracies.png")
    plt.close("all")

    pred = cnn.predict(xtest)
    pred_ = pred.argmax(axis=1)

    print(classification_report(ytest.argmax(axis=1), pred_))
    cm = confusion_matrix(ytest.argmax(axis=1), pred_, normalize="pred")
    sns.heatmap(cm)
    plt.title("Confusion Matrix")
    plt.savefig("Confusion Matrix.png")
    plt.close("all")

    cnn.save("cnn_model_.keras")
