import json
import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

DATA_PATH = os.path.join("..", "data")


def plot_history(hist):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist.get("train_loss"), label='Train', color="gray")
    plt.plot(hist.get("val_loss"), label='Val', color="red")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("train_acc1"), label="Train", color="gray")
    plt.plot(hist.get("train_acc3"), color="gray")
    plt.plot(hist.get("train_acc5"), color="gray")
    plt.plot(hist.get("val_acc1"), label='Val', color="red")
    plt.plot(hist.get("val_acc3"), color="red")
    plt.plot(hist.get("val_acc5"), color="red")
    plt.legend()
    plt.title("Accuracy")

    plt.show()


if __name__ == "__main__":

    # Get all configurations
    # f = open("config.json")
    # list_config = json.load(f)
    # model_names = list_config.get("model_names")

    model_names = [
        "full_04_mnv3_fine_tuned"
    ]

    for model_name in model_names:

        history_path = os.path.join(*[DATA_PATH, "saved_models", f"{model_name}_history.pkl"])
        history = pickle.load(open(history_path, "rb"))
        plot_history(history)

        print(model_name)
        print(f"Stopped at: {len(history.get('val_loss'))}")
        print(np.round(np.max(history.get("val_acc1")) * 100, 2))
        print(np.round(np.max(history.get("val_acc3")) * 100, 2))
        print(np.round(np.max(history.get("val_acc5")) * 100, 2))

