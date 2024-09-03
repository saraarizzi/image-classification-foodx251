import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils import MobileNetV3Small, get_transforms, evaluate

DATA_PATH = os.path.join("..", "data")
VAL_DATA_PATH = os.path.join(DATA_PATH, "val")
VAL_DEG_DATA_PATH = os.path.join(DATA_PATH, "val_degraded")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(name):

    if name == "mobilenet":
        return MobileNetV3Small(251)
    elif name == "mobilenet_freeze":
        return MobileNetV3Small(251, freeze=True)


if __name__ == "__main__":

    # Get best model configuration
    f = open("config.json")
    list_config = json.load(f)
    best_model = "04_mnv3_fine_tuned"

    # Degradation on train data percentage
    models = [
        f"full_{best_model}"
    ]

    # Read HyperParams
    d = list_config.get(best_model)

    for model_name in models:

        # Dataset
        _, val_transforms = get_transforms(d.get("data_aug"))
        val_degraded_ds = ImageFolder(VAL_DEG_DATA_PATH, transform=val_transforms)

        # Loader
        val_deg_loader = DataLoader(val_degraded_ds, batch_size=d.get("batch_size"), shuffle=True)

        # Model instance
        m = get_model(d.get("model")).to(DEVICE)

        # Define paths for saving/loading model and history
        model_path = os.path.join(*[DATA_PATH, "saved_models", f"{model_name}.pt"])
        history_path = os.path.join(*[DATA_PATH, "saved_models", f"{model_name}_history.pkl"])

        # Load model
        m.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(DEVICE)))
        history = pickle.load(open(history_path, "rb"))

        # Print history plots
        val_acc1, val_acc3, val_acc5 = evaluate(m, val_deg_loader, DEVICE)

        print(
            f"Accuracy@1 --> {np.round(val_acc1*100, 2)}\n"
            f"Accuracy@3 --> {np.round(val_acc3*100, 2)}\n"
            f"Accuracy@5 --> {np.round(val_acc5*100, 2)}\n"
        )