import json
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from utils import MobileNetV3Small, EarlyStopping, show_history, train, get_transforms, get_degraded_data

DATA_PATH = os.path.join("..", "data")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
VAL_DATA_PATH = os.path.join(DATA_PATH, "val")
VAL_DEG_DATA_PATH = os.path.join(DATA_PATH, "val_degraded")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = True


def get_model(name):

    if name == "mobilenet":
        return MobileNetV3Small(251)
    elif name == "mobilenet_freeze":
        return MobileNetV3Small(251, freeze=True)


if __name__ == "__main__":

    # Saved models folder
    os.makedirs(os.path.join(DATA_PATH, "saved_models"), exist_ok=True)

    # Get best model configuration
    f = open("config.json")
    list_config = json.load(f)
    model_name = "04_mnv3_fine_tuned" # best model on validation
    # model_name = "deg_06_mnv3_fine_tuned"

    # Degradation on train data percentage
    runs = [
        # (None, f"full_{model_name}"),
        # (0.15, f"fifteen_{model_name}"),
        # (0.30, f"thirty_{model_name}"),
        # (0.45, f"forty_five_{model_name}"),
        # (0.60, f"sixty_{model_name}"),
        (0.90, f"ninety_{model_name}"),
        # (0.90, f"ninety_{model_name}"),
    ]

    # Read HyperParams
    d = list_config.get(model_name)

    for run in runs:

        degradation_percentage = run[0]
        m_name = run[1]

        # Datasets
        if degradation_percentage is None:
            # Validate on normal validation set

            train_transforms, val_transforms = get_transforms(d.get("data_aug"))

            train_ds = ImageFolder(TRAIN_DATA_PATH, transform=train_transforms)
            val_ds = ImageFolder(VAL_DATA_PATH, transform=val_transforms)

            # Loaders
            train_loader = DataLoader(train_ds, batch_size=d.get("batch_size"), shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=d.get("batch_size"), shuffle=True)

        else:
            # Validate on degraded validation set

            _, val_transforms = get_transforms(d.get("data_aug"))

            train_ds = get_degraded_data(TRAIN_DATA_PATH, degradation_percentage)
            val_degraded_ds = ImageFolder(VAL_DEG_DATA_PATH, transform=val_transforms)

            # Loaders
            train_loader = DataLoader(train_ds, batch_size=d.get("batch_size"), shuffle=True)
            val_loader = DataLoader(val_degraded_ds, batch_size=d.get("batch_size"), shuffle=True)

        # Model instance
        m = get_model(d.get("model")).to(DEVICE)

        # Num Epochs, Loss, Optimizer, Scheduler, Early Stopping
        epochs = d.get("epochs")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(m.parameters(), lr=d.get("start_lr"), weight_decay=d.get("weight_decay"))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=d.get("end_lr"), T_max=d.get("epochs"))
        early_stopping = EarlyStopping(patience=5, mode='min')

        # Define paths for saving/loading model and history
        model_path = os.path.join(*[DATA_PATH, "saved_models", f"{m_name}.pt"])
        history_path = os.path.join(*[DATA_PATH, "saved_models", f"{m_name}_history.pkl"])

        # Train and save
        if TRAIN:
            history = train(m, train_loader, val_loader, epochs, loss_fn, optimizer, scheduler, early_stopping, DEVICE)
            torch.save(m.state_dict(), model_path)
            pickle.dump(history, open(history_path, "wb"))
        else:
            m.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
            history = pickle.load(open(history_path, "rb"))

        # Print history plots
        show_history(history)
