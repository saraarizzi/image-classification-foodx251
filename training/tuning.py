import json
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_data, MobileNetV3Small, EarlyStopping, show_history, train

DATA_PATH = os.path.join("..", "data")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
VAL_DATA_PATH = os.path.join(DATA_PATH, "val")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = True


def get_model(name):

    if name == "mobilenet":
        return MobileNetV3Small(251)
    elif name == "mobilenet_freeze":
        return MobileNetV3Small(251, freeze=True)


if __name__ == "__main__":

    # Get all configurations
    f = open("config.json")
    list_config = json.load(f)
    model_names = list_config.get("model_names")

    for model_name in model_names:

        # Read HyperParams
        d = list_config.get(model_name)

        # Datasets
        train_ds, val_ds = get_data(TRAIN_DATA_PATH, VAL_DATA_PATH, aug_level=d.get("data_aug"), keep=0.4)

        # Loaders
        train_loader = DataLoader(train_ds, batch_size=d.get("batch_size"), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=d.get("batch_size"), shuffle=True)

        # Model instance
        m = get_model(d.get("model")).to(DEVICE)

        # Num Epochs, Loss, Optimizer, Scheduler, Early Stopping
        epochs = d.get("epochs")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(m.parameters(), lr=d.get("lr"), weight_decay=d.get("weight_decay"))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=1e-1, patience=3)
        early_stopping = EarlyStopping(patience=10, mode='min')

        # Define paths for saving model and history
        model_path = os.path.join(*[DATA_PATH, "saved_models", f"{model_name}.pt"])
        history_path = os.path.join(*[DATA_PATH, "saved_models", f"{model_name}_history.pkl"])

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
