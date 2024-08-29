import os
import shutil
from zipfile import ZipFile
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

# pytorch
import torch
from torch import nn
# from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torch.utils.data import DataLoader

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN = True
GDRIVE = False

# seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def accuracy_at_k(predictions, y_one_hot, k=1):
    """
    Calculate the top-k accuracy.

    Args:
    - predictions (torch.Tensor): Tensor of shape (batch_size, num_classes) containing the model's predicted scores.
    - y_one_hot (torch.Tensor): One-hot encoded ground truth tensor of shape (batch_size, num_classes).
    - k (int): Number of top predictions to consider for accuracy.

    Returns:
    - accuracy (float): The top-k accuracy score.
    """
    # Convert one-hot encoded ground truth to label indices
    y_true = torch.argmax(y_one_hot, dim=1)

    # Get the top-k predicted indices along the last dimension (num_classes)
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    # Check if the true labels are in the top-k indices
    correct = top_k_indices.eq(y_true.view(-1, 1).expand_as(top_k_indices))

    # Calculate the accuracy
    accuracy = correct.float().sum() / predictions.size(0)

    return accuracy.item()


class EarlyStopping:

    def __init__(self, patience=5, mode='min'):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_val = None

    def __call__(self, val):
        val = float(val)

        if self.best_val is None:
            self.best_val = val
        elif self.mode == 'min' and val < self.best_val:
            self.best_val = val
            self.counter = 0
        elif self.mode == 'max' and val > self.best_val:
            self.best_val = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early Stopping!")
                return True

        return False


def train_step(model, data_loader, loss_fn, optimizer, scheduler=None):
    train_loss, train_acc1, train_acc3, train_acc5 = 0, 0, 0, 0

    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(DEVICE), y.type(torch.LongTensor).to(DEVICE)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=251).float()

        # Forward Pass
        y_pred = model(X)
        y_pred = y_pred.squeeze()

        # Calculate Loss
        loss = loss_fn(y_pred, y_one_hot)
        train_loss += loss.item()

        # Optimizer reset step
        optimizer.zero_grad()

        # Loss Backpropagation
        loss.backward(retain_graph=True)

        # Optimizer step
        optimizer.step()

        # Calculate accuracy
        train_acc1 += accuracy_at_k(y_pred, y_one_hot, 1)
        train_acc3 += accuracy_at_k(y_pred, y_one_hot, 3)
        train_acc5 += accuracy_at_k(y_pred, y_one_hot, 5)

        # Clean Cache
        torch.cuda.empty_cache()

    # Scheduler step
    lr = None
    if scheduler is not None:
        scheduler.step()
        lr = scheduler.get_last_lr()

    # Print loss and accuracy
    train_loss /= len(data_loader)
    train_acc1 /= len(data_loader)
    train_acc3 /= len(data_loader)
    train_acc5 /= len(data_loader)

    return train_loss, train_acc1, train_acc3, train_acc5, lr


def val_step(model, data_loader, loss_fn):
    val_loss, val_acc1, val_acc3, val_acc5 = 0, 0, 0, 0
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(DEVICE), y.type(torch.LongTensor).to(DEVICE)
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=251).float()

            # Forward pass
            val_pred = model(X)
            val_pred = val_pred.squeeze()

            # Calculate loss
            loss = loss_fn(val_pred, y_one_hot)
            val_loss += loss.item()

            # Calculate accuracy
            val_acc1 += accuracy_at_k(val_pred, y_one_hot, 1)
            val_acc3 += accuracy_at_k(val_pred, y_one_hot, 3)
            val_acc5 += accuracy_at_k(val_pred, y_one_hot, 5)

            # Clean Cache
            torch.cuda.empty_cache()

    val_loss /= len(data_loader)
    val_acc1 /= len(data_loader)
    val_acc3 /= len(data_loader)
    val_acc5 /= len(data_loader)

    return val_loss, val_acc1, val_acc3, val_acc5


def train(model, train_loader, test_loader, optimizer, loss_fn, epochs, early_stop, scheduler=None):
    results = {
        "train_loss": [],
        "train_acc1": [],
        "train_acc3": [],
        "train_acc5": [],
        "val_loss": [],
        "val_acc1": [],
        "val_acc3": [],
        "val_acc5": []
    }

    for epoch in range(epochs):
        train_loss, ta1, ta3, ta5, lr = train_step(model=model,
                                                   data_loader=train_loader,
                                                   loss_fn=loss_fn,
                                                   optimizer=optimizer,
                                                   scheduler=scheduler)
        val_loss, va1, va3, va5 = val_step(model=model,
                                           data_loader=test_loader,
                                           loss_fn=loss_fn)
        # Print out what's happening
        print(
            f"Epoch: {epoch} --> \t"
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"ta@1: {ta1:.4f} | "
            f"ta@3: {ta3:.4f} | "
            f"ta@5: {ta5:.4f} | "
            f"va@1: {va1:.4f} | "
            f"va@3: {va3:.4f} | "
            f"va@5: {va5:.4f} | "
            f"LR: {lr} "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc1"].append(ta1)
        results["train_acc3"].append(ta3)
        results["train_acc5"].append(ta5)
        results["val_loss"].append(val_loss)
        results["val_acc1"].append(va1)
        results["val_acc3"].append(va3)
        results["val_acc5"].append(va5)

        # early stopping
        if early_stop(val_loss):
            print(f"Stopped at epoch {epoch + 1} because of early stopping")
            break

    # Return the filled results at the end of the epochs
    return results


def get_mob_net(weights=None):
  mod = mobilenet_v3_small(weights=weights)

  # Change to 251 output class
  mod.classifier[3] = nn.Linear(
      in_features=mod.classifier[3].in_features,
      out_features=251, bias=True)

  return mod



if __name__ == "__main__":

    torch.cuda.empty_cache()

    DATA_PATH = "data"
    TRAIN_PATH = os.path.join(DATA_PATH, "train")
    VAL_PATH = os.path.join(DATA_PATH, "val")

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])
    train_dataset = ImageFolder(TRAIN_PATH, transform=train_transforms)
    val_dataset = ImageFolder(VAL_PATH, transform=val_transforms)

    # Hyper Parameters
    BATCH_SIZE = 128
    EPOCHS = 100
    LOSS_FN = nn.CrossEntropyLoss()
    LEARNING_RATE = 1e-3

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    m = get_mob_net()
    m = m.to(DEVICE)
    optimizer = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
    early_stopping = EarlyStopping(patience=5, mode='min')

    model_path = os.path.join(*[DATA_PATH, "saved_models", "mobilenetv3_from_scratch.pt"])
    history_path = os.path.join(*[DATA_PATH, "saved_models", "history_mobilenetv3_from_scratch.pkl"])
    if True:
        history = train(m, train_loader, val_loader, optimizer,
                        LOSS_FN, EPOCHS, early_stopping, scheduler)
        torch.save(m.state_dict(), model_path)
        pickle.dump(history, open(history_path, "wb"))
    else:
        m.load_state_dict(torch.load(model_path, map_location=torch.device(DEVICE)))
        history = pickle.load(open(history_path, "rb"))
