import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small, vit_b_16
from torchvision.transforms import v2


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes, freeze=False):
        super(MobileNetV3Small, self).__init__()
        self.model = mobilenet_v3_small(weights="DEFAULT")
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier[3].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


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


def get_transforms(level=None):

    train_list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((224, 224)),
                  v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    val_list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    if level == "light":
        train_list += [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomErasing(p=0.5),
            v2.RandomRotation(degrees=45),
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ]
    elif level == "heavy":
        train_list += [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(degrees=45),
            # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            # v2.GaussianBlur(kernel_size=3, sigma=1.5),
            v2.RandomInvert(p=0.5),
            # v2.RandomGrayscale(p=0.5),
            v2.RandomSolarize(p=0.5, threshold=0.2),
            v2.RandomPosterize(p=0.5, bits=4)
        ]

    train_transforms, val_transforms = v2.Compose(train_list), v2.Compose(val_list)

    return train_transforms, val_transforms


def get_data(train_path, val_path, aug_level, keep=1):

    train_transforms, val_transforms = get_transforms(aug_level)

    train_dataset = ImageFolder(train_path, transform=train_transforms)
    val_dataset = ImageFolder(val_path, transform=val_transforms)

    train_indices, _ = train_test_split(
        np.arange(len(train_dataset)), train_size=keep, stratify=train_dataset.targets, random_state=0
    )

    val_indices, _ = train_test_split(
        np.arange(len(val_dataset)), train_size=keep, stratify=val_dataset.targets, random_state=0
    )

    train_sample = Subset(train_dataset, train_indices)
    val_sample = Subset(val_dataset, val_indices)

    return train_sample, val_sample


def show_history(hist):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(hist.get("train_loss"), label='Train')
    plt.plot(hist.get("val_loss"), label='Val')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(hist.get("train_acc1"), label='Train Acc@1', ls="-", color="gray")
    plt.plot(hist.get("train_acc3"), label='Train Acc@3', ls="-", color="gray")
    plt.plot(hist.get("train_acc5"), label='Train Acc@5', ls="-", color="gray")
    plt.plot(hist.get("val_acc1"), label='Val Acc@1', color="blue")
    plt.plot(hist.get("val_acc3"), label='Val Acc@3', color="blue")
    plt.plot(hist.get("val_acc5"), label='Val Acc@5', color="blue")
    plt.legend()
    plt.title("Accuracy")

    plt.show()


def accuracy_at_k(predictions, y_one_hot, k=1):
    y_true = torch.argmax(y_one_hot, dim=1)

    _, top_k_indices = torch.topk(predictions, k, dim=1)

    correct = top_k_indices.eq(y_true.view(-1, 1).expand_as(top_k_indices))

    accuracy = correct.float().sum() / predictions.size(0)

    return accuracy.item()


def train_step(model, data_loader, loss_fn, optimizer, device):
    train_loss, train_acc1, train_acc3, train_acc5 = 0, 0, 0, 0

    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.type(torch.LongTensor).to(device)
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

    # Print loss and accuracy
    train_loss /= len(data_loader)
    train_acc1 /= len(data_loader)
    train_acc3 /= len(data_loader)
    train_acc5 /= len(data_loader)

    return train_loss, train_acc1, train_acc3, train_acc5


def val_step(model, data_loader, loss_fn, scheduler, device):
    val_loss, val_acc1, val_acc3, val_acc5 = 0, 0, 0, 0
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.type(torch.LongTensor).to(device)
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

        # Scheduler step
        lr = None
        if scheduler is not None:
            scheduler.step(val_loss / len(data_loader))
            lr = scheduler.get_last_lr()

    val_loss /= len(data_loader)
    val_acc1 /= len(data_loader)
    val_acc3 /= len(data_loader)
    val_acc5 /= len(data_loader)

    return val_loss, val_acc1, val_acc3, val_acc5, lr


def train(model, train_loader, val_loader, epochs, loss_fn, optimizer, scheduler, early_stopping, device):
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
        train_loss, ta1, ta3, ta5 = train_step(model=model,
                                               data_loader=train_loader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)
        val_loss, va1, va3, va5, lr = val_step(model=model,
                                               data_loader=val_loader,
                                               loss_fn=loss_fn,
                                               scheduler=scheduler,
                                               device=device)
        # Print epoch results
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

        # Early Stopping on Validation Loss
        if early_stopping(val_loss):
            print(f"Stopped at epoch {epoch + 1} because of early stopping")
            break

    return results
