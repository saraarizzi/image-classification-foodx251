import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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

    # List of all classes
    classes_codes = np.arange(251)

    # Get best model configuration
    f = open("config.json")
    list_config = json.load(f)
    best_model = "04_mnv3_fine_tuned"

    # Degradation on train data percentage
    models_to_evaluate = [
        (f"02_mnv3_fine_tuned", VAL_DATA_PATH),
        (f"03_mnv3_fine_tuned", VAL_DATA_PATH),
        (f"04_mnv3_fine_tuned", VAL_DATA_PATH),
        (f"04_mnv3_fine_tuned", VAL_DEG_DATA_PATH),
        (f"full_{best_model}", VAL_DATA_PATH),
        (f"full_{best_model}", VAL_DEG_DATA_PATH),
        (f"fifteen_{best_model}", VAL_DATA_PATH),
        (f"fifteen_{best_model}", VAL_DEG_DATA_PATH),
        (f"thirty_{best_model}", VAL_DATA_PATH),
        (f"thirty_{best_model}", VAL_DEG_DATA_PATH),
        (f"forty_five_{best_model}", VAL_DATA_PATH),
        (f"forty_five_{best_model}", VAL_DEG_DATA_PATH),
        (f"sixty_{best_model}", VAL_DATA_PATH),
        (f"sixty_{best_model}", VAL_DEG_DATA_PATH),
        (f"ninety_{best_model}", VAL_DATA_PATH),
        (f"ninety_{best_model}", VAL_DEG_DATA_PATH)
    ]

    models_to_evaluate = [
        (f"full_{best_model}", VAL_DATA_PATH)
    ]

    # Read HyperParams
    d = list_config.get(best_model)

    for model_info in models_to_evaluate:

        model_name = model_info[0]
        val_path = model_info[1]

        print(f"{model_name} ------------------------- Evaluated on {val_path}")

        # Dataset
        _, val_transforms = get_transforms(d.get("data_aug"))
        val_ds = ImageFolder(val_path, transform=val_transforms)

        # Loader
        val_loader = DataLoader(val_ds, batch_size=d.get("batch_size"), shuffle=True)

        # Model instance
        m = get_model(d.get("model")).to(DEVICE)

        # Define paths for saving/loading model and history
        model_path = os.path.join(*[DATA_PATH, "saved_models", f"{model_name}.pt"])
        history_path = os.path.join(*[DATA_PATH, "saved_models", f"{model_name}_history.pkl"])

        # Load model
        m.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device(DEVICE)))
        history = pickle.load(open(history_path, "rb"))

        # Evaluate
        val_acc1, val_acc3, val_acc5 = evaluate(m, val_loader, DEVICE)

        print(
            f"Accuracy@1 --> {np.round(val_acc1*100, 2)}\n"
            f"Accuracy@3 --> {np.round(val_acc3*100, 2)}\n"
            f"Accuracy@5 --> {np.round(val_acc5*100, 2)}\n"
        )

        # By single class
        accs_1, accs_3, accs_5 = [], [], []
        for class_code in classes_codes:
            subset_class = Subset(val_ds, indices=[i for i, x in enumerate(val_ds.targets) if x == class_code])
            class_loader = DataLoader(subset_class, batch_size=d.get("batch_size"), shuffle=True)

            # Evaluate
            val_acc1, val_acc3, val_acc5 = evaluate(m, class_loader, DEVICE)
            accs_1.append(np.round(val_acc1,2))
            # accs_3.append(val_acc3)
            # accs_5.append(val_acc5)

        print(accs_1)
