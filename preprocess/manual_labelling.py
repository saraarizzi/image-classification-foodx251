import os

import pandas as pd
import cv2 as cv


if __name__ == "__main__":

    image_info_path = os.path.join(*["..", "data", "raw", "train_info_dirty.csv"])
    image_data = pd.read_csv(image_info_path, header=None, names=["file", "class"])
    os.makedirs(os.path.join(*["..", "data", "manual_labelling"]), exist_ok=True)

    classes_to_label = [
        196, 15,
        0, 233,
        57, 25
    ]

    classes_names = [
        "omelette", "seaweed salad",
        "macaron", "scotch egg",
        "gnocchi", "ramen"
    ]

    for class_id, class_name in zip(classes_to_label, classes_names):

        class_images = image_data[image_data["class"] == class_id]["file"].to_list()

        to_remove = []
        for file_name in class_images:
            im = cv.imread(os.path.join(*["..", "data", "clean", "train", str(class_id), file_name]))
            cv.imshow("temp_image", im)
            cv.waitKey(1)

            if input(f"{class_name}?") == "1":
                to_remove.append(1)
            else:
                to_remove.append(0)

            cv.destroyAllWindows()

        pd.DataFrame({
            "image_name": class_images,
            "to_remove": to_remove
        }).to_csv(os.path.join(*["..", "data", "manual_labelling", f"{class_id}_to_remove.csv"]))

