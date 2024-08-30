import os.path
import time
import zipfile
import shutil
import pandas as pd

RAW_DATA_PATH = os.path.join(*["..", "data", "raw"])
CLEAN_DATA_PATH = os.path.join(*["..", "data", "clean"])

TRAIN = dict(csv_info=os.path.join(RAW_DATA_PATH, "train_info_dirty.csv"),
             start_zip=os.path.join(RAW_DATA_PATH, "train_set.zip"),
             start_folder=os.path.join(RAW_DATA_PATH, "train_set"),
             end_zip=os.path.join(CLEAN_DATA_PATH, "train_dirty.zip"),
             end_folder=os.path.join(CLEAN_DATA_PATH, "train"),
             name="TRAIN")

VAL = dict(csv_info=os.path.join(RAW_DATA_PATH, "val_info.csv"), start_zip=os.path.join(RAW_DATA_PATH, "val_set.zip"),
           start_folder=os.path.join(RAW_DATA_PATH, "val_set"), end_zip=os.path.join(CLEAN_DATA_PATH, "val.zip"),
           end_folder=os.path.join(CLEAN_DATA_PATH, "val"),
           name="VAL")

VAL_DEGRADED = dict(csv_info=os.path.join(RAW_DATA_PATH, "val_info.csv"),
                    start_zip=os.path.join(RAW_DATA_PATH, "val_set_degraded_2023.zip"),
                    start_folder=os.path.join(RAW_DATA_PATH, "val_set_degraded"),
                    end_zip=os.path.join(CLEAN_DATA_PATH, "val_degraded.zip"),
                    end_folder=os.path.join(CLEAN_DATA_PATH, "val_degraded"),
                    name="VAL_DEGRADED")

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def unzip_data(d):
    with zipfile.ZipFile(d.get("start_zip"), "r") as zf:
        zf.extractall(RAW_DATA_PATH)


def format_class_as_folder(d):
    # Load the CSV file
    csv_path = d.get("csv_info")
    data_info = pd.read_csv(csv_path, header=None, names=["file", "class"])

    # Base directory where the files are currently located
    base_dir: str = d.get("start_folder")

    # Directory where you want to create the class folders
    output_dir = d.get("end_folder")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each row in the CSV
    for index, row in data_info.iterrows():
        file_name = row["file"]  # First column (file name)
        file_class = str(row["class"])  # Second column (class)

        # Define the full path for the current file
        file_path = os.path.join(base_dir, file_name)

        # Create the class directory if it doesn't exist
        class_dir = os.path.join(output_dir, file_class)
        os.makedirs(class_dir, exist_ok=True)

        # Move the file to the class directory
        shutil.move(file_path, os.path.join(class_dir, file_name))


def zip_data(d):
    with zipfile.ZipFile(d.get("end_zip"), "w", zipfile.ZIP_DEFLATED) as zf:
        # Walk through the directory structure
        for root, dirs, files in os.walk(d.get("end_folder")):
            for file in files:
                # Create the full filepath by joining the root with the file
                full_path = os.path.join(root, file)
                # Add file to zip, removing the base directory from the arc_name to maintain folder structure
                arc_name = os.path.relpath(full_path, start=CLEAN_DATA_PATH)
                zf.write(full_path, arcname=arc_name)


if __name__ == '__main__':

    for data in [TRAIN, VAL, VAL_DEGRADED]:
        print(f"Formatting {data.get('name')}")

        start_time = time.time()
        print("Started at:", time.strftime(DATE_FORMAT))

        # Read raw/original data
        unzip_data(data)
        print("Unzipped at:", time.strftime(DATE_FORMAT))

        # Format data so that each class is a folder
        format_class_as_folder(data)
        print("Formatted at:", time.strftime(DATE_FORMAT))

        # Zip formatted data
        zip_data(data)
        print("Saved new zip at:", time.strftime(DATE_FORMAT))

        end_time = time.time()
        duration = end_time - start_time
        print(f"Script total duration: {duration:.2f} seconds")

        # Remove temp folders
        os.rmdir(data.get("start_folder"))

    print("Done :)")
