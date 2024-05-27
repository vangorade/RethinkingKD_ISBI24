import os
import shutil

# Path to the "trainn" folder
trainn_folder = "./lits/data/train/"

# Path to the "data" folder where you want to organize the data
data_folder = "./lits19/data/"


# Create the "data" folder if it doesn't exist
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Iterate through the files in the "trainn" folder
for filename in os.listdir(trainn_folder):
    if filename.startswith("volume-") and filename.endswith(".nii"):
        # Extract the case_id from the filename
        case_id = filename.split("-")[1].split(".")[0]

        # Create the case folder inside the "data" folder
        case_folder = os.path.join(data_folder, f"case_{case_id.zfill(5)}")
        if not os.path.exists(case_folder):
            os.makedirs(case_folder)

        # Copy the volume.nii file to the case folder
        volume_source_path = os.path.join(trainn_folder, filename)
        volume_dest_path = os.path.join(case_folder, "volume.nii")
        shutil.copy(volume_source_path, volume_dest_path)

        # Find the corresponding segmentation file
        segmentation_filename = f"segmentation-{case_id}.nii"
        segmentation_source_path = os.path.join(trainn_folder, segmentation_filename)

        # Copy the segmentation.nii file to the case folder
        segmentation_dest_path = os.path.join(case_folder, "segmentation.nii")
        shutil.copy(segmentation_source_path, segmentation_dest_path)

print("Data organization completed.")
