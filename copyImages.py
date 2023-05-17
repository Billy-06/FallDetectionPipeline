import os
import shutil

# Set the source and destination folder paths
source_folder = "my_data/LLVIP/visible/train/"
destination_folder = "my_data/LLVIP/visible/val/"

# Make sure the source folder exists
if not os.path.exists(source_folder):
    print(f"Source folder '{source_folder}' does not exist")
    exit()

# Make sure the destination folder exists, or create it if it doesn't
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Counter for number of files copied
files_copied = 0

# Iterate over all files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is an image file (you can modify the list of extensions to match your file types)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        # Copy the file to the destination folder
        shutil.copy2(os.path.join(source_folder, filename), destination_folder)
        files_copied += 1

        # If we've copied the desired number of files, exit the loop
        if files_copied == 3000:
            break

print(f"Copied {files_copied} files to '{destination_folder}'")

