import os

# Set the path to the folder containing the images
folder_path = r'C:\Users\jacka\Documents\kAI-havertz\cats'

# List all the files in the folder
files = os.listdir(folder_path)

# Initialize the counter
counter = 1

# Loop through the files
for file in files:
    # Check if the file is an image (assuming only .jpg files)
    if file.lower().endswith('.jpg'):
        # Construct the new file name
        new_file_name = f'cat_{counter}.jpg'

        # Rename the file
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))

        # Increment the counter
        counter += 1
