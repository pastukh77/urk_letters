import os
import shutil

folder = "letters"
folder_clean = "letters_cleaned"
for f in os.listdir(folder):
    if f.endswith("_1"):
        shutil.move(os.path.join(folder, f), os.path.join(folder_clean, f.split("_")[0]))