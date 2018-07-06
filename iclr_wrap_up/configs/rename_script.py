import os

CONFIG_DIR = "cohort_4/"

for root, dirs, files in os.walk(CONFIG_DIR):
    for name in files:
        new_name = name[:7] + "4" + name[8:]
        new_path = os.path.join(root, new_name)
        old_path = os.path.join(root, name)
        os.renames(old_path, new_path)