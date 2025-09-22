import os

# Change this to the path of your folder
folder_path = "./YOLO3D/eval/label_2"

for filename in os.listdir(folder_path):
    if filename.endswith("_gt.yml"):
        old_path = os.path.join(folder_path, filename)
        new_filename = filename.replace("_gt", "")
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")

print("Done!")
