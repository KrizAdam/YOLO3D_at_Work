import os
import yaml
from collections import Counter

def count_objects_in_yml(folder_path):
    class_counter = Counter()
    total_objects = 0

    # loop through all .yml files
    for filename in os.listdir(folder_path):
        if filename.endswith(".yml") or filename.endswith(".yaml"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                try:
                    data = yaml.safe_load(file)
                    if isinstance(data, list):  # each file contains a list of objects
                        for obj in data:
                            obj_class = obj.get("class")
                            if obj_class:
                                class_counter[obj_class] += 1
                                total_objects += 1
                except yaml.YAMLError as e:
                    print(f"⚠️ Error parsing {filename}: {e}")

    return total_objects, class_counter


if __name__ == "__main__":
    folder = "YOLO3D/eval/label_2/"  # <-- change this
    total, counts = count_objects_in_yml(folder)

    print(f"Total objects across all files: {total}\n")
    print("Counts per class:")
    for cls, cnt in counts.most_common():
        print(f"  {cls}: {cnt}")
