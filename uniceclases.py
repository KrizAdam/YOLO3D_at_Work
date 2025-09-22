import os
import json

label_dir = "./YOLO3D/data/daten/label_2"
output_file = "./class_averages.txt"

class_map = {}

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue
    with open(os.path.join(label_dir, file)) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue  # malformed line
            cls = parts[0]
            h, w, l = map(float, parts[8:11])
            if cls not in class_map:
                class_map[cls] = {"count": 0, "total": [0.0, 0.0, 0.0]}
            class_map[cls]["count"] += 1
            class_map[cls]["total"][0] += h
            class_map[cls]["total"][1] += w
            class_map[cls]["total"][2] += l

# Save to JSON
with open(output_file, "w") as f:
    json.dump(class_map, f, indent=4)

print(f"Saved class averages to {output_file}")
