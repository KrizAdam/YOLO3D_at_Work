import os
import yaml

input_dir = "YOLO3D/eval/label_2/"
output_dir = "YOLO3D/eval/label_2/"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith("_gt.yml"):
        base = filename.replace("_gt.yml", "")
        gt_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{base}.txt")

        with open(gt_path, "r") as f:
            gt_data = yaml.safe_load(f)

        kitti_lines = []

        for obj in gt_data:
            class_name = obj["class"]
            x1, y1, x2, y2 = obj["bbox"]
            h, w, l = obj["size"]
            x, y, z = obj["location_cam"]
            ry = obj["theta"]           # rotation_y in radians
            alpha = obj.get("alpha", -10.0)  # use alpha if available

            line = f"{class_name} 0 0 {alpha} {x1} {y1} {x2} {y2} {h} {w} {l} {x} {y} {z} {ry}"

            kitti_lines.append(line)

        with open(output_path, "w") as f:
            f.write("\n".join(kitti_lines))

        print(f"✔ Converted: {filename} → {base}.txt")
