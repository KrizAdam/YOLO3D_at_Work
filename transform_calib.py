import os
import yaml


info_dir = "./YOLO3D/eval/calib"  # Folder where your *_info.yml files are
output_dir = "./YOLO3D/eval/calib"

os.makedirs(output_dir, exist_ok=True)

for file in sorted(os.listdir(info_dir)):
    if not file.endswith("_info.yml"):
        continue

    file_id = file.split("_")[0]
    with open(os.path.join(info_dir, file), "r") as f:
        info = yaml.safe_load(f)

    K = info["cam_K"]
    fx, _, cx, _, fy, cy, _, _, _ = K

    P2 = [fx, 0, cx, 0,
          0, fy, cy, 0,
          0,  0,  1, 0]

    calib_file_path = os.path.join(output_dir, f"{file_id}.txt")
    with open(calib_file_path, "w") as f:
        f.write("P2: " + " ".join(f"{v:.6e}" for v in P2) + "\n")

print(f"[✓] Generated {len(os.listdir(output_dir))} calibration files in {output_dir}")
