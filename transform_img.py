import os
from PIL import Image

def strip_exif_from_pngs(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                file_path = os.path.join(dirpath, filename)
                try:
                    img = Image.open(file_path)
                    img.save(file_path, exif=None)
                    print(f"[OK] Cleaned EXIF: {file_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Remove EXIF metadata from PNG files in a directory")
    parser.add_argument("path", type=str, help="Path to the root directory containing PNG files")
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"[ERROR] Path does not exist or is not a directory: {args.path}")
        return

    print(f"[INFO] Cleaning EXIF from PNGs in: {args.path}")
    strip_exif_from_pngs(args.path)
    print("[DONE] All files processed.")

if __name__ == "__main__":
    main()
