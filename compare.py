import os

# Define the folder paths (change these to your actual folder paths)
folder_png = './YOLO3D/eval/image_2'  # Folder with PNG files
folder_txt1 = './YOLO3D/eval/calib'  # Folder with first TXT file type
folder_txt2 = './YOLO3D/eval/label_2'  # Folder with second TXT file type
folder_val = "./YOLO3D/data/daten/image_2"
# Define the range of numbers to check
file_range = range(2, 7252)  # From 000002 to 003050 (inclusive)

# Function to generate filenames
def generate_filename(num, extension):
    return f"{num:06d}.{extension}"

# Function to get the filenames in a folder
def get_files_in_folder(folder, extension):
    return {f for f in os.listdir(folder) if f.endswith(f".{extension}")}

# Collect the file names in each folder
files_png = get_files_in_folder(folder_png, 'png')
files_txt1 = get_files_in_folder(folder_txt1, 'txt')
files_txt2 = get_files_in_folder(folder_txt2, 'txt')
files_val = get_files_in_folder(folder_val, 'png')

# Prepare to store results
all_files = {generate_filename(num, 'png') for num in file_range}

# Compare files for each folder
comparison = {
    "in_both_txt": [],
    "only_in_png": [],
    "only_in_txt1": [],
    "only_in_txt2": [],
    "missing_in_png": [],
    "missing_in_txt1": [],
    "missing_in_txt2": [],
    "in_val":[],
    "in_val_a_train":[],
    "in_val_a_info":[],
    "in_val_a_label":[]
}

for num in file_range:
    filename_png = generate_filename(num, 'png')
    filename_txt1 = generate_filename(num, 'txt')
    filename_txt2 = generate_filename(num, 'txt')
    filename_val = generate_filename(num, 'png')
    

    # Check if file exists in the respective folders
    png_in_folder = filename_png in files_png
    txt1_in_folder = filename_txt1 in files_txt1
    txt2_in_folder = filename_txt2 in files_txt2
    val_png = filename_val in files_val

    # Categorize the files based on their presence
    if png_in_folder and txt1_in_folder and txt2_in_folder and not val_png:
        comparison["in_both_txt"].append(filename_png)
    if not png_in_folder and not val_png:
        comparison["missing_in_png"].append(filename_png)
    if not txt1_in_folder and not val_png:
        comparison["missing_in_txt1"].append(filename_txt1)
    if not txt2_in_folder and not val_png:
        comparison["missing_in_txt2"].append(filename_txt2)
    if png_in_folder and not txt1_in_folder and not txt2_in_folder :
        comparison["only_in_png"].append(filename_png)
    if txt1_in_folder and not png_in_folder and not txt2_in_folder:
        comparison["only_in_txt1"].append(filename_txt1)
    if txt2_in_folder and not png_in_folder and not txt1_in_folder:
        comparison["only_in_txt2"].append(filename_txt2)
    '''if val_png and not ( png_in_folder or txt2_in_folder or txt1_in_folder):
        comparison["in_val"].append(filename_val)
    if val_png and png_in_folder:
        comparison["in_val_a_train"].append(filename_val)
    if val_png and txt1_in_folder:
        comparison["in_val_a_info"].append(filename_val)
    if val_png and txt2_in_folder:
        comparison["in_val_a_label"].append(filename_val)'''
    

# Print out the results
def print_comparison(result, description, ending):
    print(f"{description}:")
    if result:
        liste = []
        cnt = 0
        for item in result:
            cnt += 1
            new_item = "\""+item[:6]+ending+"\""
            liste.append(new_item)
        liste = " ".join(liste)
        print(f"{cnt}::\n {liste}")
    else:
        print(" No files found")
    print("\n")

print_comparison(comparison["in_both_txt"], "Files present in both TXT folders","")
print_comparison(comparison["only_in_png"], "Files only in PNG folder",".png")
print_comparison(comparison["only_in_txt1"], "Files only in first TXT folder","_info.yml")
print_comparison(comparison["only_in_txt2"], "Files only in second TXT folder","_gt.yml")
print_comparison(comparison["missing_in_png"], "Files missing from PNG folder",".png")
print_comparison(comparison["missing_in_txt1"], "Files missing from first TXT folder","_info.yml")
print_comparison(comparison["missing_in_txt2"], "Files missing from second TXT folder","_gt.yml")
print_comparison(comparison["in_val"], "Files only in val","")
print_comparison(comparison["in_val_a_train"], "files in val and train","png")
print_comparison(comparison["in_val_a_label"], "files in val and label","_info.yml")
print_comparison(comparison["in_val_a_info"], "files in val and info","_gt.yml")


