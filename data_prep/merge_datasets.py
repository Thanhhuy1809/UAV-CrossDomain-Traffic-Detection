import os
import shutil

src_train_img = r"E:\Downloads\Train\Train\images"
src_train_lbl = r"E:\Downloads\Train\Train\labels"
src_test_img = r"E:\Downloads\Test\Test\images"
src_test_lbl = r"E:\Downloads\Test\Test\labels"

dst_train_img = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET-train\images"
dst_train_lbl = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET-train\labels"
dst_val_img = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET-val\images"
dst_val_lbl = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET-val\labels"

def move_files(src_dir, dst_dir, prefix):
    if not os.path.exists(src_dir):
        print(f"Skipping {src_dir}, not found.")
        return
    count = 0
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        if os.path.isfile(src_path):
            dst_path = os.path.join(dst_dir, f"{prefix}{filename}")
            # Use shutil.move to cut (move) the file
            shutil.move(src_path, dst_path)
            count += 1
    print(f"Moved {count} files from {src_dir} to {dst_dir} with prefix '{prefix}'")

# Move train data
move_files(src_train_img, dst_train_img, "songdo_")
move_files(src_train_lbl, dst_train_lbl, "songdo_")

# Move test data into val
move_files(src_test_img, dst_val_img, "songdo_")
move_files(src_test_lbl, dst_val_lbl, "songdo_")

# Delete old cache file
cache_file = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET-train\labels.cache"
if os.path.exists(cache_file):
    os.remove(cache_file)
    print("Deleted old labels.cache")

val_cache = r"e:\UAV-Traffic-Detection\AI-UAV-DL\AI\dataset\VisDrone2019-DET-val\labels.cache"
if os.path.exists(val_cache):
    os.remove(val_cache)
    print("Deleted old val labels.cache")

print("Done moving dataset!")
