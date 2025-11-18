import os 
import random 
import shutil

datasetA = ""
datasetB = ""

output_root  = ""
os.makedirs(output, exist_ok=True)
N = 10000

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

images_root = os.path.join(output_root, "images")
labels_root = os.path.join(output_root, "labels")

os.makedir(images_root, exist_ok=True)
os.makedir(labels_root, exist_ok=True)

# Collect from train, val, test in each datasets
def collect_all_image(dataset_root):
    splits = ["train", "val", "test"]
    pairs = []
    for split in splits:
        img_dir = os.path.join(dataset_root, "images", split)
        lbl_dir = os.path.join(dataset_root, "labels", split)

        if not os.path.isdir(img_dir):
            continue

        for f in os.listdir(img_dir):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(img_dir, f)
            base = os.path.splitext(f)[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")

            if not os.path.exists(lbl_path):
                print(f"No lable for image {img_path}, next pls.")
                continue

            pairs.append((img_path, lbl_path, f))

    return pairs # (image path, label path, image name)

def sample_and_copy(pairs, n_samples, prefix, images_out, labels_out):
    """
    Randomly sample n_samples from pairs and copy them to output
    """
    if n_samples > len(pairs):
        raise ValueError(
            f"Requested {n_samples} sample with prefix {prefix}, "
            f"but only {len(pairs) available.}"
        )

    selected = random.sample(pairs, n_samples)

    for img_path, lbl_path, orig_name in selected:
        base, ext = os.path.splitext(orig_name)
        new_name = f"{prefix}_{base}{ext}" # prefix + image name + .jpg
        new_lbl_name = f"{prefix}_{base}.txt"

        dst_img = os.path.join(images_out, new_name)
        dst_lbl = os.path.join(labels_out, new_lbl_name)

        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)
