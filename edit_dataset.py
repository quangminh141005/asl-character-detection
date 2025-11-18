import os 
import random 
import shutil

datasetA = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/Dataset/hand-dataset"
datasetB = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/Dataset/FreiHAND-yolo"

output_root  = "/home/qminh/Documents/qm/USTH/COURSES/B3/Project/Dataset/Hand-dataset-v2"
N = 10000

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

images_root = os.path.join(output_root, "images")
labels_root = os.path.join(output_root, "labels")

os.makedirs(images_root, exist_ok=True)
os.makedirs(labels_root, exist_ok=True)

# Collect from train, val, test in each datasets
def collect_all_images(dataset_root):
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
            f"but only {len(pairs)} available."
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

# Split dataset to train, test, val
def move_split(img_list, split_name, images_root, labels_root):
    split_img_dir = os.path.join(images_root, split_name)
    split_lbl_dir = os.path.join(labels_root, split_name)
    os.makedirs(split_img_dir, exist_ok=True)
    os.makedirs(split_lbl_dir, exist_ok=True)

    for img_name in img_list:
        # move image
        src_img = os.path.join(images_root, img_name)
        dst_img = os.path.join(split_img_dir, img_name)
        shutil.move(src_img, dst_img)

        # move label
        base = os.path.splitext(img_name)[0]
        src_lbl = os.path.join(labels_root, base + ".txt")
        dst_lbl = os.path.join(split_lbl_dir, base + ".txt")

        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            print(f"Missing label when moving {img_name}: {src_lbl}")


# Collect and sample from each dataset
print("Collect image from hand_detection...")
pairs_A = collect_all_images(datasetA)
print(f"Dataset hand_detection: found {len(pairs_A)}")

print("Collecting images from dataset FreiHAND")
pairs_B = collect_all_images(datasetB)
print(f"Dataset FreiHAND: found {len(pairs_B)}")

print("Sampling from hand_detection dataset ...")
sample_and_copy(pairs_A, N, prefix="SL", images_out=images_root, labels_out=labels_root)

print("Sampling from FreiHAND dataset ...")
sample_and_copy(pairs_B, N, prefix="FreiHAND", images_out=images_root, labels_out=labels_root)


print("\nFinished combining, now i will create and split hehehe")

# Create new train, val, test split
all_images = [
    f for f in os.listdir(images_root) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(all_images)
total = len(all_images)

n_train = int(total * TRAIN_RATIO)
n_val  = int(total * VAL_RATIO)
n_test = total - n_train - n_val

train_imgs = all_images[:n_train]
val_imgs = all_images[n_train:n_train + n_val]
test_imgs = all_images[n_train + n_val:]

print(f"Total images combined: {total}")
print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

print("Moving train images ...")
move_split(train_imgs, "train", images_root, labels_root)

print("Moving test images ...")
move_split(test_imgs, "test", images_root, labels_root)

print("Moving val images ...")
move_split(val_imgs, "val", images_root, labels_root)

print("DONE!!! dsakjhfkajwhefkjhawekjfhakwjehfauiwtyefyai uweyf nghỉ thôi")
