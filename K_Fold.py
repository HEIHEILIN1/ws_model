from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
import random
import datetime
from sklearn.model_selection import KFold
import shutil
from tqdm import tqdm

def K_fold(input_dir,k_fold_result_dir):

    dataset_path = Path(input_dir+"data/datasets/waste_classification")  # replace with 'path/to/dataset' for your custom data
    labels = sorted(dataset_path.rglob("*labels/*.txt"))  # all data in 'labels'

    yaml_file = input_dir+"data/datasets/waste_classification/waste_classification.yaml"  # your data YAML with data directories and names dictionary
    with open(yaml_file, encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())

    index = [label.stem for label in labels]  # uses base filename as ID (no extension)，label.stem：路径的文件名部分
    labels_df = pd.DataFrame([], columns=cls_idx, index=index)

    for label in labels:
        lbl_counter = Counter()

        with open(label) as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ")[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    # '0000a16e4b057580_jpg.rf.00ab48988370f64f5ca8ea4...'  0.0  0.0  0.0  0.0  0.0  7.0
    pd.set_option('future.no_silent_downcasting', True)
    labels_df = labels_df.fillna(0.0).infer_objects(copy=False)
    print(labels_df.iloc[0])

    random.seed(0)  # for reproducibility，从起始值0开始生成随机数，起始值0就是种子
    ksplit = 5
    kf = KFold(n_splits=ksplit, shuffle=True,
               random_state=20)  # setting random_state for repeatable results。如果两次划分的seed以及random_state都一样，那么他的划分结果会一样

    kfolds = list(kf.split(labels_df))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=index, columns=folds)

    for i, (train, val) in enumerate(kfolds, start=1):
        # print(train)#[    0     1     2 ... 14960 14962 14963]
        # print(val)#[   13    25    29 ... 14956 14958 14961]
        # print(labels_df.iloc[train].index)#Index(['20210000001', '20210000002', '20210000003', '20210000004',...,'20210014963', '20210014964'],dtype='object', length=11971)
        folds_df.loc[labels_df.iloc[train].index, f"split_{i}"] = "train"
        folds_df.loc[labels_df.iloc[val].index, f"split_{i}"] = "val"

    print(folds_df.iloc[0])
    # split_1    train
    # split_2    train
    # split_3      val
    # split_4    train
    # split_5    train
    # Name: 20210000001, dtype: object

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    # 我们将计算每个fold的类别标签分布，并将其作为fold中出现的类别的比率
    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    print(fold_lbl_distrb.iloc[0])
    # 0     0.24717
    # 1    0.218365
    # 2    0.254487
    # 3    0.247942
    # Name: split_1, dtype: object

    supported_extensions = [".jpg", ".jpeg", ".png"]

    # Initialize an empty list to store image file paths
    images = []

    # Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

    # Create the necessary directories and dataset YAML files
    save_path = Path(k_fold_result_dir + f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )

    for image, label in tqdm(zip(images, labels), total=len(images), desc="Copying files"):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)

    return ds_yamls

if __name__=='__main__':
    pass
