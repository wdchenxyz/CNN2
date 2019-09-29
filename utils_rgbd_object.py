import os 
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import cv2

from itertools import groupby 
from collections import defaultdict

def get_data_from_file(target_list, data_dir='dataset/rgbd_object/'):
    # data_dir = "cropped_data/{}/*"
    data_dir = os.path.join(data_dir, '{}', '*')
    target_dataset = defaultdict(list)

    for label, target in enumerate(target_list):
        target_videos = glob.glob(data_dir.format(target))
        target_videos = sorted(target_videos, key=lambda x: int(os.path.basename(x).split("_")[-1]))
#         print(target_videos)
        for target_video in target_videos:
            dataset = glob.glob(target_video + "/*_crop.png")
            dataset = sorted(dataset, key=lambda x: (int(os.path.basename(x).split("_")[2]),
                                                     int(os.path.basename(x).split("_")[3])))

            prefix = os.path.basename(target_video) + "_"
            for key, value in groupby(dataset, lambda x: os.path.basename(x).split(prefix)[1].split("_")[0]):
#                 print("{} - {}".format(target_video, key))
                value = list(value)
                target_dataset[target].append(list(zip(value, np.roll(value, -1), [label]*len(value))))
#             print("---")
    return target_dataset

def split_data_label(dataset):
    left, right, label = map(list, zip(*dataset))
    return list(zip(left, right)), label

def train_test_split(dataset, split_ratio=3, limit=1100):
    train_dataset = []
    test_dataset = []
    valid_dataset = []
    for category, data in dataset.items():
        cat_data = []
        for instance_or_video in data:
            train_dataset.extend(instance_or_video[:(len(instance_or_video) // 9) * 2])
            valid_dataset.extend(instance_or_video[(len(instance_or_video) // 9) * 2 : (len(instance_or_video) // 9) * 3])
            cat_data.extend(instance_or_video[(len(instance_or_video) // 3):])
            # test_dataset.extend(instance_or_video[(len(instance_or_video) // 3):])
        if limit:
            test_dataset.extend(cat_data[:limit])
        else:
            test_dataset.extend(cat_data)
    return train_dataset, test_dataset, valid_dataset

def revolving_dataset(dataset, split_part, idx):
    split_data = []
    for category, data in dataset.items():
        for instance_or_video in data:
            split_data.extend(instance_or_video[idx * (len(instance_or_video) // split_part) : (idx+1) * (len(instance_or_video) // split_part)])
    return split_data

def plot_data_in_row(data, cat, num_img, img_in_row=5):
    num_rows = num_img // img_in_row
    fig, axes = plt.subplots(num_rows, img_in_row, figsize=(18, 3.6 * num_rows))
    axes = axes.flatten()
    rdn_view = np.random.randint(0, len(data)-5)
    for i in range(num_img):
        idx = rdn_view + i
        img = mpimg.imread(data[idx][0])
        axes[i].imshow(img)
        axes[i].set_title("{} ({})".format(cat, data[idx][2]))
        axes[i].set_xlabel("{}".format(img.shape))
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.show()
    
def get_dataset_info(dataset, num_img):
    for idx, (cat, data_list) in enumerate(dataset.items()):
        rdn_instance = np.random.randint(0, len(data_list))
        example_per_category = sum([len(inst) for inst in data_list])
        print("{} : {} has {} examples".format(idx + 1, cat, example_per_category))
        plot_data_in_row(data_list[rdn_instance], cat, num_img)


def plot_dataset(dataset, display_object=["pitcher", "flashlight"]):
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    iter_rdn_instance = [np.random.randint(0, len(dataset[obj])) for obj in display_object]
    iter_obj = [dataset[obj][ins] for obj, ins in zip(display_object, iter_rdn_instance) for _ in (0, 1)]
    
    for i, obj in zip(range(0, 8, 2), iter_obj):
        rdn_view = np.random.randint(0, len(obj))
        for k in (0, 1):
            img = np.array(imageio.imread(obj[rdn_view][k]))
            img = cv2.resize(img, (144, 144))
            axes[i+k].imshow(img)
            axes[i+k].set_title("{} ({})".format(display_object[i//4], obj[rdn_view][2]))
            axes[i+k].set_xticks([])
            axes[i+k].set_yticks([])
    plt.tight_layout()
    plt.show()