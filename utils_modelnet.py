import os
import numpy as np
import glob
from collections import defaultdict

def get_data_from_file(class_set, dataset_path):
    data_dir = os.path.join(dataset_path, "{}", "*")
    target_dataset = defaultdict(list)
    elevation = 9
    azimuth = 72
    
    for label, target in enumerate(class_set):
        target_instacne_list = glob.glob(data_dir.format(target))
        for target_instance in target_instacne_list:
            instance_imgs = glob.glob(os.path.join(target_instance, "*.png"))
            view_in_elevations = [instance_imgs[i * azimuth : (i+1) * azimuth] for i in range(elevation)]
            binocular_data = [list(zip(view_in_elevation, np.roll(view_in_elevation, -1), [label] * len(view_in_elevation))) 
                              for view_in_elevation in view_in_elevations]
            
            target_dataset[target].append(binocular_data)
    return target_dataset


def train_test_split(dataset, split_ratio=3):
    train_dataset = []
    valid_dataset = []
    test_dataset = []
    for cat, data in dataset.items(): # 5
        for instance in data: # 15
            for elevation in instance: # 9
                train_dataset.extend(elevation[:(len(elevation) // 9) * 2])
                valid_dataset.extend(elevation[(len(elevation) // 9) * 2 : (len(elevation) // 9) * 3])
                test_dataset.extend(elevation[(len(elevation) // 9) * 3:])
    return train_dataset, valid_dataset, test_dataset

def split_data_label(dataset):
    left, right, label = map(list, zip(*dataset))
    return list(zip(left, right)), label