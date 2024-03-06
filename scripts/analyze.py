"""
Statistically analyze the dataset
"""
import os
import numpy as np
import cv2
import glob
import json


def get_dataset():
    data = None
    for fname in glob.glob('data/labels/train/**/*.json'):
        with open(fname, 'r') as f:
            labels = json.load(f)

        for key in labels.keys():
            # read in the image
            name, ext = os.path.splitext(key)
            splits = name.split('_')
            dir_name = '_'.join(splits[:-1])
            path = os.path.join('data/inputs', dir_name)
            if key in os.listdir(path):
                img = cv2.imread(os.path.join(path, key))
                img = cv2.resize(img, (448, 448))
                data = img.flatten() if data is None else np.vstack([data, img.flatten()])
            else:
                continue
    return data


if __name__ == '__main__':
    data = get_dataset()
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)

    mean = np.reshape(mean, (448, 448, 3), order='C')
    stddev = np.reshape(stddev, (448, 448, 3), order='C')
    cv2.imwrite('summaries/mean.jpg', mean)
    cv2.imwrite('summaries/stddev.jpg', stddev)
