import os
from os.path import join
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import json
import cv2
import copy
import warnings

from dataloader import *

warnings.filterwarnings("ignore")
plt.ion()


class FrameDataset(Dataset):

    # def __init__(self, file, root, mode='/train', transform=None, test=False):
    def __init__(self, pinputs, plabels, mode='train', transform=None, json_lib=False, args=None):

        # set paths for images and labels
        if mode == 'test':
            label_files = glob.glob(join(plabels, 'test/**/*.json'), recursive=True)
            label_images = glob.glob(join(pinputs, 'test/**/*.jpg'), recursive=True)
        elif mode == 'train':
            label_files = glob.glob(join(plabels, 'train/**/*.json'), recursive=True)
            label_images = glob.glob(join(pinputs, 'train/**/*.jpg'), recursive=True)
        else:
            raise ValueError('Unknown arg %s' % mode)

        self.images = []
        self.annotated_regions = []
        for fname in label_files: # load images with labels
            with open(fname, 'r') as f:
                json_lib = json.load(f)

                for key in json_lib.keys():
                    # read in the image
                    name, ext = os.path.splitext(key)
                    splits = name.split('_')
                    dir_name = '_'.join(splits[:-1])
                    path = os.path.join(pinputs, mode, dir_name)
                    inp_name = os.path.join(path, key)
                    self.images.append(inp_name)

                    self.annotated_regions.append(json_lib[key]['regions'])

                    # remove images with labels from the overall list
                    # kinda slow but quickest to code rn
                    label_images.remove(os.path.join(pinputs, mode, dir_name, key))
        # insert empty elements into the dataset
        empty_poly_annotation = {'0':{'shape_attributes':{'name':'polygon', 'all_points_x':[], 'all_points_y':[]}, 'region_attributes':{'label':'tumor'}}}
        for empty_image in label_images:
            self.images.append(empty_image)
            self.annotated_regions.append(empty_poly_annotation)
        self.args = args
        self.transform = transform
        self.mean = cv2.imread('summaries/mean.jpg')
        self.stddev = cv2.imread('summaries/stddev.jpg')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # return data pair
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.images[idx]
        image = cv2.imread(img_name)

        mask = np.zeros((image.shape[0], image.shape[1]))
        # mask = image.copy()
        for region in self.annotated_regions[idx].keys():
            x = self.annotated_regions[idx][region]['shape_attributes']['all_points_x']
            y = self.annotated_regions[idx][region]['shape_attributes']['all_points_y']

            points = np.array([x, y]).astype('int32').T
            cv2.fillConvexPoly(mask, points, 1.)


        #### RESIZE

        # image = cv2.resize(image, (448,448))
        # NORMALIZATION
        # normalization yields worse performance
        # image = (image - self.mean) / self.stddev

        # mask = cv2.resize(mask, (448,448))

        mask = np.expand_dims(mask, axis=2)


        # if 'data/inputs/test/ww10212021_manual_crop/ww10212021_manual_crop_95.jpg' == img_name:
        #     print('nolabels')

        unmod = copy.deepcopy(image)

        # using albumentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        if self.args:
            unmod = cv2.resize(unmod, (self.args.height,self.args.width))

        image = np.moveaxis(image, -1, 0)
        unmod = np.moveaxis(unmod, -1, 0)
        mask = np.moveaxis(mask, -1, 0)


        sample = {'unmod': torch.from_numpy(unmod).type(torch.FloatTensor),
                  'image': torch.from_numpy(image).type(torch.FloatTensor),
                  'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                  'name': img_name}

        return sample

if __name__ == '__main__':
    dataset = FrameDataset(transform=transforms.Compose([ToTensor(), RandomColorJitter()]))
    dataset_get = dataset[1]
    # plt.imshow(dataset_get['mask'])
    # plt.savefig('test2.jpg')

    image = dataset_get['image'].numpy().transpose((1,2,0))
    unmod = dataset_get['unmod'].numpy().transpose((1,2,0))
    cv2.imwrite('summaries/figs/unmod.jpg', unmod)
    cv2.imwrite('summaries/figs/color_jitter.jpg', image)
