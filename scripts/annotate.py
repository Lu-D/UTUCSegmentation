import os
import cv2
import json
import numpy as np
import glob
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Annotate arguments')
    parser.add_argument('--labels', '-l', type=str, default='../data/labels/train', help='path to labels. will recursively search.')
    parser.add_argument('--inputs', '-i', type=str, default='../data/inputs', help='path to inputs')

    return parser.parse_args()


def annotate(pinputs, plabels):
    for fname in glob.glob(os.path.join(plabels, '**/*.json')):

        with open(fname, 'r') as f:
            labels = json.load(f)

        for key in labels.keys():

            # read in the image
            name,ext = os.path.splitext(key)
            splits = name.split('_')
            dir_name = '_'.join(splits[:-1])
            path = os.path.join(pinputs,dir_name)
            if key in os.listdir(path):
                img = cv2.imread(os.path.join(path,key))
            else:
                continue

            print('Visualizing: ', key)

            # make the annotation directory if it doesn't exist already
            anno_path = os.path.join(plabels, dir_name)
            if not os.path.exists(anno_path):
                os.makedirs(anno_path)

            # draw all polygons on the image
            regions = labels[key]['regions']
            for region in regions.keys():

                x = regions[region]['shape_attributes']['all_points_x']
                y = regions[region]['shape_attributes']['all_points_y']

                points = np.array([x,y]).astype('int32').T
                cv2.fillConvexPoly(img, points, 255)

            # save image (Pyplot doesn't handle RGB images well --> blue overlay)
    #        plt.imshow(img)
    #        plt.savefig(os.path.join(anno_path,'anno_'+key))

            # show image
            # cv2.imshow('mask', img)
            cv2.imwrite(os.path.join(anno_path, 'anno_'+key),img)
            #cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()

    annotate(args.inputs, args.labels)