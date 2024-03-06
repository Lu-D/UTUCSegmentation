import os
import cv2
import json
import numpy as np
import glob
from util import write_video

for fname in glob.glob('data/labels/**/**/*.json'):

    with open(fname, 'r') as f:
        labels = json.load(f)

    print('Processing: ', fname)

    combined_frames = []
    save_path = None
    vid_name = None
    for key in labels.keys():
        
        # read in the image
        name, ext = os.path.splitext(key)
        splits = name.split('_')
        dir_name = '_'.join(splits[:-1])
        path = os.path.join('data/inputs',dir_name)
        if key in os.listdir(path):
            img = cv2.imread(os.path.join(path,key))
        else:
            continue

        # make the annotation directory if it doesn't exist already
        anno_path = os.path.join('data/raw+anno',dir_name)
        if not os.path.exists(anno_path):
            os.makedirs(anno_path)

        save_path = anno_path
        vid_name = dir_name
        
        # draw all polygons on the image
        anno = img.copy()
        regions = labels[key]['regions']
        for region in regions.keys():
        
            x = regions[region]['shape_attributes']['all_points_x']
            y = regions[region]['shape_attributes']['all_points_y']

            points = np.array([x,y]).astype('int32').T
            cv2.fillConvexPoly(anno, points, 255)

        img = cv2.resize(img, (448, 448))
        anno = cv2.resize(anno, (448, 448))

        combined = np.concatenate(
            (img.astype(np.uint8), anno.astype(np.uint8)),
            axis=1)
        combined_frames += [combined]

        # show image
        # cv2.imwrite(os.path.join(anno_path,'raw+anno_'+key), combined)
        cv2.imshow('combined', combined)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    out_path = os.path.join(save_path, vid_name+'.mp4')
    print('Saving to: ', out_path)
    write_video(out_path, combined_frames, 5, (448*2, 448))
