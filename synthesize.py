import os
import cv2
from tqdm import tqdm
import numpy as np
import logging
import torch
from util.Segmentation import Segmentation
from util.Video import get_frames
from util import contour_crop, write_video, get_parser
import glob
from model import get_net
import torch.nn as nn
import copy
import csv
import pandas as pd

sigmoid = nn.Sigmoid()


def get_args():
    parser = get_parser(train=False)
    return parser.parse_args()


def process_video(path, net, args):
    net_basename = os.path.basename(args.load)
    net_base, net_ext = os.path.splitext(net_basename)
    basename = os.path.basename(path)
    base, ext = os.path.splitext(basename)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    save_dir = os.path.join(args.save_dir, net_base)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    offset = 0
    delay = (0, 0)
    video = cv2.VideoCapture(path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = video.get(3)
    frame_height = video.get(4)
    frames = get_frames(video)

    combined_frames = []
    pixel_count = []
    pixel_total = []
    with tqdm(total=video.get(cv2.CAP_PROP_FRAME_COUNT), desc='Processing') as pbar:

        out_dir = os.path.join(save_dir, base)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        img_dir = os.path.join(out_dir, 'img')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        anno_dir = os.path.join(out_dir, 'anno')
        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)
        hmap_dir = os.path.join(out_dir, 'hmap')
        if not os.path.exists(hmap_dir):
            os.makedirs(hmap_dir)

        count = 0
        for image in frames:
            image = contour_crop(image)
            image = cv2.resize(image, (448, 448))
            unmod = np.copy(image)

            ### MUST NORMALIZE IMAGES IF MODEL TRAINS ON NORMALIZED IMAGES
            # o/w synthesis won't work -> check in FrameDataset
            # normalization yields worse performance
            # image = (image - mean) / stddev

            dtype = torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            unmod = unmod.transpose((2, 0, 1))
            image = image.transpose((2, 0, 1))
            unmod = np.expand_dims(unmod, axis=0)
            image = np.expand_dims(image, axis=0)

            image = torch.from_numpy(image).type(dtype)

            img = image.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                # since BCE with logits loss implicitly applies sigmoid
                # it was left out of underlying trained model -> apply at end to obtain probabilities
                pred = net(img)
                pred = sigmoid(pred)

            pred = pred.detach().cpu().numpy()

            unmod_lst = [np.moveaxis(i, 0, -1) for i in np.copy(unmod)]

            pred_seg = Segmentation(np.copy(unmod), pred, None)
            masked_imgs = pred_seg.apply_masks(smooth=args.smooth)

            pixel_count.append(np.count_nonzero((pred_seg.masks[0] >= 0.5)))
            pixel_total.append(pred_seg.masks[0].size)

            for masked_img, pred_map, unmod in zip(masked_imgs, pred_seg.masks, unmod_lst):
                # hmaps
                scaled_map = (pred_map * 255).astype(np.uint8)
                hmap = cv2.applyColorMap(scaled_map, cv2.COLORMAP_JET)

                # combine and log
                combined = np.concatenate(
                    (unmod.astype(np.uint8), masked_img.astype(np.uint8), hmap.astype(np.uint8)),
                    axis=1)
                combined_frames.append(combined)

                ######### comment out to speed up runtime
                # if count % args.freq == 0:
                #     cv2.imwrite(os.path.join(img_dir, base + '_' + str(int(count / args.frequency) + 1) + '.jpg'),
                #                 unmod)
                #     cv2.imwrite(
                #         os.path.join(anno_dir, 'anno_' + base + '_' + str(int(count / args.frequency) + 1) + '.jpg'),
                #         masked_img)
                #     cv2.imwrite(
                #         os.path.join(hmap_dir, 'hmap_' + base + '_' + str(int(count / args.frequency) + 1) + '.jpg'),
                #         hmap)
                # cv2.imshow('combined', combined) # can't show if running on ACCRE
                #########

                count += 1

            pbar.update(1)

        write_video(os.path.join(out_dir, args.net + '_combined_' + basename), combined_frames, fps,
                    (448 * 3, 448))
        video.release()
        # cv2.destroyAllWindows()

        with open(os.path.join(out_dir, args.net + '_positive_count_' + base + '.txt'), 'w') as fp:
            fp.write('\n'.join([str(num) for num in pixel_count]))
            fp.close()

        with open(os.path.join(out_dir, args.net + '_pixel_total_' + base + '.txt'), 'w') as fp:
            fp.write('\n'.join([str(num) for num in pixel_total]))
            fp.close()

        with open(os.path.join(out_dir, args.net + '_video_info_' + base + '.txt'), 'w') as fp:
            fp.write('\n'.join([base, str(fps), str(frame_height), str(frame_width), str(pixel_total[0])]))
            fp.close()

    return pixel_count, pixel_total, [base, fps, frame_height, frame_width, pixel_total[0]]


if __name__ == '__main__':
    # os.chdir('..')  # FOR ACCRE IF RUNNING FROM SLURM DIR
    args = get_args()
    input_files = glob.glob(os.path.join(args.inputs, './**/*.mp4'), recursive=True)
    print(input_files)
    # pixel_counts = []
    # total_counts = []
    # video_names = []

    store_counts = []
    store_percentages = []
    base_names = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = get_net(args, device)
    net.eval()

    for file in input_files:
        print(file)
        plist, tlist, details = process_video(file, net, args)

        # video_names.append(details[0]) # end of process_video()

        store_counts.append(plist)
        store_percentages.append(np.asarray(plist) / details[4])
        base_names.append(details[0])

    net_basename = os.path.basename(args.load)
    net_base, net_ext = os.path.splitext(net_basename)
    save_dir = os.path.join(args.save_dir, net_base)

    with open(save_dir + "pixel_counts.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(base_names)
        write.writerows(store_counts)
        f.close()

    with open(save_dir + "pixel_percentages.csv", "w") as f:
        write = csv.writer(f)
        write.writerow(base_names)
        write.writerows(store_percentages)
        f.close()
