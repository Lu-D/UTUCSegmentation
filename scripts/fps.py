import os
from os.path import join
import cv2
import argparse
import numpy as np
from util import contour_crop


def get_args():
    parser = argparse.ArgumentParser(description='Estimating FPS args')

    parser.add_argument('--pvideo', '-v', type=str, default=None, help='path to video file')
    parser.add_argument('--pcap', '-c', type=str, default=None, help='path to captured frames dir')

    return parser.parse_args()


def find_frames(caps, video):
    """
    Find a frame's index that is expected to be in a list of frames, i.e.,
    the frame is saved from a process at a lower FPS than the video recording
    that it is drawn from.
    """
    check = {i: cap for i, cap in enumerate(caps)}
    inds = [-1 for _ in caps]
    ind = 0
    success, frame = video.read()
    while success:
        if ind % 1000 == 0:
            print('Checking index', ind, 'with progress', inds)

        try:  # might not find contours
            frame = contour_crop(frame)
            frame = cv2.resize(frame, (448, 448))
        except IndexError:  # frame still exists so skip
            success, frame = video.read()
            ind += 1
            continue

        # check remaining caps for closeness to current frame
        remove = []
        for key in check.keys():
            cap = check[key]
            if np.allclose(frame, cap):
                inds[key] = ind
                remove += [key]

        # remove any caps from checklist if found
        for key in remove:
            del check[key]

        success, frame = video.read()
        ind += 1

    return inds


def estimate_capture_fps(pvideo, pcap):
    """
    Estimates the FPS of captured frames relative to the recorded video.

    Parameters:
        pvideo = path to video
        pcap = path to captures directory
    Returns:
        cap_fps = FPS for captured frames
    """
    vid = cv2.VideoCapture(pvideo)
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    vid_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Video FPS = ', vid_fps)
    tduration = vid_count / vid_fps
    print('Total duration = ', tduration)

    # get capture and video frames
    fcaps = sorted(os.listdir(pcap))
    first_frame = cv2.imread(join(pcap, fcaps[0]))
    last_frame = cv2.imread(join(pcap, fcaps[-1]))
    cap_count = len(fcaps)

    print('Captures=%d, Frames=%d' % (cap_count, vid_count))

    # estimate cap FPS based on relative lengths
    # assuming caps all contained in vid
    est_cap_fps = cap_count / tduration
    print('Estimated cap FPS = ', est_cap_fps)

    # get inds of first and last frame
    first_ind, last_ind = find_frames([first_frame, last_frame], vid)

    # error handling
    missed = False
    if first_ind == -1:
        print('First frame not found in video')
        missed = True
    else:
        print('Found first frame!')
    if last_ind == -1:
        print('Last frame not found in video')
        missed = True
    else:
        print('Found last frame!')
    if not missed and last_ind <= first_ind:
        print('Last ind oob')
        missed = True

    print('First=%d, Last=%d' % (first_ind, last_ind))

    if missed:
        return

    # compute times and fps
    duration = (last_ind - first_ind) / vid_fps
    cap_fps = cap_count / duration

    print('Absolute capture FPS = ', cap_fps)
    return cap_fps


if __name__ == '__main__':
    # os.chdir('..')

    args = get_args()
    cap_fps = estimate_capture_fps(args.pvideo, args.pcap)
