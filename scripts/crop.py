"""
Cropping script for video files.
Refer to the util module for the implementation and descriptions. 
Suggestion: use contour cropping because it is fast and automatic (no extra args). 
However, contour cropping is finicky and will definitely require quality control.
"""
import os
from util import manual_crop, center_color_crop, contour_crop
from util.Video import Video
import argparse

exts = ['.mp4','.avi',',mov']

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def MANUAL(args):
	for fname in os.listdir(args.path):
		_, ext = os.path.splitext(fname)
		if not os.path.isdir(os.path.join(args.path,fname)) and ext in exts:
			vid = Video(os.path.join(args.path,fname))
			print("Manually cropping video: ", fname)
			vid.apply(args.fps,manual_crop,x=args.x,y=args.y,w=args.width,h=args.length)

def COLOR(args):
	for fname in os.listdir(args.path):
		_, ext = os.path.splitext(fname)
		if not os.path.isdir(os.path.join(args.path,fname)) and ext in exts:
			vid = Video(os.path.join(args.path,fname))
			print("Color cropping video: ", fname)
			vid.apply(args.fps,center_color_crop,threshold=args.threshold,max_gap=args.max_gap)

def CONTOUR(args):
	for fname in os.listdir(args.path):
		_, ext = os.path.splitext(fname)
		if not os.path.isdir(os.path.join(args.path,fname)) and ext in exts:
			vid = Video(os.path.join(args.path,fname))
			print("Contour cropping video: ", fname)
			vid.apply(args.fps,contour_crop)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Arguments required for a successful cropping.')
	parser.add_argument('--path','-p',type=str,default=os.getcwd(),help="path to videos")
	parser.add_argument('--fps',type=int,default=20,help='frames per second to save')

	# manual cropping options
	parser.add_argument('--manual',action='store_true',help='perform MANUAL crop')
	parser.add_argument('--x','-x',type=int,default=0,help='MANUAL: starting x')
	parser.add_argument('--y','-y',type=int,default=0, help='MANUAL: starting y')
	parser.add_argument('--length','-l',type=int,default=256,help='MANUAL: crop length')
	parser.add_argument('--width','-w',type=int,default=256,help='MANUAL: crop width')

	# centered color cropping options
	parser.add_argument('--color',action='store_true',help='perform COLOR crop')
	parser.add_argument('--threshold',type=int,default=10,help='color threshold [0=black]')
	parser.add_argument('--max_gap',type=int,default=25,help='max gap for colored pixels from center component')

	# contour cropping options
	parser.add_argument('--contour',action='store_true',help='perform CONTOUR crop')

	args = parser.parse_args()
	if args.manual:
		MANUAL(args)
	elif args.color:
		COLOR(args)
	elif args.contour:
		CONTOUR(args)
	else:
		raise ValueError('Please specify a crop to perform {--manual,--color}.')
