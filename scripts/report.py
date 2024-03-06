import os
import glob
import json
from PIL import Image
import argparse


def get_args():
	parser = argparse.ArgumentParser(description='Usage: python report.py -p <PATH>')

	fs = "root \n" \
		 "-> vid dirs \n" \
		 "-> -> anno, hmap, img dirs \n" \
		 "-> -> -> jpg images \n"
	parser.add_argument('--path', '-p', type=str, default=os.getcwd(),
						help="path to output data; expects file structure: " + fs)

	return parser.parse_args()


def generate_report(path):
	os.chdir(path)
	dirs = os.listdir()
	for dir in dirs:
		img_dir = os.path.join(dir, 'img')
		anno_dir = os.path.join(dir, 'anno')
		for fname in glob.glob(img_dir):
			orig = Image.open(os.path.join(img_dir, fname))
			anno = Image.open(os.path.join(anno_dir, 'anno'+fname))


def generate_dataset_report():
	if not os.path.exists('summaries/reports'):
		os.makedirs('summaries/reports')

	for fname in glob.glob('data/labels/json/*.json'):
		img_lst = []

		with open(fname, 'r') as f:
			labels = json.load(f)

		for key in labels.keys():
			# read in the original
			name, ext = os.path.splitext(key)
			splits = name.split('_')
			dir_name = '_'.join(splits[:-1])
			path = os.path.join('data/inputs', dir_name)
			if key in os.listdir(path):
				fname = os.path.join(path, key)
				orig = Image.open(fname)
				print('Loaded ', fname)
			else:
				continue

			# read in annotated image
			anno_path = os.path.join('data/labels', dir_name)
			anno_name = 'anno_' + key
			if anno_name in os.listdir(anno_path):
				fname = os.path.join(anno_path, anno_name)
				anno = Image.open(fname)
				print('Loaded ', fname)
			else:
				continue

			imgs = [orig, anno]
			# https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
			widths, heights = zip(*(i.size for i in imgs))
			total_width = sum(widths)
			max_height = max(heights)

			comp_im = Image.new('RGB', (total_width, max_height))

			x_offset = 0
			for img in imgs:
				comp_im.paste(img, (x_offset, 0))
				x_offset += img.size[0]

			img_lst += [comp_im]

		name, _ = os.path.splitext(fname)
		splits = name.split('/')
		vid_name = splits[1]
		img0 = img_lst.pop(0)
		img0.save('summaries/reports/' + vid_name + '_report.pdf', 'PDF', resolution=100.0, save_all=True,
				  append_images=img_lst)
		print('Report saved to summaries dir')


if __name__ == '__main__':
	args = get_args()
	generate_dataset_report()
