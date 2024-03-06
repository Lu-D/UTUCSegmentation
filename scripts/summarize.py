"""
Included in History functionality. Summarization implemented in train.py.
"""
import os
from util.History import History
import argparse
import seaborn as sns

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='summary args')
	parser.add_argument('--path','-p',type=str,default=os.getcwd(),help='path to file with summary data')
	parser.add_argument('--dice',action='store_true',help='summarize Dice scores')

	args = parser.parse_args()
	if args.dice:
		dice = History(path=args.path)
		p = dice.plot()
		basename = os.path.basename(args.path)
		base,ext = os.path.splitext(basename)
		p.figure.savefig(os.path.join('summaries',base+'.png'))
