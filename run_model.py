# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 20:25:47 2018
@author: tyty
@e-mail: bravotty@protonmail.com
"""

import os
import argparse
import tensorflow as tf
from models import model


def parse_args():
	parser = argparse.ArgumentParser(description='rainDrop arguments')
	# train params
	parser.add_argument('--phase', type=str, default='test', help='determine whether "train" or "test" ')
	parser.add_argument('--datalist', type=str, default='./raindrop.txt', help='training datalist')
	parser.add_argument('--tfrecord_dir', type=str, default='./tf/train2.tfrecords', help='the location of tfrecord')
	parser.add_argument('--batch_size', type=int, default=10, help='training batch size')
	parser.add_argument('--epoch', type=int, default= 4000, help='training epoch number,1000 for 4 times data augmentation')
	parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
	parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
	parser.add_argument('--modelname', type=str, default='rainDrop', help='model type')

	# test params
	parser.add_argument('--restore_step', type=int, default='258000', help='restore from ckpt point')

	parser.add_argument('--inputdata_path', type=str, default='./testing_real/',
	                        help='input path for testing images')
	parser.add_argument('--output_path', type=str, default='./testing_result',
	                        help='output path for testing images')
	args = parser.parse_args()
	return args

def main(_):
	args = parse_args()
	# set gpu mode
	if int(args.gpu_id) >= 0:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
	else:
		os.environ['CUDA_VISIBLE_DEVICES'] = ''
	if args.phase == 'test':
		rainDrop = model.RainDropRemoval(args)
		rainDrop.test(args.inputdata_path,args.output_path)
	elif args.phase == 'train':
		rainDrop = model.RainDropRemoval(args)
		rainDrop.train()
if __name__ == '__main__':
	tf.app.run()
