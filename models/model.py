import os
import time
import random
import datetime
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from utils import * #get_batch decode
from metrics import *

class RainDropRemoval(object):
	def __init__(self, args):
		self.args = args
		self.chns = 3
		self.scale = 0.5
		self.crop_size = 256
		self.n_levels = 1

		self.data_list = open(args.datalist, 'rt').read().splitlines()
		self.data_list = list(map(lambda x: x.split(' '), self.data_list))
		print(self.data_list)
		self.batch_size = args.batch_size
		self.learning_rate = args.learning_rate
		self.data_size = (len(self.data_list)) // self.batch_size
		self.epoch = args.epoch
		self.max_steps = int(self.epoch * self.data_size)
		self.current_epoch = 0
		self.iscale = 0
		self.mode = 0
		self.model_name = 'rainDrop.model'
		self.train_dir = './checkpoints/'
		self.restore_step = args.restore_step
		if not os.path.exists(self.train_dir):
			os.makedirs(self.train_dir)
		self.tfrecord_dir = args.tfrecord_dir
		print (self.batch_size, self.learning_rate, self.data_size, self.epoch, self.max_steps,
			self.train_dir, self.tfrecord_dir)
	def input_producer(self, batch_size=10):
		def read_data():
			img_a = tf.image.decode_image(tf.read_file(tf.string_join(['./', self.data_queue[0]])),
										channels=3)
			img_b = tf.image.decode_image(tf.read_file(tf.string_join(['./', self.data_queue[1]])),
										channels=3)
			img_a = tf.cast(img_a, tf.float32) / 255.0
			img_b = tf.cast(img_b, tf.float32) / 255.0
			img_a, img_b = tf.unstack(tf.random_crop(tf.stack([img_a,img_b], axis=0), [2, self.crop_size, self.crop_size, self.chns]),
							axis=0)
			
			return img_a, img_b
		with tf.variable_scope('input'):
			List_all = tf.convert_to_tensor(self.data_list, dtype=tf.string)
			gt_list = List_all[:, 0]
			in_list = List_all[:, 1]

			self.data_queue = tf.train.slice_input_producer([in_list, gt_list], capacity=20)
			print("joint path")
			image_in, image_gt= read_data()
			batch_in, batch_gt= tf.train.batch([ image_in, image_gt], batch_size=batch_size, num_threads=1, capacity=20)
		return batch_in, batch_gt

	def generatorSimplified(self, inputs, reuse=False, scope='g_net'):
		n, h, w, c = inputs.get_shape().as_list()

		with tf.variable_scope(scope, reuse=reuse):
			with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
				activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
				weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
				biases_initializer=tf.constant_initializer(0.0)):

				inp_pred = inputs
				inp_rdrop = tf.image.resize_images(inputs, [h, w], method=0)
				inp_pred  = tf.stop_gradient(tf.image.resize_images(inp_pred, [h, w], method=0))

				edgemap = amitedgefinder(inp_pred,10)
				edgemap = tf.cast(edgemap,dtype=tf.float32)
				attention = conv_torque(edgemap)
				dot_product = inp_rdrop * attention
				inp_all   = tf.concat([inp_rdrop,dot_product, inp_pred], axis=3, name='inp')

				# encoder

				attention = slim.conv2d(attention, 32, [5, 5], scope='att1_1')
				conv1_1 = slim.conv2d(inp_all, 32, [5, 5], scope='enc1_1')

				conv1_2,_,_= ResnetBlock_att(conv1_1, 32, 5,attention, scope='enc1_2')
				conv1_3,_,_ = ResnetBlock_att(conv1_2, 32, 5,attention, scope='enc1_3')
				conv1_4,_,_ = ResnetBlock_att(conv1_3, 32, 5,attention, scope='enc1_4')
				conv2_1 = slim.conv2d(conv1_4, 64, [5, 5], stride=2, scope='enc2_1')
				attention128 = slim.conv2d(attention, 64, [5, 5], stride=2, scope='att2_1')

				conv2_2,_,_ = ResnetBlock_att(conv2_1, 64, 5,attention128, scope='enc2_2')
				conv2_3,_,_ = ResnetBlock_att(conv2_2, 64, 5,attention128, scope='enc2_3')
				conv2_4,_,_ = ResnetBlock_att(conv2_3, 64, 5,attention128, scope='enc2_4')
				conv3_1 = slim.conv2d(conv2_4, 128, [5, 5], stride=2, scope='enc3_1')
				attention64 = slim.conv2d(attention128, 128, [5, 5], stride=2, scope='att3_1')

				conv3_2,_,_ = ResnetBlock_att(conv3_1, 128, 5,attention64, scope='enc3_2')
				conv3_3,_,_ = ResnetBlock_att(conv3_2, 128, 5,attention64, scope='enc3_3')
				conv3_4,_,_  = ResnetBlock_att(conv3_3, 128, 5,attention64, scope='enc3_4')


				deconv3_4 = conv3_4

				# decoder
				deconv3_3,_,_ = ResnetBlock_att(deconv3_4, 128, 5,attention64, scope='dec3_3')
				deconv3_2,_,_ = ResnetBlock_att(deconv3_3, 128, 5,attention64, scope='dec3_2')
				deconv3_1,_,_ = ResnetBlock_att(deconv3_2, 128, 5,attention64, scope='dec3_1')
				deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
				cat2      = deconv2_4 + conv2_4
				deconv2_3,_,_ = ResnetBlock_att(cat2, 64, 5,attention128, scope='dec2_3')
				deconv2_2,_,_ = ResnetBlock_att(deconv2_3, 64, 5,attention128, scope='dec2_2')
				deconv2_1,_,_ = ResnetBlock_att(deconv2_2, 64, 5,attention128, scope='dec2_1')
				deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
				cat1      = deconv1_4 + conv1_4
				deconv1_3,_,_ = ResnetBlock_att(cat1, 32, 5,attention, scope='dec1_3')
				deconv1_2,_,_ = ResnetBlock_att(deconv1_3, 32, 5,attention, scope='dec1_2')
				deconv1_1,_,_ = ResnetBlock_att(deconv1_2, 32, 5,attention, scope='dec1_1')
				inp_pred  = slim.conv2d(deconv1_1, self.chns, [5, 5], activation_fn=None, scope='dec1_0')

			return inp_pred


	def build_model(self):
		img_in, img_gt= self.input_producer(self.batch_size)

		tf.summary.image('img_in', im2uint8(img_in))
		tf.summary.image('img_gt', im2uint8(img_gt))
		print ('img_in, img_gt' , img_in.get_shape(), img_gt.get_shape())

		x_unwarp = self.generatorSimplified(img_in, reuse=False)


		self.loss_total = 0

		_, hi, wi, _ = x_unwarp.get_shape().as_list()
		gt_i = tf.image.resize_images(img_gt, [hi, wi], method=0)

		loss = tf.reduce_mean(tf.abs(gt_i - x_unwarp))
		self.loss_total += loss

		all_vars = tf.trainable_variables()
		self.all_vars = all_vars
		self.g_var    = [var for var in all_vars if 'g_net' in var.name]
		for var in all_vars:
			print (var.name)



	def train(self):
		def get_optimizer(loss, global_step=None, var_list=None, is_gradient_clip=False):
			train_op = tf.train.AdamOptimizer(self.lr)
			if is_gradient_clip:
				pass
			else:
				train_op = train_op.minimize(loss, global_step, var_list)
			return train_op

		global_step = tf.Variable(initial_value=self.restore_step, dtype=tf.int32, trainable=False)
		self.global_step = global_step

		# build model

		self.build_model()
		self.lr = tf.train.polynomial_decay(self.learning_rate, global_step, self.max_steps,
			end_learning_rate=0.0, power=0.3)
		tf.summary.scalar('learning_rate', self.lr)

		train_gnet = get_optimizer(self.loss_total, global_step, self.g_var)

		# session and thread
		gpu_options = tf.GPUOptions(allow_growth=True)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.sess = sess
		self.saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=1)

		dir =  './checkpoints/'
		sess.run(tf.global_variables_initializer())

		if self.restore_step!=0:
			self.load(sess, dir, step=self.restore_step)

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)


		for step in range(sess.run(global_step), self.max_steps + 1):
			start_time = time.time()
			_, loss_total_val = sess.run([train_gnet, self.loss_total])
			duration = time.time() - start_time
			# print loss value
			assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

			self.current_epoch = step / self.data_size
			if step % 10 == 0:
				num_examples_per_step = self.batch_size
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)
				format_str = ('%s: step %d, final_step %d, epoch %d, lr %.5f, loss = (%.5f; %.5f, %.5f)(%.1f data/s; %.3f s/bch)')
				print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, self.max_steps,
				 					self.current_epoch, sess.run(self.lr), loss_total_val, 0.0, 0.0, examples_per_sec, sec_per_batch))


			if step>self.restore_step and step % 5000 == 0 and step > 10000 or step == self.max_steps:
				checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
				self.save(sess, checkpoint_path, step)
	def save(self, sess, checkpoint_dir, step):
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		self.saver.save(sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)
	def load(self, sess, checkpoint_dir, step=None):
		print (' [*] Reading checkpoints...')
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

		if step is not None:
			ckpt_name = self.model_name + '-' +  str(step)
			self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
			print (' [*] Reading intermediate checkpoints... Success')
			return str(step)
		elif ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			ckpt_iter = ckpt_name.split('-')[1]
			self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
			print (' [*] Reading checkpoints... Success')
			return ckpt_iter
		else:
			print (' [*] Reading checkpoints... ERROR')
			return False

	def test(self,  inputdata_path, output_path):
		if not os.path.exists(output_path):
			os.makedirs(output_path)
		input_path = inputdata_path + '/data'
		gt_path = inputdata_path + '/gt'
		imgsName = sorted(os.listdir(input_path))
		gtsName  = sorted(os.listdir(gt_path))
		num  = len(imgsName)
		H, W = 480, 720
		inp_chns = 3
		self.batch_size = 1
		inputs = tf.placeholder(shape=[self.batch_size, H, W, inp_chns], dtype=tf.float32)
		outputs = self.generatorSimplified(inputs, reuse=False)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

		self.saver = tf.train.Saver()
		dir =  './checkpoints/'
		self.load(sess, dir, step=self.restore_step)
		total_psnr = 0
		total_rgbpsnr = 0
		total_ssim = 0

		for i in range(num):
			gt = scipy.misc.imread(os.path.join(gt_path, gtsName[i]))
			rDrop = scipy.misc.imread(os.path.join(input_path, imgsName[i]))
			h, w, c = rDrop.shape
			rDrop = rDrop[:,:,0:3]
			print( rDrop.shape)
			# make sure the width is larger than the height
			rot = False
			if h > w:
				rDrop = np.transpose(rDrop, [1, 0, 2])
				rot = True
			h = int(rDrop.shape[0])
			w = int(rDrop.shape[1])
			resize = False
			if h > H or w > W:
				scale = min(1.0 * H / h, 1.0 * W / w)
				new_h = int(h * scale)
				new_w = int(w * scale)
				rDrop = scipy.misc.imresize(rDrop, [new_h, new_w], 'bicubic')
				resize = True
				rDropPad = np.pad(rDrop, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
			else:
				rDropPad = np.pad(rDrop, ((0, H - h), (0, W - w), (0, 0)), 'edge')
			rDropPad = np.expand_dims(rDropPad, 0)


			start = time.time()
			derDrop = sess.run(outputs, feed_dict={inputs: rDropPad / 255.0})
			duration = time.time() - start

			print ('Saving results: %s ... %4.3fs' % (os.path.join(output_path, imgsName[i]), duration))

			res = im2uint8(derDrop[0, :, :, :])
			# crop the image into original size
			if resize:
				res = res[:new_h, :new_w]
				res = scipy.misc.imresize(res, [h, w], 'bicubic')
			else:
				res = res[:h, :w, :]
			if rot:
				res = np.transpose(res, [1, 0, 2])
			scipy.misc.imsave(os.path.join(output_path, imgsName[i]), res)
			res = align_to_four(res)
			gt = align_to_four(gt)

			ssim = calc_ssim(res, gt)
			psnr = calc_psnr(res, gt)
			rgbpsnr = psnrrgb(res,gt)
			print('picture ' + str(i) + '\'s psnr:', psnr)
			print('picture ' + str(i) + '\'s psnr(rgb):', rgbpsnr)
			print('picture ' + str(i) + '\'s ssim:', ssim)
			total_psnr += psnr
			total_rgbpsnr +=rgbpsnr
			total_ssim += ssim
		print('average psnr:', total_psnr / num)
		print('average psnr(rgb):', total_rgbpsnr / num)
		print('average ssim:', total_ssim / num)
