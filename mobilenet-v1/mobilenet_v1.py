#!/usr/bin/env python3

import sys
import caffe
import numpy as np
from scipy.signal import convolve2d

def kernel_reverse(convk):
	conv_k = convk.reshape(-1, convk.shape[-2], convk.shape[-1])
	res    = np.empty(conv_k.shape, dtype=np.float32)
	for idx in range(conv_k.shape[0]):
		res[idx] = conv_k[idx, ::-1, ::-1]
	res = res.reshape(convk.shape)
	return res

def get_bn(wtf):
	what = wtf[1].data
	# what = np.sqrt(np.float32(0.00001) + np.square(what))
	what = np.sqrt(np.float32(0.00001) + what)
	return (wtf[0].data, what, wtf[2].data)

def get_scale(wtf):
	return (wtf[0].data, wtf[1].data)

class MobileNetV1(object):
	def __init__(self, deploy='./mobilenet_deploy.prototxt', cmode='./mobilenet.caffemodel'):
		mbv1 = caffe.Net(deploy, cmode, caffe.TEST)
		self.conv1                 = kernel_reverse(mbv1.params['conv1'][0].data)
		self.conv1_bn              = get_bn(mbv1.params['conv1/bn'])
		self.conv1_scale           = get_scale(mbv1.params['conv1/scale'])

		self.conv2_1_dw            = kernel_reverse(mbv1.params['conv2_1/dw'][0].data)
		self.conv2_1_dw_bn         = get_bn(mbv1.params['conv2_1/dw/bn'])
		self.conv2_1_dw_scale      = get_scale(mbv1.params['conv2_1/dw/scale'])
		self.conv2_1_sep           = mbv1.params['conv2_1/sep'][0].data
		self.conv2_1_sep_bn        = get_bn(mbv1.params['conv2_1/sep/bn'])
		self.conv2_1_sep_scale     = get_scale(mbv1.params['conv2_1/sep/scale'])

		self.conv2_2_dw            = kernel_reverse(mbv1.params['conv2_2/dw'][0].data)
		self.conv2_2_dw_bn         = get_bn(mbv1.params['conv2_2/dw/bn'])
		self.conv2_2_dw_scale      = get_scale(mbv1.params['conv2_2/dw/scale'])
		self.conv2_2_sep           = mbv1.params['conv2_2/sep'][0].data
		self.conv2_2_sep_bn        = get_bn(mbv1.params['conv2_2/sep/bn'])
		self.conv2_2_sep_scale     = get_scale(mbv1.params['conv2_2/sep/scale'])

		self.conv3_1_dw            = kernel_reverse(mbv1.params['conv3_1/dw'][0].data)
		self.conv3_1_dw_bn         = get_bn(mbv1.params['conv3_1/dw/bn'])
		self.conv3_1_dw_scale      = get_scale(mbv1.params['conv3_1/dw/scale'])
		self.conv3_1_sep           = mbv1.params['conv3_1/sep'][0].data
		self.conv3_1_sep_bn        = get_bn(mbv1.params['conv3_1/sep/bn'])
		self.conv3_1_sep_scale     = get_scale(mbv1.params['conv3_1/sep/scale'])

		self.conv3_2_dw            = kernel_reverse(mbv1.params['conv3_2/dw'][0].data)
		self.conv3_2_dw_bn         = get_bn(mbv1.params['conv3_2/dw/bn'])
		self.conv3_2_dw_scale      = get_scale(mbv1.params['conv3_2/dw/scale'])
		self.conv3_2_sep           = mbv1.params['conv3_2/sep'][0].data
		self.conv3_2_sep_bn        = get_bn(mbv1.params['conv3_2/sep/bn'])
		self.conv3_2_sep_scale     = get_scale(mbv1.params['conv3_2/sep/scale'])

		self.conv4_1_dw            = kernel_reverse(mbv1.params['conv4_1/dw'][0].data)
		self.conv4_1_dw_bn         = get_bn(mbv1.params['conv4_1/dw/bn'])
		self.conv4_1_dw_scale      = get_scale(mbv1.params['conv4_1/dw/scale'])
		self.conv4_1_sep           = mbv1.params['conv4_1/sep'][0].data
		self.conv4_1_sep_bn        = get_bn(mbv1.params['conv4_1/sep/bn'])
		self.conv4_1_sep_scale     = get_scale(mbv1.params['conv4_1/sep/scale'])

		self.conv4_2_dw            = kernel_reverse(mbv1.params['conv4_2/dw'][0].data)
		self.conv4_2_dw_bn         = get_bn(mbv1.params['conv4_2/dw/bn'])
		self.conv4_2_dw_scale      = get_scale(mbv1.params['conv4_2/dw/scale'])
		self.conv4_2_sep           = mbv1.params['conv4_2/sep'][0].data
		self.conv4_2_sep_bn        = get_bn(mbv1.params['conv4_2/sep/bn'])
		self.conv4_2_sep_scale     = get_scale(mbv1.params['conv4_2/sep/scale'])

		self.conv5_1_dw            = kernel_reverse(mbv1.params['conv5_1/dw'][0].data)
		self.conv5_1_dw_bn         = get_bn(mbv1.params['conv5_1/dw/bn'])
		self.conv5_1_dw_scale      = get_scale(mbv1.params['conv5_1/dw/scale'])
		self.conv5_1_sep           = mbv1.params['conv5_1/sep'][0].data
		self.conv5_1_sep_bn        = get_bn(mbv1.params['conv5_1/sep/bn'])
		self.conv5_1_sep_scale     = get_scale(mbv1.params['conv5_1/sep/scale'])

		self.conv5_2_dw            = kernel_reverse(mbv1.params['conv5_2/dw'][0].data)
		self.conv5_2_dw_bn         = get_bn(mbv1.params['conv5_2/dw/bn'])
		self.conv5_2_dw_scale      = get_scale(mbv1.params['conv5_2/dw/scale'])
		self.conv5_2_sep           = mbv1.params['conv5_2/sep'][0].data
		self.conv5_2_sep_bn        = get_bn(mbv1.params['conv5_2/sep/bn'])
		self.conv5_2_sep_scale     = get_scale(mbv1.params['conv5_2/sep/scale'])

		self.conv5_3_dw            = kernel_reverse(mbv1.params['conv5_3/dw'][0].data)
		self.conv5_3_dw_bn         = get_bn(mbv1.params['conv5_3/dw/bn'])
		self.conv5_3_dw_scale      = get_scale(mbv1.params['conv5_3/dw/scale'])
		self.conv5_3_sep           = mbv1.params['conv5_3/sep'][0].data
		self.conv5_3_sep_bn        = get_bn(mbv1.params['conv5_3/sep/bn'])
		self.conv5_3_sep_scale     = get_scale(mbv1.params['conv5_3/sep/scale'])

		self.conv5_4_dw            = kernel_reverse(mbv1.params['conv5_4/dw'][0].data)
		self.conv5_4_dw_bn         = get_bn(mbv1.params['conv5_4/dw/bn'])
		self.conv5_4_dw_scale      = get_scale(mbv1.params['conv5_4/dw/scale'])
		self.conv5_4_sep           = mbv1.params['conv5_4/sep'][0].data
		self.conv5_4_sep_bn        = get_bn(mbv1.params['conv5_4/sep/bn'])
		self.conv5_4_sep_scale     = get_scale(mbv1.params['conv5_4/sep/scale'])

		self.conv5_5_dw            = kernel_reverse(mbv1.params['conv5_5/dw'][0].data)
		self.conv5_5_dw_bn         = get_bn(mbv1.params['conv5_5/dw/bn'])
		self.conv5_5_dw_scale      = get_scale(mbv1.params['conv5_5/dw/scale'])
		self.conv5_5_sep           = mbv1.params['conv5_5/sep'][0].data
		self.conv5_5_sep_bn        = get_bn(mbv1.params['conv5_5/sep/bn'])
		self.conv5_5_sep_scale     = get_scale(mbv1.params['conv5_5/sep/scale'])

		self.conv5_6_dw            = kernel_reverse(mbv1.params['conv5_6/dw'][0].data)
		self.conv5_6_dw_bn         = get_bn(mbv1.params['conv5_6/dw/bn'])
		self.conv5_6_dw_scale      = get_scale(mbv1.params['conv5_6/dw/scale'])
		self.conv5_6_sep           = mbv1.params['conv5_6/sep'][0].data
		self.conv5_6_sep_bn        = get_bn(mbv1.params['conv5_6/sep/bn'])
		self.conv5_6_sep_scale     = get_scale(mbv1.params['conv5_6/sep/scale'])

		self.conv6_dw              = kernel_reverse(mbv1.params['conv6/dw'][0].data)
		self.conv6_dw_bn           = get_bn(mbv1.params['conv6/dw/bn'])
		self.conv6_dw_scale        = get_scale(mbv1.params['conv6/dw/scale'])
		self.conv6_sep             = mbv1.params['conv6/sep'][0].data
		self.conv6_sep_bn          = get_bn(mbv1.params['conv6/sep/bn'])
		self.conv6_sep_scale       = get_scale(mbv1.params['conv6/sep/scale'])

		tmpv                       = mbv1.params['fc7'][0].data
		self.inner                 = tmpv.reshape(tmpv.shape[0], tmpv.shape[1])
		self.bias                  = mbv1.params['fc7'][1].data
		del mbv1

	# convolution mode: SAME, stride: 2
	def convolve_norm(self, convk, pImg, stride=2):
		if convk.ndim != 4 or pImg.ndim != 3:
			print("Error, invalid convolution dimensions: {0}, {1}".format(
				convk.ndim, pImg.ndim), file=sys.stderr)
			return None
		if convk.shape[1] != pImg.shape[0]:
			print("Error, convolution shape mismatch: {0}, {1}".format(
				convk.shape, pImg.shape), file=sys.stderr)
			return None
		siz1, siz2 = pImg.shape[1], pImg.shape[2]
		res = np.empty((convk.shape[0], siz1, siz2), dtype=np.float32)
		for idx in range(convk.shape[0]):
			tmpVal = np.empty((convk.shape[1], siz1, siz2), dtype=np.float32)
			for jdx in range(convk.shape[1]):
				tmpVal[jdx] = convolve2d(pImg[jdx], convk[idx, jdx], mode='same')
			tmpVal = np.sum(tmpVal, axis=0)
			res[idx] = tmpVal
		if stride > 1:
			res = res[:, ::stride, ::stride]
		return res

	def batchnorm(self, pImg, bn):
		bn_m, bn_v = bn[0], bn[1]
		if bn_m.shape[0] != bn_v.shape[0] or pImg.shape[0] != bn_m.shape[0]:
			print("Error, invalid batch normalization sizes: {0}, {1}, {2}".format(
				bn_m.shape[0], bn_v.shape[0], bn_m.shape[0]), file=sys.stderr)
			return None
		res = np.empty(pImg.shape, dtype=np.float32)
		for idx in range(pImg.shape[0]):
			res[idx] = (pImg[idx] - bn_m[idx]) / bn_v[idx]
		return res

	def bn_scale(self, pImg, scale, relu=True):
		s0, s1 = scale[0], scale[1]
		if s0.shape[0] != s1.shape[0] or pImg.shape[0] != s0.shape[0]:
			print("Error, invalid scale sizes: {0}, {1}, {2}".format(
				s0.shape[0], s1.shape[0], pImg.shape[0]), file=sys.stderr)
			return None
		res = np.empty(pImg.shape, dtype=np.float32)
		for idx in range(pImg.shape[0]):
			res[idx] = pImg[idx] * s0[idx] + s1[idx]
		if relu:
			res = np.where(res > 0, res, 0)
		return res

	def convolve_depth(self, pImg, convk, stride=2):
		if convk.shape[1] != 1:
			print("Error, invalid depth convolution: {0}".format(pImg.shape), file=sys.stderr)
			return None
		siz1, siz2 = pImg.shape[1], pImg.shape[2]
		res = np.empty((convk.shape[0], siz1, siz2), dtype=np.float32)
		for idx in range(convk.shape[0]):
			res[idx] = convolve2d(pImg[idx], convk[idx, 0], mode='same')
		if stride > 1:
			res = res[:, ::stride, ::stride]
		return res

	def convolve_point(self, pImg, convk, stride=1):
		if pImg.shape[0] != convk.shape[1]:
			print("Error, invalid pointwise convolutions shapes: {0}, {1}".format(
				pImg.shape[0], convk.shape[1]), file=sys.stderr)
			return None
		conv_k = convk.reshape(convk.shape[0], convk.shape[1])
		siz1, siz2 = pImg.shape[1], pImg.shape[2]
		res = np.empty((conv_k.shape[0], siz1, siz2), dtype=np.float32)
		for idx in range(conv_k.shape[0]):
			tmpVal = np.empty(pImg.shape, dtype=np.float32)
			for jdx in range(pImg.shape[0]):
				tmpVal[jdx] = conv_k[idx, jdx] * pImg[jdx]
			tmpVal = np.sum(tmpVal, axis=0)
			res[idx] = tmpVal
		if stride > 1:
			res = res[:, ::stride, ::stride]
		return res

	def depth_wise_scale(self, pImg, convk, bn, scale, convStride=1):
		if convk.ndim != 4:
			print("Error, invalid convolution shape of dw/scale: {0}".format(convk.shape), file=sys.stderr)
			return None
		res = None
		tmpVal = convk.shape[-2] * convk.shape[-1]
		if tmpVal != 1:
			res = self.convolve_depth(pImg, convk, stride=convStride)
		else:
			res = self.convolve_point(pImg, convk)
		res = self.batchnorm(res, bn)
		res = self.bn_scale(res, scale)
		return res

	def forward1(self, pImg):
		res = self.convolve_norm(self.conv1, pImg)
		res = self.batchnorm(res, self.conv1_bn)
		res = self.bn_scale(res, self.conv1_scale)

		res = self.depth_wise_scale(res,
			self.conv2_1_dw, self.conv2_1_dw_bn, self.conv2_1_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv2_1_sep, self.conv2_1_sep_bn, self.conv2_1_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv2_2_dw, self.conv2_2_dw_bn, self.conv2_2_dw_scale, convStride=2)
		res = self.depth_wise_scale(res,
			self.conv2_2_sep, self.conv2_2_sep_bn, self.conv2_2_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv3_1_dw, self.conv3_1_dw_bn, self.conv3_1_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv3_1_sep, self.conv3_1_sep_bn, self.conv3_1_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv3_2_dw, self.conv3_2_dw_bn, self.conv3_2_dw_scale, convStride=2)
		res = self.depth_wise_scale(res,
			self.conv3_2_sep, self.conv3_2_sep_bn, self.conv3_2_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv4_1_dw, self.conv4_1_dw_bn, self.conv4_1_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv4_1_sep, self.conv4_1_sep_bn, self.conv4_1_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv4_2_dw, self.conv4_2_dw_bn, self.conv4_2_dw_scale, convStride=2)
		res = self.depth_wise_scale(res,
			self.conv4_2_sep, self.conv4_2_sep_bn, self.conv4_2_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv5_1_dw, self.conv5_1_dw_bn, self.conv5_1_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv5_1_sep, self.conv5_1_sep_bn, self.conv5_1_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv5_2_dw, self.conv5_2_dw_bn, self.conv5_2_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv5_2_sep, self.conv5_2_sep_bn, self.conv5_2_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv5_3_dw, self.conv5_3_dw_bn, self.conv5_3_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv5_3_sep, self.conv5_3_sep_bn, self.conv5_3_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv5_4_dw, self.conv5_4_dw_bn, self.conv5_4_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv5_4_sep, self.conv5_4_sep_bn, self.conv5_4_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv5_5_dw, self.conv5_5_dw_bn, self.conv5_5_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv5_5_sep, self.conv5_5_sep_bn, self.conv5_5_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv5_6_dw, self.conv5_6_dw_bn, self.conv5_6_dw_scale, convStride=2)
		res = self.depth_wise_scale(res,
			self.conv5_6_sep, self.conv5_6_sep_bn, self.conv5_6_sep_scale)

		res = self.depth_wise_scale(res,
			self.conv6_dw, self.conv6_dw_bn, self.conv6_dw_scale)
		res = self.depth_wise_scale(res,
			self.conv6_sep, self.conv6_sep_bn, self.conv6_sep_scale)

		ret = np.empty((res.shape[0],), dtype=np.float32)
		for idx in range(res.shape[0]):
			ret[idx] = np.mean(res[idx])
		res = np.dot(self.inner, ret) + self.bias
		res = np.exp(res)
		res /= np.sum(res)
		return res

