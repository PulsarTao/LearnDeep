#!/usr/bin/env python3

import sys
import caffe
import numpy as np
from scipy.signal import convolve2d

def flip_reverse(convk):
	convSize = convk.shape[-2] * convk.shape[-1]
	if convSize == 1:
		# return convk.reshape(*convk.shape[:-2])
		return convk
	conv = convk.reshape(-1, convk.shape[-2], convk.shape[-1])
	res = np.empty(conv.shape, dtype=np.float32)
	for idx in range(conv.shape[0]):
		res[idx] = conv[idx, ::-1, ::-1]
	res = res.reshape(*convk.shape)
	return res

def get_bn(bvec):
	var = bvec[1].data
	var = np.sqrt(np.float32(0.00001) + var)
	return (bvec[0].data, var, bvec[2].data)

def get_scale(bvec):
	return (bvec[0].data, bvec[1].data)

class MobileNetV2(object):
	def __init__(self, deploy='./mobilenet_v2_deploy.prototxt', cMode='./mobilenet_v2.caffemodel'):
		mbv2                           = caffe.Net(deploy, cMode, caffe.TEST)
		mbv2                           = mbv2.params
		self.conv1                     = flip_reverse(mbv2['conv1'][0].data)
		self.conv1_bn                  = get_bn(mbv2['conv1/bn'])
		self.conv1_scale               = get_scale(mbv2['conv1/scale'])

		self.conv2_1_expand, self.conv2_1_expand_bn, self.conv2_1_expand_scale = self.get_conv_bn_scale(mbv2, 'conv2_1/expand')
		self.conv2_1_dwise,  self.conv2_1_dwise_bn,  self.conv2_1_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv2_1/dwise')
		self.conv2_1_linear, self.conv2_1_linear_bn, self.conv2_1_linear_scale = self.get_conv_bn_scale(mbv2, 'conv2_1/linear')

		self.conv2_2_expand, self.conv2_2_expand_bn, self.conv2_2_expand_scale = self.get_conv_bn_scale(mbv2, 'conv2_2/expand')
		self.conv2_2_dwise,  self.conv2_2_dwise_bn,  self.conv2_2_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv2_2/dwise')
		self.conv2_2_linear, self.conv2_2_linear_bn, self.conv2_2_linear_scale = self.get_conv_bn_scale(mbv2, 'conv2_2/linear')

		self.conv3_1_expand, self.conv3_1_expand_bn, self.conv3_1_expand_scale = self.get_conv_bn_scale(mbv2, 'conv3_1/expand')
		self.conv3_1_dwise,  self.conv3_1_dwise_bn,  self.conv3_1_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv3_1/dwise')
		self.conv3_1_linear, self.conv3_1_linear_bn, self.conv3_1_linear_scale = self.get_conv_bn_scale(mbv2, 'conv3_1/linear')

		self.conv3_2_expand, self.conv3_2_expand_bn, self.conv3_2_expand_scale = self.get_conv_bn_scale(mbv2, 'conv3_2/expand')
		self.conv3_2_dwise,  self.conv3_2_dwise_bn,  self.conv3_2_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv3_2/dwise')
		self.conv3_2_linear, self.conv3_2_linear_bn, self.conv3_2_linear_scale = self.get_conv_bn_scale(mbv2, 'conv3_2/linear')

		self.conv4_1_expand, self.conv4_1_expand_bn, self.conv4_1_expand_scale = self.get_conv_bn_scale(mbv2, 'conv4_1/expand')
		self.conv4_1_dwise,  self.conv4_1_dwise_bn,  self.conv4_1_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv4_1/dwise')
		self.conv4_1_linear, self.conv4_1_linear_bn, self.conv4_1_linear_scale = self.get_conv_bn_scale(mbv2, 'conv4_1/linear')

		self.conv4_2_expand, self.conv4_2_expand_bn, self.conv4_2_expand_scale = self.get_conv_bn_scale(mbv2, 'conv4_2/expand')
		self.conv4_2_dwise,  self.conv4_2_dwise_bn,  self.conv4_2_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv4_2/dwise')
		self.conv4_2_linear, self.conv4_2_linear_bn, self.conv4_2_linear_scale = self.get_conv_bn_scale(mbv2, 'conv4_2/linear')

		self.conv4_3_expand, self.conv4_3_expand_bn, self.conv4_3_expand_scale = self.get_conv_bn_scale(mbv2, 'conv4_3/expand')
		self.conv4_3_dwise,  self.conv4_3_dwise_bn,  self.conv4_3_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv4_3/dwise')
		self.conv4_3_linear, self.conv4_3_linear_bn, self.conv4_3_linear_scale = self.get_conv_bn_scale(mbv2, 'conv4_3/linear')

		self.conv4_4_expand, self.conv4_4_expand_bn, self.conv4_4_expand_scale = self.get_conv_bn_scale(mbv2, 'conv4_4/expand')
		self.conv4_4_dwise,  self.conv4_4_dwise_bn,  self.conv4_4_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv4_4/dwise')
		self.conv4_4_linear, self.conv4_4_linear_bn, self.conv4_4_linear_scale = self.get_conv_bn_scale(mbv2, 'conv4_4/linear')

		self.conv4_5_expand, self.conv4_5_expand_bn, self.conv4_5_expand_scale = self.get_conv_bn_scale(mbv2, 'conv4_5/expand')
		self.conv4_5_dwise,  self.conv4_5_dwise_bn,  self.conv4_5_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv4_5/dwise')
		self.conv4_5_linear, self.conv4_5_linear_bn, self.conv4_5_linear_scale = self.get_conv_bn_scale(mbv2, 'conv4_5/linear')

		self.conv4_6_expand, self.conv4_6_expand_bn, self.conv4_6_expand_scale = self.get_conv_bn_scale(mbv2, 'conv4_6/expand')
		self.conv4_6_dwise,  self.conv4_6_dwise_bn,  self.conv4_6_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv4_6/dwise')
		self.conv4_6_linear, self.conv4_6_linear_bn, self.conv4_6_linear_scale = self.get_conv_bn_scale(mbv2, 'conv4_6/linear')

		self.conv4_7_expand, self.conv4_7_expand_bn, self.conv4_7_expand_scale = self.get_conv_bn_scale(mbv2, 'conv4_7/expand')
		self.conv4_7_dwise,  self.conv4_7_dwise_bn,  self.conv4_7_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv4_7/dwise')
		self.conv4_7_linear, self.conv4_7_linear_bn, self.conv4_7_linear_scale = self.get_conv_bn_scale(mbv2, 'conv4_7/linear')

		self.conv5_1_expand, self.conv5_1_expand_bn, self.conv5_1_expand_scale = self.get_conv_bn_scale(mbv2, 'conv5_1/expand')
		self.conv5_1_dwise,  self.conv5_1_dwise_bn,  self.conv5_1_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv5_1/dwise')
		self.conv5_1_linear, self.conv5_1_linear_bn, self.conv5_1_linear_scale = self.get_conv_bn_scale(mbv2, 'conv5_1/linear')

		self.conv5_2_expand, self.conv5_2_expand_bn, self.conv5_2_expand_scale = self.get_conv_bn_scale(mbv2, 'conv5_2/expand')
		self.conv5_2_dwise,  self.conv5_2_dwise_bn,  self.conv5_2_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv5_2/dwise')
		self.conv5_2_linear, self.conv5_2_linear_bn, self.conv5_2_linear_scale = self.get_conv_bn_scale(mbv2, 'conv5_2/linear')

		self.conv5_3_expand, self.conv5_3_expand_bn, self.conv5_3_expand_scale = self.get_conv_bn_scale(mbv2, 'conv5_3/expand')
		self.conv5_3_dwise,  self.conv5_3_dwise_bn,  self.conv5_3_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv5_3/dwise')
		self.conv5_3_linear, self.conv5_3_linear_bn, self.conv5_3_linear_scale = self.get_conv_bn_scale(mbv2, 'conv5_3/linear')

		self.conv6_1_expand, self.conv6_1_expand_bn, self.conv6_1_expand_scale = self.get_conv_bn_scale(mbv2, 'conv6_1/expand')
		self.conv6_1_dwise,  self.conv6_1_dwise_bn,  self.conv6_1_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv6_1/dwise')
		self.conv6_1_linear, self.conv6_1_linear_bn, self.conv6_1_linear_scale = self.get_conv_bn_scale(mbv2, 'conv6_1/linear')

		self.conv6_2_expand, self.conv6_2_expand_bn, self.conv6_2_expand_scale = self.get_conv_bn_scale(mbv2, 'conv6_2/expand')
		self.conv6_2_dwise,  self.conv6_2_dwise_bn,  self.conv6_2_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv6_2/dwise')
		self.conv6_2_linear, self.conv6_2_linear_bn, self.conv6_2_linear_scale = self.get_conv_bn_scale(mbv2, 'conv6_2/linear')

		self.conv6_3_expand, self.conv6_3_expand_bn, self.conv6_3_expand_scale = self.get_conv_bn_scale(mbv2, 'conv6_3/expand')
		self.conv6_3_dwise,  self.conv6_3_dwise_bn,  self.conv6_3_dwise_scale  = self.get_conv_bn_scale(mbv2, 'conv6_3/dwise')
		self.conv6_3_linear, self.conv6_3_linear_bn, self.conv6_3_linear_scale = self.get_conv_bn_scale(mbv2, 'conv6_3/linear')

		self.conv6_4         = mbv2['conv6_4'][0].data
		self.conv6_4_bn      = get_bn(mbv2['conv6_4/bn'])
		self.conv6_4_scale   = get_scale(mbv2['conv6_4/scale'])
		self.inner           = mbv2['fc7'][0].data.reshape(1000, 1280)
		self.bias            = mbv2['fc7'][1].data
		del mbv2

	def get_conv_bn_scale(self, mbNet, prefix):
		convk = flip_reverse(mbNet[prefix][0].data)
		bn    = get_bn(mbNet[prefix + '/bn'])
		scale = get_scale(mbNet[prefix + '/scale'])
		return (convk, bn, scale)

	def convolve_same(self, convk, pImg, stride=1):
		assert pImg.ndim == 3
		assert convk.ndim == 4
		assert pImg.shape[0] == convk.shape[1]
		siz1, siz2 = pImg.shape[1], pImg.shape[2]
		res = np.empty((convk.shape[0], siz1, siz2), dtype=np.float32)
		for idx in range(convk.shape[0]):
			tmpVal = np.empty((convk.shape[1], siz1, siz2), dtype=np.float32)
			for jdx in range(convk.shape[1]):
				tmpVal[jdx] = convolve2d(pImg[jdx], convk[idx, jdx], mode='same')
			res[idx] = np.sum(tmpVal, axis=0)
		if stride > 1:
			res = res[:, ::stride, ::stride]
		return res

	def convolve_dwise(self, convk, pImg, stride=1):
		assert pImg.ndim == 3
		assert convk.ndim == 4
		assert convk.shape[1] == 1
		assert convk.shape[0] == pImg.shape[0]
		# siz1, siz2 = pImg.shape[1], pImg.shape[2]
		res = np.empty(pImg.shape, dtype=np.float32)
		for idx in range(convk.shape[0]):
			res[idx] = convolve2d(pImg[idx], convk[idx, 0], mode='same')
		if stride > 1:
			res = res[:, ::stride, ::stride]
		return res

	def convolve_point(self, convk, pImg):
		assert pImg.ndim == 3
		conv = convk.reshape(convk.shape[0], convk.shape[1])
		assert pImg.shape[0] == conv.shape[1]
		siz1, siz2 = pImg.shape[1], pImg.shape[2]
		res = np.empty((conv.shape[0], siz1, siz2), dtype=np.float32)
		tmpVal = np.empty((conv.shape[1], siz1, siz2), dtype=np.float32)
		for idx in range(conv.shape[0]):
			for jdx in range(conv.shape[1]):
				tmpVal[jdx] = pImg[jdx] * conv[idx, jdx]
			res[idx] = np.sum(tmpVal, axis=0)
		return res

	def bn_scale(self, scale, pImg, relu=True):
		s0, s1 = scale[0], scale[1]
		assert s0.shape[0] == s1.shape[0]
		assert pImg.shape[0] == s0.shape[0]
		res = np.empty(pImg.shape, dtype=np.float32)
		for idx in range(pImg.shape[0]):
			res[idx] = pImg[idx] * s0[idx] + s1[idx]
		if relu:
			res = np.where(res > 0, res, 0)
		return res

	def batchnorm(self, bnVec, pImg, scale=None, ReLU=True):
		assert pImg.ndim == 3
		Mean, Var = bnVec[0], bnVec[1]
		assert Mean.shape[0] == Var.shape[0]
		assert pImg.shape[0] == Mean.shape[0]
		res = np.empty(pImg.shape, dtype=np.float32)
		for idx in range(pImg.shape[0]):
			res[idx] = (pImg[idx] - Mean[idx]) / Var[idx]
		if scale is not None:
			res = self.bn_scale(scale, res, relu=ReLU)
		return res

	def conv_bn_scale(self, convk, bn, scale, pImg, stride=1, relu=True):
		res = None
		assert convk.ndim == 4
		convSiz = convk.shape[-2] * convk.shape[-1]
		if convSiz > 1:
			res = self.convolve_dwise(convk, pImg, stride=stride)
		else:
			res = self.convolve_point(convk, pImg)
		res = self.batchnorm(bnVec=bn, pImg=res, scale=scale, ReLU=relu)
		return res

	def forward1(self, pImg):
		res = self.convolve_same(self.conv1, pImg, stride=2)
		res = self.batchnorm(self.conv1_bn, res, scale=self.conv1_scale)

		res = self.conv_bn_scale(self.conv2_1_expand, self.conv2_1_expand_bn, self.conv2_1_expand_scale, res)
		res = self.conv_bn_scale(self.conv2_1_dwise,  self.conv2_1_dwise_bn,  self.conv2_1_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv2_1_linear, self.conv2_1_linear_bn, self.conv2_1_linear_scale, res, relu=False)

		res = self.conv_bn_scale(self.conv2_2_expand, self.conv2_2_expand_bn, self.conv2_2_expand_scale, res)
		res = self.conv_bn_scale(self.conv2_2_dwise,  self.conv2_2_dwise_bn,  self.conv2_2_dwise_scale,  res, stride=2)
		res = self.conv_bn_scale(self.conv2_2_linear, self.conv2_2_linear_bn, self.conv2_2_linear_scale, res, relu=False)
		tmpv = res

		res = self.conv_bn_scale(self.conv3_1_expand, self.conv3_1_expand_bn, self.conv3_1_expand_scale, res)
		res = self.conv_bn_scale(self.conv3_1_dwise,  self.conv3_1_dwise_bn,  self.conv3_1_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv3_1_linear, self.conv3_1_linear_bn, self.conv3_1_linear_scale, res, relu=False)
		res += tmpv

		res = self.conv_bn_scale(self.conv3_2_expand, self.conv3_2_expand_bn, self.conv3_2_expand_scale, res)
		res = self.conv_bn_scale(self.conv3_2_dwise,  self.conv3_2_dwise_bn,  self.conv3_2_dwise_scale,  res, stride=2)
		res = self.conv_bn_scale(self.conv3_2_linear, self.conv3_2_linear_bn, self.conv3_2_linear_scale, res, relu=False)
		tmpv = res

		res = self.conv_bn_scale(self.conv4_1_expand, self.conv4_1_expand_bn, self.conv4_1_expand_scale, res)
		res = self.conv_bn_scale(self.conv4_1_dwise,  self.conv4_1_dwise_bn,  self.conv4_1_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv4_1_linear, self.conv4_1_linear_bn, self.conv4_1_linear_scale, res, relu=False)
		res += tmpv
		tmpv = res

		res = self.conv_bn_scale(self.conv4_2_expand, self.conv4_2_expand_bn, self.conv4_2_expand_scale, res)
		res = self.conv_bn_scale(self.conv4_2_dwise,  self.conv4_2_dwise_bn,  self.conv4_2_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv4_2_linear, self.conv4_2_linear_bn, self.conv4_2_linear_scale, res, relu=False)
		res += tmpv

		res = self.conv_bn_scale(self.conv4_3_expand, self.conv4_3_expand_bn, self.conv4_3_expand_scale, res)
		res = self.conv_bn_scale(self.conv4_3_dwise,  self.conv4_3_dwise_bn,  self.conv4_3_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv4_3_linear, self.conv4_3_linear_bn, self.conv4_3_linear_scale, res, relu=False)
		tmpv = res

		res = self.conv_bn_scale(self.conv4_4_expand, self.conv4_4_expand_bn, self.conv4_4_expand_scale, res)
		res = self.conv_bn_scale(self.conv4_4_dwise,  self.conv4_4_dwise_bn,  self.conv4_4_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv4_4_linear, self.conv4_4_linear_bn, self.conv4_4_linear_scale, res, relu=False)
		res += tmpv
		tmpv = res

		res = self.conv_bn_scale(self.conv4_5_expand, self.conv4_5_expand_bn, self.conv4_5_expand_scale, res)
		res = self.conv_bn_scale(self.conv4_5_dwise,  self.conv4_5_dwise_bn,  self.conv4_5_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv4_5_linear, self.conv4_5_linear_bn, self.conv4_5_linear_scale, res, relu=False)
		res += tmpv
		tmpv = res

		res = self.conv_bn_scale(self.conv4_6_expand, self.conv4_6_expand_bn, self.conv4_6_expand_scale, res)
		res = self.conv_bn_scale(self.conv4_6_dwise,  self.conv4_6_dwise_bn,  self.conv4_6_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv4_6_linear, self.conv4_6_linear_bn, self.conv4_6_linear_scale, res, relu=False)
		res += tmpv

		res = self.conv_bn_scale(self.conv4_7_expand, self.conv4_7_expand_bn, self.conv4_7_expand_scale, res)
		res = self.conv_bn_scale(self.conv4_7_dwise,  self.conv4_7_dwise_bn,  self.conv4_7_dwise_scale,  res, stride=2)
		res = self.conv_bn_scale(self.conv4_7_linear, self.conv4_7_linear_bn, self.conv4_7_linear_scale, res, relu=False)
		tmpv = res

		res = self.conv_bn_scale(self.conv5_1_expand, self.conv5_1_expand_bn, self.conv5_1_expand_scale, res)
		res = self.conv_bn_scale(self.conv5_1_dwise,  self.conv5_1_dwise_bn,  self.conv5_1_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv5_1_linear, self.conv5_1_linear_bn, self.conv5_1_linear_scale, res, relu=False)
		res += tmpv
		tmpv = res

		res = self.conv_bn_scale(self.conv5_2_expand, self.conv5_2_expand_bn, self.conv5_2_expand_scale, res)
		res = self.conv_bn_scale(self.conv5_2_dwise,  self.conv5_2_dwise_bn,  self.conv5_2_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv5_2_linear, self.conv5_2_linear_bn, self.conv5_2_linear_scale, res, relu=False)
		res += tmpv

		res = self.conv_bn_scale(self.conv5_3_expand, self.conv5_3_expand_bn, self.conv5_3_expand_scale, res)
		res = self.conv_bn_scale(self.conv5_3_dwise,  self.conv5_3_dwise_bn,  self.conv5_3_dwise_scale,  res, stride=2)
		res = self.conv_bn_scale(self.conv5_3_linear, self.conv5_3_linear_bn, self.conv5_3_linear_scale, res, relu=False)
		tmpv = res

		res = self.conv_bn_scale(self.conv6_1_expand, self.conv6_1_expand_bn, self.conv6_1_expand_scale, res)
		res = self.conv_bn_scale(self.conv6_1_dwise,  self.conv6_1_dwise_bn,  self.conv6_1_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv6_1_linear, self.conv6_1_linear_bn, self.conv6_1_linear_scale, res, relu=False)
		res += tmpv
		tmpv = res

		res = self.conv_bn_scale(self.conv6_2_expand, self.conv6_2_expand_bn, self.conv6_2_expand_scale, res)
		res = self.conv_bn_scale(self.conv6_2_dwise,  self.conv6_2_dwise_bn,  self.conv6_2_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv6_2_linear, self.conv6_2_linear_bn, self.conv6_2_linear_scale, res, relu=False)
		res += tmpv

		res = self.conv_bn_scale(self.conv6_3_expand, self.conv6_3_expand_bn, self.conv6_3_expand_scale, res)
		res = self.conv_bn_scale(self.conv6_3_dwise,  self.conv6_3_dwise_bn,  self.conv6_3_dwise_scale,  res)
		res = self.conv_bn_scale(self.conv6_3_linear, self.conv6_3_linear_bn, self.conv6_3_linear_scale, res, relu=False)

		res = self.conv_bn_scale(self.conv6_4, self.conv6_4_bn, self.conv6_4_scale, res)
		res = res.reshape(res.shape[0], -1)
		res = np.mean(res, axis=1, dtype=np.float32)
		res = np.dot(self.inner, res) + self.bias
		res = np.exp(res)
		res /= np.sum(res)
		return res

