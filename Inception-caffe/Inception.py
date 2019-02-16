#!/usr/bin/env python3

import sys
import caffe
import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce

def kernel_reverse(convk):
	res = convk.reshape(-1, convk.shape[-2], convk.shape[-1])
	ret = np.empty(res.shape, dtype=np.float32)
	for idx in range(res.shape[0]):
		ret[idx] = res[idx, ::-1, ::-1]
	ret = ret.reshape(convk.shape)
	return ret

class InceptionV1(object):
	def __init__(self, deploy='./deploy.prototxt', caffem='./bvlc_googlenet.caffemodel'):
		inception = caffe.Net(deploy, caffem, caffe.TEST)
		# Extract parameters
		tmpVal           = inception.params['conv1/7x7_s2']
		self.conv1       = kernel_reverse(tmpVal[0].data)
		self.bias1       = tmpVal[1].data

		tmpVal           = inception.params['conv2/3x3_reduce']
		self.conv2_3r    = tmpVal[0].data
		self.bias2_3r    = tmpVal[1].data

		tmpVal           = inception.params['conv2/3x3']
		self.conv2_3     = kernel_reverse(tmpVal[0].data)
		self.bias2_3     = tmpVal[1].data

		tmpVal           = inception.params['inception_3a/1x1']
		self.conv_3a_1   = tmpVal[0].data
		self.bias_3a_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_3a/3x3_reduce']
		self.conv_3a_3r  = tmpVal[0].data
		self.bias_3a_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_3a/3x3']
		self.conv_3a_3   = kernel_reverse(tmpVal[0].data)
		self.bias_3a_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_3a/5x5_reduce']
		self.conv_3a_5r  = tmpVal[0].data
		self.bias_3a_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_3a/5x5']
		self.conv_3a_5   = kernel_reverse(tmpVal[0].data)
		self.bias_3a_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_3a/pool_proj']
		self.conv_3a_pp  = tmpVal[0].data
		self.bias_3a_pp  = tmpVal[1].data

		tmpVal           = inception.params['inception_3b/1x1']
		self.conv_3b_1   = tmpVal[0].data
		self.bias_3b_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_3b/3x3_reduce']
		self.conv_3b_3r  = tmpVal[0].data
		self.bias_3b_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_3b/3x3']
		self.conv_3b_3   = kernel_reverse(tmpVal[0].data)
		self.bias_3b_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_3b/5x5_reduce']
		self.conv_3b_5r  = tmpVal[0].data
		self.bias_3b_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_3b/5x5']
		self.conv_3b_5   = kernel_reverse(tmpVal[0].data)
		self.bias_3b_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_3b/pool_proj']
		self.conv_3b_pp  = tmpVal[0].data
		self.bias_3b_pp  = tmpVal[1].data

		tmpVal           = inception.params['inception_4a/1x1']
		self.conv_4a_1   = tmpVal[0].data
		self.bias_4a_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_4a/3x3_reduce']
		self.conv_4a_3r  = tmpVal[0].data
		self.bias_4a_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4a/3x3']
		self.conv_4a_3   = kernel_reverse(tmpVal[0].data)
		self.bias_4a_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_4a/5x5_reduce']
		self.conv_4a_5r  = tmpVal[0].data
		self.bias_4a_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4a/5x5']
		self.conv_4a_5   = kernel_reverse(tmpVal[0].data)
		self.bias_4a_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_4a/pool_proj']
		self.conv_4a_pp  = tmpVal[0].data
		self.bias_4a_pp  = tmpVal[1].data

		tmpVal           = inception.params['inception_4b/1x1']
		self.conv_4b_1   = tmpVal[0].data
		self.bias_4b_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_4b/3x3_reduce']
		self.conv_4b_3r  = tmpVal[0].data
		self.bias_4b_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4b/3x3']
		self.conv_4b_3   = kernel_reverse(tmpVal[0].data)
		self.bias_4b_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_4b/5x5_reduce']
		self.conv_4b_5r  = tmpVal[0].data
		self.bias_4b_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4b/5x5']
		self.conv_4b_5   = kernel_reverse(tmpVal[0].data)
		self.bias_4b_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_4b/pool_proj']
		self.conv_4b_pp  = tmpVal[0].data
		self.bias_4b_pp  = tmpVal[1].data

		tmpVal           = inception.params['inception_4c/1x1']
		self.conv_4c_1   = tmpVal[0].data
		self.bias_4c_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_4c/3x3_reduce']
		self.conv_4c_3r  = tmpVal[0].data
		self.bias_4c_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4c/3x3']
		self.conv_4c_3   = kernel_reverse(tmpVal[0].data)
		self.bias_4c_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_4c/5x5_reduce']
		self.conv_4c_5r  = tmpVal[0].data
		self.bias_4c_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4c/5x5']
		self.conv_4c_5   = kernel_reverse(tmpVal[0].data)
		self.bias_4c_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_4c/pool_proj']
		self.conv_4c_pp  = tmpVal[0].data
		self.bias_4c_pp  = tmpVal[1].data

		tmpVal           = inception.params['inception_4d/1x1']
		self.conv_4d_1   = tmpVal[0].data
		self.bias_4d_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_4d/3x3_reduce']
		self.conv_4d_3r  = tmpVal[0].data
		self.bias_4d_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4d/3x3']
		self.conv_4d_3   = kernel_reverse(tmpVal[0].data)
		self.bias_4d_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_4d/5x5_reduce']
		self.conv_4d_5r  = tmpVal[0].data
		self.bias_4d_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4d/5x5']
		self.conv_4d_5   = kernel_reverse(tmpVal[0].data)
		self.bias_4d_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_4d/pool_proj']
		self.conv_4d_pp  = tmpVal[0].data
		self.bias_4d_pp  = tmpVal[1].data

		tmpVal           = inception.params['inception_4e/1x1']
		self.conv_4e_1   = tmpVal[0].data
		self.bias_4e_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_4e/3x3_reduce']
		self.conv_4e_3r  = tmpVal[0].data
		self.bias_4e_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4e/3x3']
		self.conv_4e_3   = kernel_reverse(tmpVal[0].data)
		self.bias_4e_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_4e/5x5_reduce']
		self.conv_4e_5r  = tmpVal[0].data
		self.bias_4e_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_4e/5x5']
		self.conv_4e_5   = kernel_reverse(tmpVal[0].data)
		self.bias_4e_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_4e/pool_proj']
		self.conv_4e_pp  = tmpVal[0].data
		self.bias_4e_pp  = tmpVal[1].data

		tmpVal           = inception.params['inception_5a/1x1']
		self.conv_5a_1   = tmpVal[0].data
		self.bias_5a_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_5a/3x3_reduce']
		self.conv_5a_3r  = tmpVal[0].data
		self.bias_5a_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_5a/3x3']
		self.conv_5a_3   = kernel_reverse(tmpVal[0].data)
		self.bias_5a_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_5a/5x5_reduce']
		self.conv_5a_5r  = tmpVal[0].data
		self.bias_5a_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_5a/5x5']
		self.conv_5a_5   = kernel_reverse(tmpVal[0].data)
		self.bias_5a_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_5a/pool_proj']
		self.conv_5a_pp  = tmpVal[0].data
		self.bias_5a_pp  = tmpVal[1].data

		tmpVal           = inception.params['inception_5b/1x1']
		self.conv_5b_1   = tmpVal[0].data
		self.bias_5b_1   = tmpVal[1].data

		tmpVal           = inception.params['inception_5b/3x3_reduce']
		self.conv_5b_3r  = tmpVal[0].data
		self.bias_5b_3r  = tmpVal[1].data

		tmpVal           = inception.params['inception_5b/3x3']
		self.conv_5b_3   = kernel_reverse(tmpVal[0].data)
		self.bias_5b_3   = tmpVal[1].data

		tmpVal           = inception.params['inception_5b/5x5_reduce']
		self.conv_5b_5r  = tmpVal[0].data
		self.bias_5b_5r  = tmpVal[1].data

		tmpVal           = inception.params['inception_5b/5x5']
		self.conv_5b_5   = kernel_reverse(tmpVal[0].data)
		self.bias_5b_5   = tmpVal[1].data

		tmpVal           = inception.params['inception_5b/pool_proj']
		self.conv_5b_pp  = tmpVal[0].data
		self.bias_5b_pp  = tmpVal[1].data

		tmpVal           = inception.params['loss3/classifier']
		self.inner       = tmpVal[0].data
		self.bias        = tmpVal[1].data
		del inception

	def _convovle_pad(self, conv_k, bias_k, inImg, padLen=0, stride=1):
		if conv_k.ndim != 4 or inImg.ndim != 3:
			print("Error, convolution kernel or input image dimensions: {0}, {1}".format(conv_k.ndim, inImg.ndim), file=sys.stderr)
			return None
		if conv_k.shape[1] != inImg.shape[0]:
			print("Error, invalid shapes: {0}, {1}".format(conv_k.shape, inImg.shape), file=sys.stderr)
			return None
		if bias_k.ndim != 1 or bias_k.shape[0] != conv_k.shape[0]:
			print("Error, invalid bias shape: {0}".format(bias_k.shape), file=sys.stderr)
			return None
		if conv_k.shape[2] != conv_k.shape[3]:
			print("Error, invalid convolution kernel: {0}".format(conv_k.shape), file=sys.stderr)
			return None

		cmode = 'valid'
		siz1 = inImg.shape[1] - conv_k.shape[2] + 1
		siz2 = inImg.shape[2] - conv_k.shape[3] + 1
		if padLen == 0:
			pass
		elif padLen == (conv_k.shape[2] // 2):
			cmode = 'same'
			siz1 = inImg.shape[1]
			siz2 = inImg.shape[2]
		elif padLen == (conv_k.shape[2] - 1):
			cmode = 'full'
			siz1 = inImg.shape[1] + conv_k.shape[2] - 1
			siz2 = inImg.shape[2] + conv_k.shape[3] - 1
		else:
			print("Error, invalid padding length: {0} for convolution: {1}".format(padLen, conv_k.shape), file=sys.stderr)
			return None

		res = np.empty((conv_k.shape[0], siz1, siz2), dtype=np.float32)
		for idx in range(conv_k.shape[0]):
			tmpVal = np.empty((conv_k.shape[1], siz1, siz2), dtype=np.float32)
			for jdx in range(conv_k.shape[1]):
				tmpVal[jdx] = convolve2d(inImg[jdx], conv_k[idx, jdx], mode=cmode)
			tmpVal = np.sum(tmpVal, axis=0)
			tmpVal += bias_k[idx]
			res[idx] = tmpVal
		if stride > 1:
			res = res[:, ::stride, ::stride]
		return res

	def _convovle_reduce(self, conv_k, bias_k, pImg):
		if conv_k.ndim != 4 or conv_k.shape[2] != 1 or conv_k.shape[3] != 1:
			print("Error, invalid reduce shape: {0}".format(conv_k.shape), file=sys.stderr)
			return None
		if pImg.ndim != 3 or pImg.shape[0] != conv_k.shape[1]:
			print("Error, invalid shapes, image: {0}, kernel: {1}".format(pImg.shape, conv_k.shape), file=sys.stderr)
			return None
		if bias_k.shape[0] != conv_k.shape[0]:
			print("Error, invalid bias shape: {0} <=> {1}".format(bias_k.shape, conv_k.shape), file=sys.stderr)
			return None
		convk = conv_k.reshape(conv_k.shape[0], -1)
		siz1, siz2 = pImg.shape[1], pImg.shape[2]
		res = np.empty((convk.shape[0], siz1, siz2), dtype=np.float32)
		for idx in range(convk.shape[0]):
			tmpVal = np.empty((convk.shape[1], siz1, siz1), dtype=np.float32)
			for jdx in range(convk.shape[1]):
				tmpVal[jdx] = convk[idx, jdx] * pImg[jdx]
			tmpVal = np.sum(tmpVal, axis=0)
			tmpVal += bias_k[idx]
			res[idx] = tmpVal
		return res

	def relu_activate(self, pImg):
		return np.where(pImg > 0, pImg, 0)

	def maxpool(self, pImg, kSize=3, stride=2):
		res = pImg.reshape(-1, pImg.shape[-2], pImg.shape[-1])
		siz_1, siz_2 = res.shape[1], res.shape[2]
		siz1, siz2 = siz_1 // stride, siz_2 // stride
		ret = np.empty((res.shape[0], siz1, siz2), dtype=np.float32)
		if kSize == stride:
			for idx in range(res.shape[0]):
				ret[idx] = block_reduce(res[idx], (kSize, kSize), np.max)
			if pImg.ndim > 3:
				ret = ret.reshape(*pImg.shape[:2], siz1, siz2)
			elif pImg.ndim == 2:
				ret = ret.reshape(siz1, siz2)
			return ret
		offSet = kSize // 2
		for idx in range(res.shape[0]):
			for jdx in range(siz1):
				tmpv = jdx * stride + 1
				w0 = tmpv - offSet
				w1 = w0 + kSize
				w1 = siz_1 if w1 > siz_1 else w1
				w0 = 0 if w0 < 0 else w0
				tmpv = 1
				for kdx in range(siz2):
					h0 = tmpv - offSet
					h1 = h0 + kSize
					h1 = siz_2 if h1 > siz_2 else h1
					h0 = 0 if h0 < 0 else h0
					ret[idx, jdx, kdx] = np.max(res[idx, w0:w1, h0:h1])
					tmpv += stride
		if pImg.ndim > 3:
			ret = ret.reshape(*pImg.shape[:2], siz1, siz2)
		elif pImg.ndim == 2:
			ret = ret.reshape(siz1, siz2)
		return ret

	def maxpool_same(self, pImg, kSize=3):
		if pImg.ndim != 3:
			print("Error, invalid max-pool-same shape: {0}".format(pImg.shape), file=sys.stderr)
			return None
		offSet = kSize // 2
		siz1, siz2 = pImg.shape[1], pImg.shape[2]
		ret = np.empty(pImg.shape, dtype=np.float32)
		for idx in range(pImg.shape[0]):
			for jdx in range(siz1):
				w0 = jdx - offSet
				w1 = w0 + kSize
				w0 = 0 if w0 < 0 else w0
				w1 = siz1 if w1 > siz1 else w1
				for kdx in range(siz2):
					h0 = kdx - offSet
					h1 = h0 + kSize
					h0 = 0 if h0 < 0 else h0
					h1 = siz2 if h1 > siz2 else h1
					ret[idx, jdx, kdx] = np.max(pImg[idx, w0:w1, h0:h1])
		return ret

	def lrn3(self, alpha, beta, nls, pImg):
		if pImg.ndim != 3:
			raise ValueError("Invalid LRN dimension: %d" % pImg.ndim)
			return None
		nlsh = nls // 2; nls2 = nlsh + (nls & 1)
		img2 = np.square(pImg); maxN = pImg.shape[0]
		result = np.empty(pImg.shape, dtype=np.float32)
		for idx in range(pImg.shape[1]):
			for jdx in range(pImg.shape[2]):
				for kdx in range(maxN):
					start = kdx - nlsh if kdx > nlsh else 0
					end = maxN if maxN < (kdx + nls2) else kdx + nls2
					result[kdx, idx, jdx] = pImg[kdx, idx, jdx] / np.power(1 + alpha * np.sum(img2[start:end, idx, jdx]) / nls, beta)
		return result

	def incept(self, pImg, conv1, bias1, conv3r, bias3r, conv3, bias3, conv5r, bias5r, conv5, bias5, conv_pp, bias_pp):
		res        = self._convovle_reduce(conv1, bias1, pImg)
		res_conv1  = self.relu_activate(res)
		res        = self._convovle_reduce(conv3r, bias3r, pImg)
		res        = self.relu_activate(res)
		res        = self._convovle_pad(conv3, bias3, res, padLen=1)
		res_conv3  = self.relu_activate(res)
		res        = self._convovle_reduce(conv5r, bias5r, pImg)
		res        = self.relu_activate(res)
		res        = self._convovle_pad(conv5, bias5, res, padLen=2)
		res_conv5  = self.relu_activate(res)
		res        = self.maxpool_same(pImg)
		res        = self._convovle_reduce(conv_pp, bias_pp, res)
		res_convpp = self.relu_activate(res)
		return np.vstack((res_conv1, res_conv3, res_conv5, res_convpp))

	def forward1(self, pImg):
		res = self._convovle_pad(self.conv1, self.bias1, pImg, padLen=3, stride=2)
		res = self.relu_activate(res)
		res = self.maxpool(res)
		res = self.lrn3(0.0001, 0.75, 5, res)

		res = self._convovle_reduce(self.conv2_3r, self.bias2_3r, res)
		res = self.relu_activate(res)

		res = self._convovle_pad(self.conv2_3, self.bias2_3, res, padLen=1)
		res = self.relu_activate(res)
		res = self.lrn3(0.0001, 0.75, 5, res)
		res = self.maxpool(res)

		res = self.incept(res, conv1=self.conv_3a_1, bias1=self.bias_3a_1,
			conv3r=self.conv_3a_3r, bias3r=self.bias_3a_3r, conv3=self.conv_3a_3, bias3=self.bias_3a_3,
			conv5r=self.conv_3a_5r, bias5r=self.bias_3a_5r, conv5=self.conv_3a_5, bias5=self.bias_3a_5,
			conv_pp=self.conv_3a_pp, bias_pp=self.bias_3a_pp)

		res = self.incept(res, conv1=self.conv_3b_1, bias1=self.bias_3b_1,
			conv3r=self.conv_3b_3r, bias3r=self.bias_3b_3r, conv3=self.conv_3b_3, bias3=self.bias_3b_3,
			conv5r=self.conv_3b_5r, bias5r=self.bias_3b_5r, conv5=self.conv_3b_5, bias5=self.bias_3b_5,
			conv_pp=self.conv_3b_pp, bias_pp=self.bias_3b_pp)
		res = self.maxpool(res)

		res = self.incept(res, conv1=self.conv_4a_1, bias1=self.bias_4a_1,
			conv3r=self.conv_4a_3r, bias3r=self.bias_4a_3r, conv3=self.conv_4a_3, bias3=self.bias_4a_3,
			conv5r=self.conv_4a_5r, bias5r=self.bias_4a_5r, conv5=self.conv_4a_5, bias5=self.bias_4a_5,
			conv_pp=self.conv_4a_pp, bias_pp=self.bias_4a_pp)
		res = self.incept(res, conv1=self.conv_4b_1, bias1=self.bias_4b_1,
			conv3r=self.conv_4b_3r, bias3r=self.bias_4b_3r, conv3=self.conv_4b_3, bias3=self.bias_4b_3,
			conv5r=self.conv_4b_5r, bias5r=self.bias_4b_5r, conv5=self.conv_4b_5, bias5=self.bias_4b_5,
			conv_pp=self.conv_4b_pp, bias_pp=self.bias_4b_pp)
		res = self.incept(res, conv1=self.conv_4c_1, bias1=self.bias_4c_1,
			conv3r=self.conv_4c_3r, bias3r=self.bias_4c_3r, conv3=self.conv_4c_3, bias3=self.bias_4c_3,
			conv5r=self.conv_4c_5r, bias5r=self.bias_4c_5r, conv5=self.conv_4c_5, bias5=self.bias_4c_5,
			conv_pp=self.conv_4c_pp, bias_pp=self.bias_4c_pp)
		res = self.incept(res, conv1=self.conv_4d_1, bias1=self.bias_4d_1,
			conv3r=self.conv_4d_3r, bias3r=self.bias_4d_3r, conv3=self.conv_4d_3, bias3=self.bias_4d_3,
			conv5r=self.conv_4d_5r, bias5r=self.bias_4d_5r, conv5=self.conv_4d_5, bias5=self.bias_4d_5,
			conv_pp=self.conv_4d_pp, bias_pp=self.bias_4d_pp)
		res = self.incept(res, conv1=self.conv_4e_1, bias1=self.bias_4e_1,
			conv3r=self.conv_4e_3r, bias3r=self.bias_4e_3r, conv3=self.conv_4e_3, bias3=self.bias_4e_3,
			conv5r=self.conv_4e_5r, bias5r=self.bias_4e_5r, conv5=self.conv_4e_5, bias5=self.bias_4e_5,
			conv_pp=self.conv_4e_pp, bias_pp=self.bias_4e_pp)
		res = self.maxpool(res)

		res = self.incept(res, conv1=self.conv_5a_1, bias1=self.bias_5a_1,
			conv3r=self.conv_5a_3r, bias3r=self.bias_5a_3r, conv3=self.conv_5a_3, bias3=self.bias_5a_3,
			conv5r=self.conv_5a_5r, bias5r=self.bias_5a_5r, conv5=self.conv_5a_5, bias5=self.bias_5a_5,
			conv_pp=self.conv_5a_pp, bias_pp=self.bias_5a_pp)
		res = self.incept(res, conv1=self.conv_5b_1, bias1=self.bias_5b_1,
			conv3r=self.conv_5b_3r, bias3r=self.bias_5b_3r, conv3=self.conv_5b_3, bias3=self.bias_5b_3,
			conv5r=self.conv_5b_5r, bias5r=self.bias_5b_5r, conv5=self.conv_5b_5, bias5=self.bias_5b_5,
			conv_pp=self.conv_5b_pp, bias_pp=self.bias_5b_pp)
		if res.ndim != 3 or res.shape[1] != 7 or res.shape[2] != 7:
			print("Error, Invalid Inception output, shape: {0}".format(res.shape), file=sys.stderr)
			return None
		ret = np.empty((res.shape[0],), dtype=np.float32)
		for idx in range(res.shape[0]):
			ret[idx] = np.mean(res[idx])
		res = np.dot(self.inner, ret) + self.bias
		res = np.exp(res)
		res /= np.sum(res)
		return res

