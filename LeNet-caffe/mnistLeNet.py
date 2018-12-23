#!/usr/bin/python3

import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce

class mnistLeNet(object):
	def __init__(self, weightFile='LeNet-Caffe.npz'):
		ln_wei = np.load(weightFile) # LeNet Weights
		conv1          = ln_wei['conv1_weights'].reshape(20, 5, 5)
		self.bias1     = ln_wei['conv1_bias'] # Length 20
		conv2          = ln_wei['conv2_weights'].reshape(50, 20, 5, 5)
		self.bias2     = ln_wei['conv2_bias'] # Length 50
		self.inner1    = ln_wei['inner1_weights'].reshape(500, 800)
		self.bias1_in  = ln_wei['inner1_bias'] # 500
		self.inner2    = ln_wei['inner2_weights'].reshape(10, 500)
		self.bias2_in  = ln_wei['inner2_bias'] # 10

		for idx in range(20):
			conv1[idx] = conv1[idx, ::-1, ::-1]
		self.conv1 = conv1
		for idx in range(50):
			for jdx in range(20):
				conv2[idx, jdx] = conv2[idx, jdx, ::-1, ::-1]
		self.conv2 = conv2

	def conv2d(self, c1, b1, pImg):
		siz1 = (pImg.shape[0] - c1.shape[1] + 1) if pImg.shape[0] > c1.shape[1] else (c1.shape[1] - pImg.shape[0] + 1)
		siz2 = (pImg.shape[1] - c1.shape[2] + 1) if pImg.shape[1] > c1.shape[2] else (c1.shape[2] - pImg.shape[1] + 1)
		result = np.zeros((c1.shape[0], siz1, siz2), np.float32)

		for chn in range(c1.shape[0]):
			convRes  = convolve2d(pImg, c1[chn], mode='valid')
			convRes += b1[chn]
			result[chn] = convRes
		return result

	def conv2d_(self, c1, b1, pImg):
		siz1 = (pImg.shape[1] - c1.shape[2] + 1) if pImg.shape[1] > c1.shape[2] else (c1.shape[2] - pImg.shape[1] + 1)
		siz2 = (pImg.shape[2] - c1.shape[3] + 1) if pImg.shape[2] > c1.shape[3] else (c1.shape[3] - pImg.shape[2] + 1)
		result = np.zeros((c1.shape[0], siz1, siz2), dtype=np.float32)

		for idx in range(c1.shape[0]):
			cr = np.zeros((siz1, siz2), dtype=np.float32)
			for jdx in range(c1.shape[1]):
				cr += convolve2d(pImg[jdx], c1[idx, jdx], mode='valid')
			cr += b1[idx]
			result[idx] = cr
		return result

	def maxpool2x2(self, pImg):
		siz1 = pImg.shape[pImg.ndim - 2] // 2
		siz2 = pImg.shape[pImg.ndim - 1] // 2
		imgShape = (siz1, siz2)
		if pImg.ndim > 2:
			imgShape = (*pImg.shape[0 : pImg.ndim - 2], siz1, siz2)
		img = np.zeros(imgShape, dtype=pImg.dtype)
		if pImg.ndim == 3:
			for idx in range(pImg.shape[0]):
				img[idx] = block_reduce(pImg[idx], (2, 2), np.max)
		elif pImg.ndim == 2:
			img = block_reduce(pImg, (2, 2), np.max)
		else:
			raise ValueError("Invalid dimension: %d" % pImg.ndim)
		return img

	def reLU_activate(self, pImg):
		return np.where(pImg > 0, pImg, 0)

	def softmax(self, pImg):
		nxp = np.exp(pImg)
		return nxp / np.sum(nxp)

	def predict(self, Img):
		res = self.conv2d(self.conv1, self.bias1, Img)
		res = self.maxpool2x2(res)

		res = self.conv2d_(self.conv2, self.bias2, res)
		res = self.maxpool2x2(res)

		res = np.dot(self.inner1, res.reshape(res.size))
		res += self.bias1_in
		res = self.reLU_activate(res)

		res = np.dot(self.inner2, res)
		res += self.bias2_in
		return res
		# res = self.softmax(res)
		# return np.argmax(res)

