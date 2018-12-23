#!/usr/bin/env python3

import numpy as np
from scipy.signal import convolve2d

def flipReverse(convk):
	if convk.ndim != 4:
		raise ValueError("Invalid convolution kernel: {0}".format(convk.ndim))
		return None
	for idx in range(convk.shape[0]):
		for jdx in range(convk.shape[1]):
			convk[idx, jdx] = convk[idx, jdx, ::-1, ::-1]
	return convk

# AlexNet 
class AlexNetCM(object):
	def __init__(self, weightFile='Weights-AlexNet-caffe.npz'):
		weights = np.load(weightFile)
		self.conv1   = weights['conv1']
		self.bias1   = weights['bias1']
		self.conv1   = flipReverse(self.conv1)

		self.conv2   = weights['conv2']
		self.bias2   = weights['bias2']
		self.conv2   = flipReverse(self.conv2)

		self.conv3   = weights['conv3']
		self.bias3   = weights['bias3']
		self.conv3   = flipReverse(self.conv3)

		self.conv4   = weights['conv4']
		self.bias4   = weights['bias4']
		self.conv4   = flipReverse(self.conv4)

		self.conv5   = weights['conv5']
		self.bias5   = weights['bias5']
		self.conv5   = flipReverse(self.conv5)

		self.fc6     = weights['fc6']
		self.bias6   = weights['bias6']
		self.fc7     = weights['fc7']
		self.bias7   = weights['bias7']
		self.fc8     = weights['fc8']
		self.bias8   = weights['bias8']

	def conv2d_stride(self, conv, bias, pImg, stride=1):
		if pImg.ndim != 3:
			raise ValueError("Error, invalid image dimension: {0}".format(pImg.ndim))
			return None
		if conv.ndim != 4:
			raise ValueError("Error, invalid dimension of kernel: {0}".format(conv.ndim))
			return None
		if conv.shape[1] != pImg.shape[0]:
			raise ValueError("Error, conv/pImg shape mismatch: {0}, {1}".format(conv.shape[1], pImg.shape[0]))
			return None
		if bias.shape[0] != conv.shape[0]:
			raise ValueError("Error, bias/conv shape mismatch: {0}, {1}".format(bias.shape[0], conv.shape[0]))
			return None

		siz1 = pImg.shape[1] - conv.shape[2] + 1
		siz2 = pImg.shape[2] - conv.shape[3] + 1
		sizr = (conv.shape[0], siz1, siz2)
		if stride > 1:
			sizr = (conv.shape[0], (siz1 - 1) // stride + 1, (siz2 - 1) // stride + 1)
		result = np.empty(sizr, dtype=np.float32)

		tmpVal = np.empty((conv.shape[1], siz1, siz2), dtype=np.float32)
		for idx in range(conv.shape[0]):
			for jdx in range(conv.shape[1]):
				tmpVal[jdx] = convolve2d(conv[idx, jdx], pImg[jdx], mode='valid')
			tmp_val = np.sum(tmpVal, axis=0)
			tmp_val += bias[idx]
			if stride > 1:
				result[idx] = tmp_val[::stride, ::stride]
			else:
				result[idx] = tmp_val
		return result

	def conv2d_pad_group(self, conv, bias, pImg, padLen=0, ngroup=1):
		if pImg.ndim != 3:
			raise ValueError("Error, invalid image dimention: ${0}".format(pImg.ndim))
			return None
		if conv.ndim != 4:
			raise ValueError("Error, invalid dimension of kernel: {0}".format(conv.ndim))
			return None
		if conv.shape[0] % ngroup != 0:
			raise ValueError("Error, invalid ngroup: {0}, conv.shape[0]: {1}".format(ngroup, conv.shape[0]))
			return None
		if pImg.shape[0] % ngroup != 0:
			raise ValueError("Error, invalid ngroup: {0}, pImg.shape[0]: {1}".format(ngroup, pImg.shape[0]))
			return None
		if pImg.shape[0] // ngroup != conv.shape[1]:
			raise ValueError("Error, invalid ngroup: {0}, shape(0, 1): ({1}, {2})".format(
				ngroup, pImg.shape[0], conv.shape[1]))
			return None
		if bias.shape[0] != conv.shape[0]:
			raise ValueError("Error, bias/conv shape mismatch: {0}, {1}".format(bias.shape[0], conv.shape[0]))
			return None

		convMode = 'same'
		siz1 = pImg.shape[1] + 2 * padLen - conv.shape[2] + 1
		siz2 = pImg.shape[2] + 2 * padLen - conv.shape[3] + 1
		if padLen == 0:
			convMode = 'valid'
		elif siz1 != pImg.shape[1] or siz2 != pImg.shape[2]:
			# NOTE: actually, the paddings in Caffe-AlexNet pre-trained model,
			# just make sure that the convolution operations works in 'SAME' mode
			raise ValueError("Error, invalid padding value: {0}".format(padLen))
			return None
		result = np.empty((conv.shape[0], siz1, siz2), dtype=np.float32)
		if ngroup == 1:
			tmpVal = np.empty((pImg.shape[0], siz1, siz2), dtype=np.float32)
			for idx in range(conv.shape[0]):
				for jdx in range(pImg.shape[0]):
					tmpVal[jdx] = convolve2d(pImg[jdx], conv[idx, jdx], mode=convMode)
				tmp_val = np.sum(tmpVal, axis=0)
				tmp_val += bias[idx]
				result[idx] = tmp_val
			return result
		step0 = conv.shape[0] // ngroup
		step1 = pImg.shape[0] // ngroup
		tmpVal = np.empty((step1, siz1, siz2), dtype=np.float32)
		for idx in range(ngroup):
			for jdx in range(step0):
				for kdx in range(step1):
					tmpVal[kdx] = convolve2d(pImg[step1 * idx + kdx], conv[step0 * idx + jdx, kdx], mode=convMode)
				tmp_val = np.sum(tmpVal, axis=0)
				ldx = step0 * idx + jdx
				tmp_val += bias[ldx]
				result[ldx] = tmp_val
		return result

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

	def max_pool3(self, pImg, ksize=3, stride=2):
		if pImg.ndim != 3:
			raise ValueError("Error, invalid max pool image dimension: {0}".format(pImg.ndim))
			return None
		if (pImg.shape[1] - ksize) % stride != 0:
			raise ValueError("Error, invalid max pool image shape[1]: {0}".format(pImg.shape[1]))
			return None
		if (pImg.shape[2] - ksize) % stride != 0:
			raise ValueError("Error, invalid max pool image shape[2]: {0}".format(pImg.shape[1]))
			return None

		siz1 = (pImg.shape[1] - ksize) // stride + 1
		siz2 = (pImg.shape[2] - ksize) // stride + 1
		result = np.empty((pImg.shape[0], siz1, siz2), dtype=np.float32)
		for idx in range(pImg.shape[0]):
			for jdx in range(siz1):
				for kdx in range(siz2):
					s0 = jdx * stride
					s1 = kdx * stride
					result[idx, jdx, kdx] = np.max(pImg[idx, s0 : s0 + ksize, s1 : s1 + ksize])
		# sorry, I can't come up with any faster solution now.
		return result

	def relu_activation(self, pImg):
		return np.where(pImg > 0, pImg, 0)

	def predict(self, pImg):
		res = self.conv2d_stride(self.conv1, self.bias1, pImg, stride=4)
		res = self.relu_activation(res)
		res = self.lrn3(0.0001, 0.75, 5, res)
		res = self.max_pool3(res)

		res = self.conv2d_pad_group(self.conv2, self.bias2, res, padLen=2, ngroup=2)
		res = self.relu_activation(res)
		res = self.lrn3(0.0001, 0.75, 5, res)
		res = self.max_pool3(res)

		res = self.conv2d_pad_group(self.conv3, self.bias3, res, padLen=1)
		res = self.relu_activation(res)
		res = self.conv2d_pad_group(self.conv4, self.bias4, res, padLen=1, ngroup=2)
		res = self.relu_activation(res)

		res = self.conv2d_pad_group(self.conv5, self.bias5, res, padLen=1, ngroup=2)
		res = self.relu_activation(res)
		res = self.max_pool3(res)

		res = np.dot(self.fc6, res.reshape(res.size))
		res += self.bias6
		res = self.relu_activation(res)

		res = np.dot(self.fc7, res)
		res += self.bias7
		res = self.relu_activation(res)

		res = np.dot(self.fc8, res)
		res += self.bias8

		res = np.exp(res)
		return res / np.sum(res)

