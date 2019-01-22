#!/usr/bin/env python3

import numpy as np
from skimage.measure import block_reduce

class MaxPoolLayer(object):
	def __init__(self, KSize=2):
		self.ksize = KSize

	def forward(self, pImg, doDrop=False):
		img = pImg
		ksize = self.ksize
		siz0 = img.shape[img.ndim - 2]
		siz1 = img.shape[img.ndim - 1]
		img = img.reshape(-1, siz0, siz1)
		siz0, siz1 = siz0 // ksize, siz1 // ksize
		fres = np.empty((img.shape[0], siz0, siz1), dtype=np.float32)
		for idx in range(img.shape[0]):
			fres[idx] = block_reduce(img[idx], (ksize, ksize), func=np.max)
		if pImg.ndim > 2:
			ndim = pImg.ndim - 2
			fres = fres.reshape(*pImg.shape[:ndim], siz0, siz1)
		else:
			fres = fres.reshape(siz0, siz1)
		return fres

	def backward(self, pImg, lr=0.01):
		img = pImg
		ksize = self.ksize
		siz0 = img.shape[img.ndim - 2]
		siz1 = img.shape[img.ndim - 1]
		img = img.reshape(-1, siz0, siz1)
		siz0, siz1 = siz0 * ksize, siz1 * ksize
		fres = np.empty((img.shape[0], siz0, siz1), dtype=np.float32)
		for idx in range(img.shape[0]):
			for jdx in range(ksize):
				for kdx in range(ksize):
					fres[idx, jdx::ksize, kdx::ksize] = img[idx]
		if pImg.ndim > 2:
			ndim = pImg.ndim - 2
			fres = fres.reshape(*pImg.shape[:ndim], siz0, siz1)
		else:
			fres = fres.reshape(siz0, siz1)
		return fres

