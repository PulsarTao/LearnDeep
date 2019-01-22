#!/usr/bin/env python3

import sys
import math
import numpy as np
from scipy.signal import convolve2d

class ConvLayer(object):
	def __init__(self, nInput, nOutput, KSize=5, activation='relu'):
		# save parameters
		self.nin = nInput; self.nout = nOutput; self.ksize = KSize
		# Initialize bias
		self.bias    = np.zeros((nOutput,), dtype=np.float32)
		# Initialize convolutional weights 
		bound = math.sqrt(6 / (nInput + nOutput))
		self.weights = np.random.uniform(low=-bound, high=bound, size=(nOutput, nInput, KSize, KSize))
		self.weights = self.weights.astype(np.float32)
		# Determine activation function
		self.act_fn  = lambda MatIn : np.where(MatIn > 0, MatIn, 0)
		self.der_fn  = lambda MatIn : np.where(MatIn > 0, 1, 0)
		if activation == 'tanh':
			self.act_fn = np.tanh
			self.der_fn = lambda MatIn : 1 - np.square(MatIn)
		elif activation == 'sigmoid':
			self.act_fn = lambda MatIn : 1 / (1 + np.exp(-MatIn))
			self.der_fn = lambda MatIn : np.multiply(MatIn, 1 - MatIn)
		elif activation != 'relu':
			self.act_fn, self.der_fn = None, None
			print("Error, invalid activation: {0}".format(activation), file=sys.stderr)
		# Set input and derivative vectors to None
		self.mInput = None
		self.derivative = None

	def forward(self, mInput, doDrop=False):
		if mInput.ndim != 4:
			print("Error, invalid convolutional input dimension: {0}".format(mInput.ndim), file=sys.stderr)
			return None
		if mInput.shape[1] != self.nin:
			print("Error, input size mismatch: {0} <=> {1}".format(mInput.shape[1], self.nin), file=sys.stderr)
			return None
		minibatch = mInput.shape[0]
		convw, convb = self.weights, self.bias
		siz0 = mInput.shape[2] - self.ksize + 1
		siz1 = mInput.shape[3] - self.ksize + 1
		fres = np.empty((minibatch, self.nout, siz0, siz1), dtype=np.float32)
		for mb in range(minibatch):
			for idx in range(self.nout):
				tmpMat = np.zeros((siz0, siz1), dtype=np.float32)
				for jdx in range(self.nin):
					tmpMat += convolve2d(mInput[mb, jdx], convw[idx, jdx], mode='valid')
				tmpMat += self.nin * convb[idx]
				fres[mb, idx]  = self.act_fn(tmpMat)
		self.mInput = mInput
		self.derivative = self.der_fn(fres)
		return fres

	def flipConvW(self, convw_):
		convw = np.copy(convw_)
		for idx in range(convw.shape[0]):
			for jdx in range(convw.shape[1]):
				convw[idx, jdx] = convw[idx, jdx, ::-1, ::-1]
		return convw

	def backward(self, deltas, lr=0.01):
		mInput = self.mInput
		deriv = self.derivative
		if deriv is None:
			print("Error, derivative is None!", file=sys.stderr)
			return None
		if deltas.ndim != 4:
			print("Error, invalid dimension for deltas: {0}".format(deltas.shape), file=sys.stderr)
			return None
		minibatch = deltas.shape[0]
		if minibatch != deriv.shape[0]:
			print("Error, mini-batch size mismatch: {0} <=> {1}".format(minibatch, deriv.shape[0]), file=sys.stderr)
			return None
		siz0 = deltas.shape[2] + self.ksize - 1
		siz1 = deltas.shape[3] + self.ksize - 1
		convw = self.flipConvW(self.weights)
		newd = np.zeros(mInput.shape, dtype=np.float32)
		d_bias = np.zeros((minibatch, self.nout), dtype=np.float32)
		d_convw = np.zeros((minibatch, *convw.shape), dtype=np.float32)
		for mb in range(minibatch):
			for idx in range(self.nout):
				gdy = deltas[mb, idx] * deriv[mb, idx]
				gdy = gdy[::-1, ::-1]
				for jdx in range(self.nin):
					newd[mb, jdx] += convolve2d(deltas[mb, idx], convw[idx, jdx], mode='full')
					tmpMat = convolve2d(mInput[mb, jdx], gdy, mode='valid')
					d_convw[mb, idx, jdx] = tmpMat[::-1, ::-1]
				d_bias[mb, idx] = np.sum(gdy)
		d_bias = np.sum(d_bias, axis=0)
		d_convw = np.sum(d_convw, axis=0)
		self.bias -= lr * d_bias
		self.weights -= lr * d_convw
		self.mInput = None
		self.derivative = None
		return newd

