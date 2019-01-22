#!/usr/bin/env python3

import sys
import math
import numpy as np

class FullConLayer(object):
	def __init__(self, nInput, nOutput, activation='relu', dropout=False, keepProb=0.5):
		# save object parameters
		self.nin = nInput; self.nout = nOutput
		self.dropout = dropout; self.keepProb = keepProb
		self.upScale = np.float32(1 / keepProb) if dropout else None
		# initialize bias vector
		self.bias = np.zeros((nOutput,), dtype=np.float32)
		# initialize weights vector
		bound = math.sqrt(6 / (nInput + nOutput))
		self.weights = np.random.uniform(low=-bound, high=bound, size=(nInput, nOutput))
		self.weights = self.weights.astype(np.float32)
		# determine activation function
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
			print("Error, invalid activation function: {0}".format(activation), file=sys.stderr)
		# Set input and derivative vectors to None
		self.mInput = None
		self.dropMask = None
		self.derivative = None

	def forward(self, mInput, doDrop=False):
		if mInput.ndim != 2:
			print("Error, invalid dimension of mInput for FC: {0}".format(mInput.ndim), file=sys.stderr)
			return None
		minibatch = mInput.shape[0]
		if mInput.shape[1] != self.nin:
			print("Error, vector length mismatch: {0} <=> {1}".format(mInput.shape[1], self.nin), file=sys.stderr)
			return None
		fcw, bias = self.weights, self.bias
		fres = np.empty((minibatch, self.nout), dtype=np.float32)
		for mb in range(minibatch):
			fres[mb] = np.dot(mInput[mb], fcw) + bias
		fres = self.act_fn(fres)
		self.mInput = mInput
		self.dropMask = None
		self.derivative = self.der_fn(fres)
		if doDrop and self.dropout:
			mask = np.random.binomial(1, self.keepProb, size=fres.shape)
			self.dropMask = mask
			fres = np.where(mask != 0, fres, 0)
			fres *= self.upScale
		return fres

	def backward(self, deltas, lr=0.01):
		mInput = self.mInput
		deriv = self.derivative
		if deriv is None:
			print("Error, not derivative vector found.", file=sys.stderr)
			return None
		minibatch = deriv.shape[0]
		if deltas.ndim != 2:
			print("Error, invalid dimension for deltas in FC: {0}".format(deltas.ndim), file=sys.stderr)
			return None
		if deltas.shape[0] != minibatch or deltas.shape[1] != self.nout:
			print("Error, invalid shape for deltas in FC: {0}".format(deltas.shape), file=sys.stderr)
			return None
		if self.dropMask is not None:
			deltas = np.where(self.dropMask != 0, deltas, 0)
		fcw = self.weights
		newd = np.empty(mInput.shape, dtype=np.float32)
		db = np.empty((minibatch, self.nout), dtype=np.float32)
		dw = np.empty((minibatch, self.nin, self.nout), dtype=np.float32)
		for mb in range(minibatch):
			tmpMat0 = fcw * deriv[mb]
			newd[mb] = np.dot(tmpMat0, deltas[mb])
			tmpMat0 = mInput[mb]; tmpMat0 = tmpMat0.reshape(tmpMat0.size, 1)
			tmpMat1 = deriv[mb] * deltas[mb]
			db[mb] = tmpMat1; tmpMat1 = tmpMat1.reshape(1, tmpMat1.size)
			dw[mb] = np.dot(tmpMat0, tmpMat1)
		db = np.sum(db, axis=0)
		dw = np.sum(dw, axis=0)
		self.bias -= lr * db
		self.weights -= lr * dw
		self.mInput = None
		self.dropMask = None
		self.derivative = None
		return newd

