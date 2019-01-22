#!/usr/bin/env python3

import sys
import math
import numpy as np

class SoftMaxLayer(object):
	def __init__(self, nInput, nOutput):
		# save object paramters
		self.nin = nInput; self.nout = nOutput
		# Initialize bias vector
		self.bias = np.zeros((nOutput,), dtype=np.float32)
		# Initialize weight vector
		bound = math.sqrt(6 / (nInput + nOutput))
		self.weights = np.random.uniform(low=-bound, high=bound, size=(nInput, nOutput))
		self.weights = self.weights.astype(np.float32)
		# Set derivative vector to None
		self.mInput = None
		self.derivative = None

	def forward(self, mInput, doDrop=True):
		if mInput.ndim != 2:
			print("Error, invalid dimension of mInput for SoftMaxLayer: {0}".format(mInput.ndim), file=sys.stderr)
			return None
		minibatch = mInput.shape[0]
		if mInput.shape[1] != self.nin:
			print("Error, invalid shape of mInput for SoftMaxLayer: {0}".format(mInput.shape), file=sys.stderr)
			return None
		smw, bias = self.weights, self.bias
		fres = np.empty((minibatch, self.nout), dtype=np.float32)
		deriv = np.empty((minibatch, self.nout, self.nout), dtype=np.float32)
		for mb in range(minibatch):
			tmpMat = np.dot(mInput[mb], smw) + bias
			tmpMat = np.exp(tmpMat)
			tmpMat = tmpMat / np.sum(tmpMat)
			fres[mb] = tmpMat
			deriv[mb] = np.diag(tmpMat) - np.dot(tmpMat.reshape(tmpMat.size, 1), tmpMat.reshape(1, tmpMat.size))
		# Set input and derivative vectors to None
		self.mInput = mInput
		self.derivative = deriv
		return fres

	def backward(self, deltas, lr=0.01):
		mInput = self.mInput
		deriv = self.derivative
		if deriv is None:
			print("Error, no derivative vector found for SoftMaxLayer.", file=sys.stderr)
			return None
		minibatch = deriv.shape[0]
		if deltas.ndim != 2:
			print("Error, invalid dimension for SoftMaxLayer: {0}".format(deltas.ndim), file=sys.stderr)
			return None
		if minibatch != deltas.shape[0]:
			print("Error, mini-batch size mismatch: {0} <=> {1}".format(minibatch, deltas.shape[0]), file=sys.stderr)
			return None
		if deltas.shape[1] != self.nout:
			print("Error, shape size(s) mismatch for SoftMaxLayer: {0} <=> {1}".format( deltas.shape[1], self.nout), dtype=np.float32)
			return None
		smw = self.weights
		newd = np.empty(mInput.shape, dtype=np.float32)
		db = np.empty((minibatch, self.nout), dtype=np.float32)
		dw = np.empty((minibatch, self.nin, self.nout), dtype=np.float32)
		for mb in range(minibatch):
			tmpMat = np.dot(smw, deriv[mb])
			newd[mb] = np.dot(tmpMat, deltas[mb])
			tmpMat = np.dot(deriv[mb], deltas[mb])
			db[mb] = tmpMat; tmpMat = tmpMat.reshape(1, tmpMat.size)
			tmat = mInput[mb]; tmat = tmat.reshape(tmat.size, 1)
			dw[mb] = np.dot(tmat, tmpMat)
		db = np.sum(db, axis=0)
		dw = np.sum(dw, axis=0)
		self.bias -= lr * db
		self.weights -= lr * dw
		self.mInput = None
		self.derivative = None
		return newd

