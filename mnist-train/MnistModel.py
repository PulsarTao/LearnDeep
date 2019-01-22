#!/usr/bin/env python3

import sys
import os
import struct
import numpy as np
from ConvLayer import ConvLayer
from FullConLayer import FullConLayer
from SoftMaxLayer import SoftMaxLayer
from MaxPoolLayer import MaxPoolLayer

def onehot(labels, n_classes):
	result = np.zeros((n_classes, labels.shape[0]), dtype=np.float32)
	for idx, val in enumerate(labels.astype(int)):
		result[val, idx] = 1
	return result.T

def load_mnist(path, kind='train', lenet5=False):
	"""Load MNIST data from `path`"""
	labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
	images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype=np.uint8)
	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
		if lenet5:
			images = (images / 256.0).astype(np.float32)
		else:
			images = ((images / 256.0 - 0.5) * 2).astype(np.float32)
			std_val = np.std(images)
			mean_val = np.mean(images, axis=0)
			images = (images - mean_val) / std_val
		images = images.reshape(images.shape[0], 1, 28, 28)
	return images, labels

def forwardLeNet(res, layers, dofit=True):
	for ldx in range(4):
		layer = layers[ldx]
		res = layer.forward(res, doDrop=dofit)
	tmpShape = res.shape
	res = res.reshape(res.shape[0], -1)
	for ldx in range(4, len(layers)):
		layer = layers[ldx]
		res = layer.forward(res, doDrop=dofit)
	return res, tmpShape

def backwardLeNet(res, layers, tmps, learnR):
	for ldx in range(len(layers) - 1, 3, -1):
		layer = layers[ldx]
		res = layer.backward(res, learnR)
	res = res.reshape(tmps)
	for ldx in range(3, -1, -1):
		layer = layers[ldx]
		res = layer.backward(res, learnR)
	return None

def trainLeNet(X_train, y_train, X_valid, y_valid, layers, epoch=10, minibatch=64, learnR=0.0001):
	rand = np.random.RandomState(123)
	y_train_enc = onehot(y_train, 10)
	for epo in range(epoch):
		indices = np.arange(X_train.shape[0])
		rand.shuffle(indices)
		tloss = np.float32(0) # define total loss variable
		for jdx in range(0, indices.shape[0] - minibatch + 1, minibatch):
			batch_idx = indices[jdx : jdx + minibatch]
			imgs = X_train[batch_idx]
			# forward
			res, ts = forwardLeNet(imgs, layers)
			# backpropagation
			res -= y_train_enc[batch_idx]
			loss = np.sum(np.square(res)); tloss += loss
			backwardLeNet(res, layers, ts, learnR)
			print("\r MiniBatch: {0}, loss: {1:.4f}    ".format(jdx, loss), end='', file=sys.stdout)
			sys.stdout.flush()
		res, _ = forwardLeNet(X_valid, layers, dofit=False)
		res    = np.argmax(res, axis=1)
		res    = np.sum(res == y_valid)
		res   /= y_valid.shape[0]
		res   *= 100
		print("\n Epoch: {0}, validate accuracy: {1:.2f}, total loss: {2:.4f}".format(epo, res, tloss))

isLeNet5 = False
if 'lenet5' in sys.argv[1:]:
	isLeNet5 = True
Num = 55000
ln_layers = []
print("About to train model, lenet-5: {0}".format(isLeNet5))
if isLeNet5:
	lenet_act = 'tanh'
	# lenet_act = 'sigmoid'
	ln_layers.append(ConvLayer(1, 6, KSize=5, activation=lenet_act))
	ln_layers.append(MaxPoolLayer())
	ln_layers.append(ConvLayer(6, 16, KSize=5, activation=lenet_act))
	ln_layers.append(MaxPoolLayer())
	ln_layers.append(FullConLayer(256, 120, activation=lenet_act))
	ln_layers.append(FullConLayer(120, 84, activation=lenet_act, dropout=True))
	ln_layers.append(SoftMaxLayer(84, 10))
else:
	Num = 49920
	lenet_act = 'relu'
	ln_layers.append(ConvLayer(1, 32, KSize=5, activation=lenet_act))
	ln_layers.append(MaxPoolLayer())
	ln_layers.append(ConvLayer(32, 64, KSize=5, activation=lenet_act))
	ln_layers.append(MaxPoolLayer())
	ln_layers.append(FullConLayer(1024, 1024, activation=lenet_act, dropout=True))
	ln_layers.append(SoftMaxLayer(1024, 10))

X_train, y_train = load_mnist('./', kind='train', lenet5=isLeNet5)
if isLeNet5:
	trainLeNet(X_train[:Num], y_train[:Num], X_train[Num:], y_train[Num:], ln_layers, minibatch=100, learnR=0.01)
else:
	trainLeNet(X_train[:Num], y_train[:Num], X_train[Num:], y_train[Num:], ln_layers)

