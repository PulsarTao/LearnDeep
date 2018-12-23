#!/usr/bin/python3

# import caffe
import numpy as np
from mnistLeNet import mnistLeNet
from load_mnist import load_mnist

# cnet = caffe.Net('lenet.prototxt', './lenet_iter_10000.caffemodel', caffe.TEST)

leNet = mnistLeNet()
images, labels = load_mnist('./', 't10k')

err = 0
for idx in range(len(labels)):
	dimg = images[idx].reshape(28, 28)
	retv = leNet.predict(dimg)
	retv = leNet.softmax(retv); retv = np.argmax(retv)
	if retv != labels[idx]:
		print("retv: %d, labels[%d]: %d" % (retv, idx, labels[idx]))
		err += 1
	if idx != 0 and idx & 0xFF == 0:
		print("Process %d, error: %d" % (idx, err))

