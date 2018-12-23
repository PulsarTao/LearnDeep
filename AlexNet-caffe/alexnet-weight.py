#!/usr/bin/env python3

import caffe
import numpy as np

ANet = caffe.Net('./deploy.prototxt',
	'./bvlc_alexnet.caffemodel', caffe.TEST)

conv1 = ANet.params['conv1'][0].data
bias1 = ANet.params['conv1'][1].data
conv2 = ANet.params['conv2'][0].data
bias2 = ANet.params['conv2'][1].data
conv3 = ANet.params['conv3'][0].data
bias3 = ANet.params['conv3'][1].data
conv4 = ANet.params['conv4'][0].data
bias4 = ANet.params['conv4'][1].data
conv5 = ANet.params['conv5'][0].data
bias5 = ANet.params['conv5'][1].data

fc6   = ANet.params['fc6'][0].data
bias6 = ANet.params['fc6'][1].data
fc7   = ANet.params['fc7'][0].data
bias7 = ANet.params['fc7'][1].data
fc8   = ANet.params['fc8'][0].data
bias8 = ANet.params['fc8'][1].data

np.savez('Weights-AlexNet-caffe.npz',
	conv1=conv1, bias1=bias1, conv2=conv2, bias2=bias2,
	conv3=conv3, bias3=bias3, conv4=conv4, bias4=bias4,
	conv5=conv5, bias5=bias5, fc6=fc6,     bias6=bias6,
	fc7=fc7,     bias7=bias7, fc8=fc8,     bias8=bias8)

