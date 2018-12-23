#!/usr/bin/env python3

import caffe
import numpy as np

cnet = caffe.Net('./lenet.prototxt',
    './lenet_iter_10000.caffemodel', caffe.TEST);

print(cnet.params)

conv1_weights       = cnet.params['conv1'][0].data
conv1_bias          = cnet.params['conv1'][1].data
conv2_weights       = cnet.params['conv2'][0].data
conv2_bias          = cnet.params['conv2'][1].data

inner1_weights      = cnet.params['ip1'][0].data
inner1_bias         = cnet.params['ip1'][1].data
inner2_weights      = cnet.params['ip2'][0].data
inner2_bias         = cnet.params['ip2'][1].data

np.savez_compressed('LeNet-Caffe.npz', conv1_weights=conv1_weights,
	conv1_bias=conv1_bias, conv2_weights=conv2_weights, conv2_bias=conv2_bias,
	inner1_weights=inner1_weights, inner1_bias=inner1_bias,
	inner2_weights=inner2_weights, inner2_bias=inner2_bias);

