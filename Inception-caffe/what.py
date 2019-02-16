#!/usr/bin/env python3

import caffe
import numpy as np

def dump_params(what):
	for key, val in what.items():
		lenVec = len(val)
		print("{0:25} => BlobVec, {1}".format(key, lenVec))
		for idx in range(lenVec):
			blob = val[idx].data
			print("\t{0}".format(blob.shape))
	return None

def dump_blobs(what):
	for key, val in what.items():
		blob = val.data
		print("{0:50} => Blob, {1}".format(key, blob.shape))
	return None

inet = caffe.Net('./deploy.prototxt', './bvlc_googlenet.caffemodel', caffe.TEST)
dump_blobs(inet.blobs)
dump_params(inet.params)

