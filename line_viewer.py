import pickle

import numpy
import cupy
import cv2

import finder


with open('f.pickle', 'rb') as fp:
    f = pickle.load(fp)

xs = numpy.array([numpy.ones(f.a.shape) * i for i in range(256)], numpy.uint8)
ys = numpy.array([f(cupy.array(x)).get() for x in xs], numpy.uint8)

graph = cupy.zeros([256, 256, 3])

for x in range(256):
    print('\r{0:>3} / 255 [{1:>7.2%}]'.format(x, x/255), end='')
    colors = f(cupy.ones(f.a.shape) * x).transpose([2, 0, 1]).reshape([3, numpy.prod(f.a.shape[:2])]).astype(numpy.uint8)

    for y in range(256):
        graph[255 - y, x] = (colors == y).sum(axis=1)

    cpy = graph.get()
    cpy[:,min(255, x+1),:] = 255
    cpy[cpy > 255] = 255
    cv2.imshow('progress', cpy.astype(numpy.uint8))
    cv2.waitKey(1)
print()

graph = graph.get()
graph[graph > 255] = 255
cv2.imwrite('out.jpg', graph.astype(numpy.uint8))
