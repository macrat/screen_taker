import pickle

import cv2
import numpy

import finder


with open('f.pickle', 'rb') as fp:
    f = pickle.load(fp)


coefficient = f.a - f.a.min()
coefficient /= coefficient.max()
coefficient *= 255
cv2.imwrite('coefficient.jpg', coefficient.astype(numpy.uint8))


bias = f.b - f.b.min()
bias /= bias.max()
bias *= 255
cv2.imwrite('bias.jpg', bias.astype(numpy.uint8))


xs = numpy.array([numpy.ones(f.a.shape) * i for i in range(256)], numpy.uint8)
ys = numpy.array([f(x) for x in xs], numpy.uint8)

graph = numpy.zeros([256, 256, 3])

for x in range(256):
    print('\r{0:>3} / 255 [{1:>7.2%}]'.format(x, x/255), end='')
    colors = f(numpy.ones(f.a.shape) * x).transpose([2, 0, 1]).reshape([3, numpy.prod(f.a.shape[:2])]).astype(numpy.uint8)

    for y in range(256):
        graph[255 - y, x] = (colors == y).sum(axis=1)

    cpy = graph.copy()
    cpy[:,min(255, x+1),:] = 255
    cpy[cpy > 255] = 255
    cv2.imshow('progress', cpy.astype(numpy.uint8))
    cv2.waitKey(1)
print()

graph[graph > 255] = 255
cv2.imwrite('color_lines.jpg', graph.astype(numpy.uint8))
