import typing

import cv2
import numpy


class LinearFunction:
    def __init__(self, a: typing.Union[numpy.array, numpy.array, int, float],
                       b: typing.Union[numpy.array, numpy.array, int, float]):
        self.a = a
        self.b = b

    @classmethod
    def from_least_squares(cls, xs: typing.Union[numpy.array, numpy.array],
                                ys: typing.Union[numpy.array, numpy.array]):
        assert len(xs) == len(ys)

        n = len(xs)

        xs = xs.astype(numpy.float64)
        ys = ys.astype(numpy.float64)

        xy = sum(xs * ys)
        x = sum(xs)
        y = sum(ys)
        x2 = sum(xs ** 2)

        bottom = n * x2 - x**2

        return cls(
            (n * xy - x * y) / bottom,
            (x2 * y - xy * x) / bottom,
        )

    def __call__(self, xs: typing.Union[numpy.array, numpy.array, int, float]):
        return self.a * xs + self.b
