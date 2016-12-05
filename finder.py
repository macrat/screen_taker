import asyncio
import typing

import cv2
import numpy


class Camera:
    def __init__(self, id_: int = 0):
        self.cam = cv2.VideoCapture(id_)

        if not self.cam.isOpened():
            raise IOError('failed open camera')

    def set_mode(self, width: int, height: int) -> None:
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get(self) -> numpy.array:
        ok, img = self.cam.read()

        if not ok:
            raise IOError('failed take photo')

        return img

    def get_size(self) -> typing.Tuple[int, int]:
        return (
            int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )


class CameraMonitor(Camera):
    def __init__(self, id_: int = 0, name: str = 'live monitor', fps: int = 30):
        super().__init__(id_)

        self.name = name
        self.fps = fps
        self._img = None

    #def __del__(self):
    #    cv2.destroyWindow(self.name)

    def _get(self) -> numpy.array:
        self._img = super().get()
        return self._img

    def get(self) -> numpy.array:
        return self._img

    async def mainloop(self) -> None:
        while True:
            cv2.imshow(self.name, self._get())
            show_image()
            await asyncio.sleep((1000 / self.fps - 1) / 1000)


class SimpleImagePhotoTaker:
    def __init__(self, cam: Camera = None, name: str = 'screen'):
        if cam is None:
            self.cam = Camera()
        else:
            self.cam = cam
        self.name = name

    def __del__(self):
        cv2.destroyWindow(self.name)

    def prepare(self) -> None:
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, img: numpy.array, delay: int = 1) -> None:
        self.prepare()
        cv2.imshow(self.name, img)

        show_image(delay)

    def take(self, img: numpy.array, delay: int = 100) -> numpy.array:
        self.show(img, delay)

        return self.cam.get()

    async def take_async(self, img: numpy.array, delay: int = 100) -> numpy.array:
        self.show(img)

        await asyncio.sleep((delay - 1) / 1000)

        return self.cam.get()


class MultipleTaker(SimpleImagePhotoTaker):
    def __init__(self, imgs: tuple, cam: Camera = None, name: str = 'screen'):
        super().__init__(cam, name)
        self.imgs = imgs
        self.results = numpy.zeros([len(imgs), *cam.get_size(), 3], numpy.uint32)

        self.count = 0

    def take(self, delay: int = 100) -> None:
        for i, img in enumerate(self.imgs):
            self.results[i] += super().take(img, delay)

        self.count += 1

    async def take_async(self, delay: int = 100) -> None:
        for i, img in enumerate(self.imgs):
            self.results[i] += await super().take_async(img, delay)

        self.count += 1

    def get_results(self) -> list:
        return self.results / self.count


async def wait_for_return() -> None:
    keys = (ord('\n'), ord('n'))
    while wait_for_return._key%256 not in keys and cv2.waitKey(1)%256 not in keys:
        await asyncio.sleep(0.001)
    wait_for_return._key = -1
wait_for_return._key = -1


def show_image(delay: int = 1) -> None:
    k = cv2.waitKey(delay)
    if k >= 0:
        wait_for_return._key = k


async def screen_take(cam: Camera, images: tuple, loop: int = 3) -> numpy.array:
    taker = MultipleTaker(images, cam=cam)

    taker.prepare()

    print('ready for initialize. please hit space key.')
    await wait_for_return()

    for i in range(loop):
        await taker.take_async(300)

    result = taker.get_results()
    for i, img in enumerate(result.astype(numpy.uint8)):
        cv2.imshow(str(i), img)

    return result


def make_circles_grid(width: int, height: int, grid_size: typing.Tuple[int, int] = (6, 4)) -> typing.Tuple[numpy.array, numpy.array]:
    img = numpy.zeros((height, width), numpy.uint8)

    for y in range(1, grid_size[1]+1):
        for x in range(1, grid_size[0]+1):
            cv2.circle(img, (width * x // (grid_size[0]+1), height * y // (grid_size[1]+1)), min(width, height) // 20, 255, -1)

    return img, 255 - img


async def find_screen(cam: Camera, width: int, height: int) -> numpy.array:
    grid_size = (6, 4)

    taker = MultipleTaker(make_circles_grid(width, height, grid_size), cam=cam)
    taker.prepare()

    await wait_for_return()

    while True:
        await taker.take_async(300)

        result = taker.get_results()

        diff = result[0].astype(numpy.int32) - result[1].astype(numpy.int32)
        diff[diff < 0] = 0
        diff[diff > 8] = 255
        diff = 255 - cv2.cvtColor(diff.astype(numpy.uint8), cv2.COLOR_BGR2GRAY)
        cv2.imshow('progress', diff)

        ok, grid = cv2.findCirclesGrid(diff, grid_size, None)
        if ok:
            cv2.destroyWindow('progress')
            break

    diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(diff, grid_size, grid, ok)

    poses = [[grid[x + y*grid_size[0]][0] for x in range(grid_size[0])] for y in range(grid_size[1])]

    left_top = poses[0][0] - (poses[1][0] - poses[0][0]) - (poses[0][1] - poses[0][0])
    right_top = poses[0][-1] - (poses[1][-1] - poses[0][-1]) + (poses[0][-1] - poses[0][-2])
    left_bottom = poses[-1][0] + (poses[-1][0] - poses[-2][0]) - (poses[-1][1] - poses[-1][0])
    right_bottom = poses[-1][-1] + (poses[-1][-1] - poses[-2][-1]) + (poses[-1][-1] - poses[-1][-2])

    cv2.circle(diff, tuple(left_top), 3, (0, 255, 0), -1)
    cv2.circle(diff, tuple(right_top), 3, (0, 255, 0), -1)
    cv2.circle(diff, tuple(left_bottom), 3, (0, 255, 0), -1)
    cv2.circle(diff, tuple(right_bottom), 3, (0, 255, 0), -1)

    cv2.imshow('found', diff)

    return numpy.array([
        [left_top, right_top],
        [left_bottom, right_bottom],
    ])


class PerspectiveCamera(Camera):
    def __init__(self, cam: Camera, warp_matrix: numpy.array, size: typing.Tuple[int, int]):
        self.cam = cam
        self.matrix = warp_matrix
        self.size = size

    @classmethod
    async def auto_create(cls, cam: Camera, width: int, height: int):
        screen = await find_screen(cam, width, height)

        m = cv2.getPerspectiveTransform(
            screen.reshape((4, 2)).astype(numpy.float32),
            numpy.array([[0, 0], [width, 0], [0, height], [width, height]], numpy.float32),
        )

        return cls(cam, m, (width, height))

    def set_mode(self, width: int, height: int) -> None:
        self.cam.set_mode(width, height)

    def get(self) -> numpy.array:
        return cv2.warpPerspective(self.cam.get(), self.matrix, self.size, flags=cv2.INTER_CUBIC)

    def get_size(self) -> typing.Tuple[int, int]:
        return self.size[::-1]
