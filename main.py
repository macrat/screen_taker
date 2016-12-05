import asyncio
import random

import cv2
import numpy

import converter
import finder


def make_random_imgee(size: tuple, num: int = 10):
    img = numpy.zeros([*size, 3], numpy.uint8)
    img[:,:,0] = random.randint(0, 256)
    img[:,:,1] = random.randint(0, 256)
    img[:,:,2] = random.randint(0, 256)

    for i in range(num):
        r = random.randint(max(*size)//8, max(*size)//2)
        cv2.circle(img, (random.randint(-r, size[1] + r), random.randint(-r, size[0] + r)), r, [random.randint(0, 256) for i in range(3)], -1)

    return cv2.GaussianBlur(img, (19, 19), 512, 0)


async def take_random_images(cam: finder.Camera, width: int, height: int, num: int = 50):
    taker = finder.SimpleImagePhotoTaker(cam)
    taker.prepare()

    await finder.wait_for_return()

    results = []
    inputs = []
    for i in range(num):
        print('{0:.1%}'.format(i / num))

        img = make_random_imgee((height, width))

        result = await taker.take_async(img, 300)

        inputs.append(img)
        results.append(result)

    return numpy.array(inputs), numpy.array(results)


async def main(cam: finder.Camera, width: int, height: int):
    pcam = await finder.PerspectiveCamera.auto_create(cam, width, height)

    xs, ys = await take_random_images(pcam, width, height)

    print('make linear function...')
    f = converter.LinearFunction.from_least_squares(xs, ys)
    print('done')

    with open('f.pickle', 'wb') as fp:
        __import__('pickle').dump(f, fp)

    while True:
        show = make_random_imgee((height, width), num=100)
        cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('screen', show)

        expect = f(show)
        exp_show = expect.copy()
        exp_show[exp_show < 0] = 0
        exp_show[exp_show > 255] = 255
        cv2.imshow('expect', exp_show.astype(numpy.uint8))

        for i in range(3 * 16):
            img = pcam.get()

            img = img - expect
            img[img < 0] = 0
            img[img > 255] = 255
            cv2.imshow('diff', img.astype(numpy.uint8))

            finder.show_image()
            await asyncio.sleep(16 / 1000)


if __name__ == '__main__':
    cam = finder.CameraMonitor()
    cam.set_mode(1280, 800)
    asyncio.get_event_loop().run_until_complete(asyncio.wait([cam.mainloop(), main(cam, 1680, 1050)]))
