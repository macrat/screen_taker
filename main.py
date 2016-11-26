import asyncio
import random

import cupy
import cv2
import numpy

import converter
import finder


def make_random_imgee(size):
    img = numpy.zeros([*size, 3], numpy.uint8)

    for i in range(1000):
        r = random.randint(0, int(max(size) / ((i+1)/4)))
        cv2.circle(img, (random.randint(-r, size[1] + r), random.randint(-r, size[0] + r)), r, [random.randint(0, 256) for i in range(3)], -1)

    return img


async def take_level_images(cam: finder.Camera, width: int, height: int, num: int = 50):
    taker = finder.SimpleImagePhotoTaker(cam)
    taker.prepare()

    await finder.wait_for_return()

    results = []
    inputs = []
    for i in range(num):
        print('{0:.1%}'.format(i / num))

        img = make_random_imgee((height, width))

        result = await taker.take_async(img, 300)

        cv2.imwrite('out/show/{0:04}.png'.format(i), img)
        cv2.imwrite('out/saw/{0:04}.png'.format(i), result)

        inputs.append(img)
        results.append(result)

    return numpy.array(inputs), numpy.array(results)


async def main(cam: finder.Camera, width: int, height: int):
    pcam = await finder.PerspectiveCamera.auto_create(cam, width, height)

    ys, xs = await take_level_images(pcam, width, height)

    print('make linear function...')
    f = converter.LinearFunction.from_least_squares(cupy.array(xs), cupy.array(ys))
    print('done')

    import pickle
    with open('f.pickle', 'wb') as fp:
        pickle.dump(f, fp)

    while True:
        img = cupy.array(pcam.get(), numpy.uint8)

        img = f(img).get()
        img[img < 0] = 0
        img[img > 255] = 255
        cv2.imshow('unwarp', img.astype(numpy.uint8))

        finder.show_image()
        await asyncio.sleep(16 / 1000)


if __name__ == '__main__':
    cam = finder.CameraMonitor()
    cam.set_mode(1280, 800)
    asyncio.get_event_loop().run_until_complete(asyncio.wait([cam.mainloop(), main(cam, 1680, 1050)]))
