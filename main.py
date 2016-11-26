import asyncio

import cupy
import cv2
import numpy

import converter
import finder


async def take_level_images(cam: finder.Camera, width: int, height: int, step: int = 51, loop: int = 10):
    taker = finder.SimpleImagePhotoTaker(cam)
    taker.prepare()

    inputs = [
        numpy.ones((height, width, 3), numpy.uint8) * i
        for i in range(0, 255, step)
    ]

    await finder.wait_for_return()

    results = []
    for i in range(loop):
        for j, img in enumerate(inputs):
            id_ = i * len(inputs) + j

            print('{0:.1%}'.format(id_ / (loop * len(inputs))))

            result = await taker.take_async(img, 500)

            cv2.imwrite('out/show/{0:04}.png'.format(id_), img)
            cv2.imwrite('out/saw/{0:04}.png'.format(id_), result)

            results.append(result)

    return numpy.array(inputs * loop), numpy.array(results)


async def main(cam: finder.Camera, width: int, height: int):
    pcam = await finder.PerspectiveCamera.auto_create(cam, width, height)

    ys, xs = await take_level_images(pcam, width, height, step=51, loop=10)

    f = converter.LinearFunction.from_least_squares(cupy.array(xs), cupy.array(ys))

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
    cam = finder.CameraMonitor(1)
    cam.set_mode(1280, 800)
    asyncio.get_event_loop().run_until_complete(asyncio.wait([cam.mainloop(), main(cam, 1680, 1050)]))
