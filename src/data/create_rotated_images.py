# creates artificial training images from original ones by rotating them
# 23 times for 15° each time.
##################
# WARNING! DO NOT run this when you already created rotated images, since it
# will create rotated ones for those as well,
# creating exponentially more images!
##################
from PIL import Image, ImageChops
import os
import numpy as np
from skimage.util import random_noise
from joblib import Parallel, delayed


def average_image_color(i):
    # Calculates the average color of an image
    h = i.histogram()

    # split into red, green, blue
    r = h[0:256]
    g = h[256:256 * 2]
    b = h[256 * 2: 256 * 3]

    # perform the weighted average of each channel:
    # the *index* is the channel value, and the *value* is its weight
    return (
        int(sum(i * w for i, w in enumerate(r)) / sum(r)),
        int(sum(i * w for i, w in enumerate(g)) / sum(g)),
        int(sum(i * w for i, w in enumerate(b)) / sum(b))
    )


def trim(im):
    #trims an image to a bounding box
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def get_white_noise_image(width, height):
    # generates a random noise image
    # pil_map = Image.new("RGBA", (width, height), 255)
    random_grid = np.random.randint(0, high=255, size=(width, height, 3))
    # pil_map.putdata(random_grid)
    return Image.fromarray(random_grid.astype('uint8'), 'RGB')


def create_rotated_images(dir, existing):
    # creates rotations of images in a folder with parallelization
    for root, dirs, files in os.walk(dir):
        Parallel(n_jobs=8)(
            delayed(RotateImage)(fn=fn, root=root, existing=existing) for fn in files)


def RotateImage(fn, root, existing):
    # creates 24 copies of an image by rotating is 15 degrees each time
    fn = os.path.join(root, fn)

    if not '/'.join(fn.split('/')[-2:]) in existing and fn.endswith(".PNG"):
        size = (250, 250)
        image = Image.open(fn)
        # image = trim(image)
        average_color = average_image_color(image)

        for i in range(1, 23):
            angle = 15 * i

            # dst_im = get_white_noise_image(size[0], size[1]).convert('RGBA')
            dst_im = Image.new("RGBA", size, average_color)
            im = image.convert('RGBA')
            rot = im.rotate(angle, resample=Image.BICUBIC, expand=1)

            width, height = rot.size  # Get dimensions

            left = 0
            right = width
            top = 0
            bottom = height

            if width > size[0]:
                left = (width - size[0]) / 2
                right = (width + size[0]) / 2
            if height > size[1]:
                top = (height - size[1]) / 2
                bottom = (height + size[1]) / 2

            rot = rot.crop((left, top, right, bottom))
            width, height = rot.size
            tmp_im = Image.new("RGBA", size)

            tmp_im.paste(rot, (
                int((size[0] - width) / 2),
                int((size[1] - height) / 2),
                int(width + (size[0] - width) / 2),
                int(height + (size[1] - height) / 2)))

            dst_im = Image.composite(tmp_im, dst_im, tmp_im).convert('RGB')

            # add random noise
            noise = random_noise(
                np.asarray(dst_im),
                mode='gaussian',
                seed=None,
                clip=True,
                var=0.001)
            dst_im = Image.fromarray(np.uint8(np.multiply(noise, 255.0)))
            dst_im.save(fn.replace(".PNG", "_") + str(i) + ".PNG", 'PNG',
                        quality=95)
