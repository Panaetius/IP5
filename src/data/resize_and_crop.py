#This file crops (centered) images in the 'orig' directory ending in .JPG and saves them as 250x250px bitmaps in the corresponding train dir
from PIL import Image, ImageChops
import os
from joblib import Parallel, delayed
import multiprocessing


def resize_and_crop(dir, target_dir):
    for root, dirs, files in os.walk(dir):
        Parallel(n_jobs=8)(delayed(CopyAndResizeImage)(fn=fn, dir=dir, root=root, target_dir=target_dir) for fn in files)


def CopyAndResizeImage(dir, fn, root, target_dir):
    if fn.endswith(".JPG"):
        fn = os.path.join(root, fn)
        size = (250, 250)

        image = Image.open(fn)
        image_size = image.size

        bbox = 0

        # resize the image so it's smaller side is 250 pixels
        if image_size[0] < image_size[1]:
            bbox = (size[0], int((size[0] / image_size[0]) * image_size[1]))
        else:
            bbox = (int((size[1] / image_size[1]) * image_size[0]), size[1])

        image = image.resize(bbox, Image.ANTIALIAS)
        image_size = image.size

        # crop the image to 250,250 centered
        thumb = image.crop(((image_size[0] - size[0]) / 2, (image_size[1] - size[1]) / 2, (image_size[0] + size[0]) / 2,
                            (image_size[1] + size[1]) / 2))

        new_fn = fn.replace(dir, target_dir).replace('.JPG', '.PNG')

        # create directory in training dir if it doesn't already exist
        if not os.path.exists(os.path.dirname(new_fn)):
            os.makedirs(os.path.dirname(new_fn))

        thumb.save(new_fn, "PNG", quality=95)