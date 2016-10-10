from PIL import Image, ImageChops
import os

for fn in os.listdir('.'):
     if os.path.isfile(fn) and fn.endswith(".JPG"):
        size = (224,224)

        image = Image.open(fn)
        image.thumbnail(size, Image.ANTIALIAS)
        image_size = image.size

        thumb = image.crop( (0, 0, size[0], size[1]) )

        offset_x = int(max( (size[0] - image_size[0]) / 2, 0 ))
        offset_y = int(max( (size[1] - image_size[1]) / 2, 0 ))

        thumb = ImageChops.offset(thumb, offset_x, offset_y)
        thumb.save(fn)