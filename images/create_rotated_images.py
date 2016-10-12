#creates artificial training images from original ones by rotating them 23 times for 15° each time.
##################
#WARNING! DO NOT run this when you already created rotated images, since it will create rotated ones for those as well, creating exponentially more images!
##################
from PIL import Image, ImageChops, ImageColor
import os
from numpy import random

def average_image_color(i):
	h = i.histogram()

	# split into red, green, blue
	r = h[0:256]
	g = h[256:256*2]
	b = h[256*2: 256*3]

	# perform the weighted average of each channel:
	# the *index* is the channel value, and the *value* is its weight
	return (
		sum( i*w for i, w in enumerate(r) ) / sum(r),
		sum( i*w for i, w in enumerate(g) ) / sum(g),
		sum( i*w for i, w in enumerate(b) ) / sum(b)
	)

def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def get_white_noise_image(width, height):
    #pil_map = Image.new("RGBA", (width, height), 255)
    random_grid = random.randint(0, high=255, size=(width, height, 3))
    #pil_map.putdata(random_grid)
    return Image.fromarray(random_grid.astype('uint8'), 'RGB')

def create_rotated_images(dir):
    count = 0
    for root, dirs, files in os.walk(dir):
        for fn in files:
            if fn.endswith(".PNG"):

                fn = os.path.join(root, fn)
                size = (250,250)
                image = Image.open(fn)
                #image = trim(image)
                average_color = average_image_color(image)

                for i in range(1,23):
                    angle = 15 * i
                    count += 1

                    dst_im = get_white_noise_image(size[0], size[1]).convert('RGBA')
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

                    tmp_im.paste(rot,(int((size[0]-width)/2),int((size[1]-height)/2),int(width + (size[0]-width)/2),int(height + (size[1]-height)/2)))

                    dst_im = Image.composite(tmp_im, dst_im, tmp_im).convert('RGB')
                    dst_im.save(fn.replace(".PNG","_")+str(i)+".PNG", 'PNG', quality=95)

                    end=""

                    if count%100 == 0:
                        end = "\n"
                    print('.', end=end, flush=True)
    print('.', end="\n", flush=True)