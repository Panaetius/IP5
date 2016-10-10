from PIL import Image, ImageChops, ImageColor
import os,random

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
    pil_map = Image.new("RGBA", (width, height), 255)
    random_grid = map(lambda x: (
            int(random.random() * 256),
            int(random.random() * 256),
            int(random.random() * 256)
        ), [0] * width * height)
    pil_map.putdata(list(random_grid))
    return pil_map

for fn in os.listdir('.'):
     if os.path.isfile(fn) and fn.endswith(".JPG"):
        size = (224,224)
        image = Image.open(fn)
        image = trim(image)
        average_color = average_image_color(image)

        for i in range(1,23):
            angle = 15 * i

            dst_im = get_white_noise_image(size[0], size[1])#Image.new("RGBA", size, (int(average_color[0]), int(average_color[1]),int(average_color[2])))
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

            dst_im = Image.composite(tmp_im, dst_im, tmp_im)
            dst_im.save(os.getcwd() + "/" + fn.replace(".JPG","_")+str(i)+".JPG", 'JPEG', quality=100)