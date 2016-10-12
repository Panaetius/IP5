#splits the images in the train folder into train (70%), test (15%), valid(15%) and creates corresponding load-files for tensorflow
import os
from numpy import random
import create_rotated_images
import resize_and_crop

def create_import_list(top):
    entries = []
    for root, dirs, files in os.walk(top):
        for fn in files:
            if fn.endswith(".PNG"):
                fn = os.path.join(root, fn)
                folder_path = os.path.dirname(fn)
                _, folder_name = os.path.split(folder_path)
                entries.append(fn + ' ' + folder_name)
    #shuffle list to improve training
    f = open(os.path.join(top, 'files.txt'), 'w')
    random.shuffle(entries)
    f.write('\n'.join(entries))
    f.close()

# print('Resizing original images to 250x250 with centered crop')
# resize_and_crop.resize_and_crop('./orig', './train')
#
# #copy images to test and validation
# print('Randomly copying images to test and validation directories')
# i=0
# for root, dirs, files in os.walk('./train'):
#     for fn in files:
#         if fn.endswith(".PNG"):
#             i+=1
#             fn = os.path.join(root, fn)
#             rand = random.random()
#
#             new_fn = ''
#
#             if rand > 0.85:
#                 #move to test
#                 new_fn = fn.replace('train', 'test')
#
#             elif rand > 0.7:
#                 #move to valid
#                 new_fn = fn.replace('train', 'validation')
#             else:
#                 continue
#
#             if not os.path.exists(os.path.dirname(new_fn)):
#                 os.makedirs(os.path.dirname(new_fn))
#
#             os.rename(fn, new_fn)
#             end = ""
#
#             if i % 100 == 0:
#                 end = "\n"
#             print('.', end=end, flush=True)
# print('.', end="\n", flush=True)

print('Creating rotations of train')
create_rotated_images.create_rotated_images('./train')
print('Creating rotations of test')
create_rotated_images.create_rotated_images('./test')
print('Creating rotations of validation')
create_rotated_images.create_rotated_images('./validation')

print('Creating import file lists')
create_import_list('./train')
create_import_list('./test')
create_import_list('./validation')