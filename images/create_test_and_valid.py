#splits the images in the train folder into train (70%), test (15%), valid(15%) and creates corresponding load-files for tensorflow
import os
from numpy import random

def create_import_list(top):
    f = open(os.path.join(top, 'files.txt'), 'w')
    for root, dirs, files in os.walk(top):
        for fn in files:
            if fn.endswith(".PNG"):
                fn = os.path.join(root, fn)
                folder_path = os.path.dirname(fn)
                _, folder_name = os.path.split(folder_path)
                f.write(fn + ' ' + folder_name + '\n')

    f.close()

for root, dirs, files in os.walk('./train'):
    for fn in files:
        if fn.endswith(".PNG"):
            fn = os.path.join(root, fn)
            rand = random.random()

            new_fn = ''

            if rand > 0.85:
                #move to test
                new_fn = fn.replace('train', 'test')

            elif rand > 0.7:
                #move to valid
                new_fn = fn.replace('train', 'validation')
            else:
                continue

            if not os.path.exists(os.path.dirname(new_fn)):
                os.makedirs(os.path.dirname(new_fn))

            os.rename(fn, new_fn)

create_import_list('./train')
create_import_list('./test')
create_import_list('./validation')