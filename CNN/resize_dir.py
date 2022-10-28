from PIL import Image
import os, sys
from tqdm import tqdm


def resize(path):
    dirs = os.listdir( path )
    for item in tqdm(dirs):
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((224,224), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

resize("data/Healthy/")
resize("data/Mild/")
resize("data/Moderate/")
resize("data/Severe/")