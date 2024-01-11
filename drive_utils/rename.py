import os
from PIL import Image
folder = 'data_night'
files = os.listdir(folder)
for i, file in enumerate(files):
    dst = f'night_{str(i)}.jpg'
    src = f'{folder}/{file}'
    dst = f'{folder}/{dst}'
    os.rename(src, dst)