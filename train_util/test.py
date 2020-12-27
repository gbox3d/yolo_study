#%%
import os
import shutil
import argparse 
from pathlib import Path
import numpy as np
import cv2

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

#%%
_img = Image.open('../../out/images/eggle_1_0.jpg')

# _img = _img.resize((416,416),Image.ANTIALIAS)

with open('../../out/labels/eggle_1_0.txt') as fd :
    lines = fd.readlines()

print(lines)

_col = lines[0].split(' ')

_xcenter = float(_col[1]) * _img.width 
_ycenter = float(_col[2]) * _img.height 
_width = float(_col[3]) * _img.width 
_height = float(_col[4]) * _img.height 

# print(_col)

drawer = ImageDraw.Draw(_img)
drawer.line(
    [(_xcenter-(_width/2)  ,_ycenter-(_height/2)),
    (_xcenter+(_width/2)  ,_ycenter-(_height/2)),
    (_xcenter+(_width/2)  ,_ycenter+(_height/2)),
    (_xcenter-(_width/2)  ,_ycenter+(_height/2)),
    (_xcenter-(_width/2)  ,_ycenter-(_height/2)),
    ],
    width=2,fill=(0,255,0))

display(_img)
# %%
