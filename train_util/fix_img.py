# 실행 예 : python fix_img.py --src-path=/Users/gbox3d/Desktop/work/myproject/neuronetwork/Datasetwork/toy/voc --img-size=640
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

src_path = '/Users/gbox3d/Desktop/work/myproject/neuronetwork/Datasetwork/toy/voc'
img_size = 640
#%% 매개변수 파싱 
parser = argparse.ArgumentParser()
parser.add_argument('--src-path', type=str, default='./', help='source data path')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--rotation', type=int, default=0, help='inference size (pixels)')

opt = parser.parse_args()
# print(opt)
src_path = opt.src_path
img_size = opt.img_size

#%%
_path = Path(src_path)
files = _path.glob('*')
_file_list = list(files)

# %%
image_files = [x for x in _file_list if str(x).split('.')[-1].lower() in ['jpg','png']]
# print(image_files)

#%% make output dir
output_dir = f'{src_path}/out'
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # delete output folder
os.makedirs(output_dir) 

#%%
for _filePath in image_files :
# _filePath = image_files[0]
    _dir,_fname = os.path.split(_filePath)
    _img = Image.open(_filePath)

    if _img.width > _img.height : 
        _ratio = _img.height / _img.width
        rsz_img = _img.resize( ( img_size, int(img_size*_ratio) ),Image.ANTIALIAS )
    else :
        _ratio =  _img.width /_img.height
        rsz_img = _img.resize( ( int(img_size*_ratio),img_size ),Image.ANTIALIAS )

    if opt.rotation == 270 : 
        rsz_img = rsz_img.transpose(Image.ROTATE_270)
    elif opt.rotation == 90 :
        rsz_img = rsz_img.transpose(Image.ROTATE_90)
    elif opt.rotation == 180 :
        rsz_img = rsz_img.transpose(Image.ROTATE_180)
    else :
        print('rotation only 90,180,270')

    rsz_img.save( f'{output_dir}/{_fname}')
    print(f'save : {_fname},{rsz_img.size}')


print(f'done {len(image_files)} files')

# rsz_img.show('out')
# display(rsz_img)
# print(rsz_img.size)


# %%
# np_img = np.array(rsz_img)
# np_img = cv2.cvtColor(np_img,cv2.COLOR_RGB2BGR)
# cv2.imshow('img',np_img)


# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
