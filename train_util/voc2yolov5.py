#%% voc 혈식을 torch 형식으로 
import os
import shutil
import argparse 
from pathlib import Path
import numpy as np
import cv2
import shutil

from IPython.display import display
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw
import PIL.ImageColor as ImageColor
import PIL.Image as Image

from xml.etree.ElementTree import parse

src_path = '/Users/gbox3d/Desktop/work/myproject/neuronetwork/Datasetwork/toyset/ori_images'
out_path = '../../out'
labelcls_path = '/Users/gbox3d/Desktop/work/myproject/neuronetwork/Datasetwork/toyset/classes.txt'

print('voc2yolov5 version 1.1')

#%% agument parse
parser = argparse.ArgumentParser()
parser.add_argument('--src-path', type=str, default='./images', help='source data path')
parser.add_argument('--out-path', type=str, default='./labels', help='source data path')
parser.add_argument('--class-path', type=str, default='./labels', help='source data path')
parser.add_argument('--imagsz', type=int, default=416, help='source data path')

opt = parser.parse_args()
src_path = opt.src_path
out_path = opt.out_path
labelcls_path = opt.class_path

#%% read label file
label_dic = {}
with open(labelcls_path) as fd:
    # print(fd.read())
    _labels = fd.read().split('\n')
    # print(_labels)
    for index,_label in enumerate(_labels) :
        label_dic[_label] = index
    print(label_dic)


#%% 변환 후 출력 
def _doLabelTxt(out_path,src_path) :
    # _test_dir = './test'
    _out_path = out_path + '/labels'
    if os.path.exists(_out_path):
        shutil.rmtree(_out_path)  # delete output folder
    os.makedirs(_out_path) 

    _path = Path(src_path)
    files = _path.glob('*')
    _file_list = list(files)
    xml_files = [x for x in _file_list if str(x).split('.')[-1].lower() in ['xml']]
    # print(image_files)

    for _file in xml_files :
        tree = parse(_file)
        rootNode = tree.getroot()
        _fname = rootNode.find('filename').text.split('.')[0]

        _fPath = f'{_out_path}/{_fname}.txt'
        print(_fname)
        _objs = rootNode.findall('object')
        for _obj in _objs :
            _label = _obj.find('name').text
            _bbox = _obj.find('bndbox')
            xmin =  float(_bbox.find('xmin').text)
            ymin =  float(_bbox.find('ymin').text)
            xmax =  float(_bbox.find('xmax').text)
            ymax =  float(_bbox.find('ymax').text)

            _imgW = float(rootNode.find('size').find('width').text)
            _imgH = float(rootNode.find('size').find('height').text)

            _xcenter = (((xmin + xmax)/2) / _imgW)
            _ycenter = (((ymin + ymax)/2) / _imgH)
            _w = ((xmax - xmin) / _imgW )
            _h = ((ymax - ymin) / _imgH )


            _out = f'{label_dic[_label]} {_xcenter} {_ycenter} {_w} {_h} \n'
            print(_out)
            with open(_fPath,'a') as fd:
                fd.write(_out)

# %%
def _resizeImg(src_path,out_path,imgsz) :
    _out_path = out_path + '/images'
    if os.path.exists(_out_path):
        shutil.rmtree(_out_path)  # delete output folder
    os.makedirs(_out_path) 

    _path = Path(src_path)
    files = _path.glob('*')
    _file_list = list(files)
    _files = [x for x in _file_list if str(x).split('.')[-1].lower() in ['jpg','png','jpeg']]
    print(_files)

    for _file in _files : 
        _img = Image.open(_file)
        __img = _img.resize((imgsz,imgsz),Image.ANTIALIAS)
        _path,_filename = os.path.split(_file)
        _fPath = f'{_out_path}/{_filename}'
        __img.save(_fPath)
        print(f'resize : {_fPath}')

# %% 라벨 파일 생성
_doLabelTxt(out_path,src_path)

#%%이미지 크기 변경
_resizeImg(src_path,out_path,416)
# %%
