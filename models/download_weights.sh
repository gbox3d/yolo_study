#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov5/releases

python3 - <<EOF
import sys
sys.path.append('../')
from libs.utils.google_utils import attempt_download

for x in ['s', 'm', 'l', 'x']:
    attempt_download(f'yolov5{x}.pt')

EOF