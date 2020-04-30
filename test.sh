#!/bin/sh
#python -m torch.utils.bottleneck test.py MatrixNetAnchorsResnet50 --testiter 60000 
python test.py MatrixNetAnchorsResnet50_smoothmask_contours --testiter 490000
