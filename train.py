import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt
import visualize
from datasets import my_dataset
from datasets import data_generator
# from datasets.utils import get_original_image
from mrcnn import mask_rcnn

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
assert tf.__version__.startswith('2.')
tf.random.set_seed(22)
np.random.seed(22)

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)
batch_size = 1

num_classes = 2

train_dataset = my_dataset.DemoDataSet('images', 'train')
train_dataset.load_image()

# train_dataset.load_mask()
# load data
img, img_meta = train_dataset[1]

# create model
model = mask_rcnn.MaskRCNN(num_classes=num_classes)

########################### testing  RPN #################################
# TWO THINGS ARE IMPORTANT, RPN ANCHOR REGRESSION AND ROI POOLING
# TODO FOR RPN, NMS NEED TO BE IMPLEMENTED FOR INTERVIEW
proposals = model.simple_test_rpn(img, img_meta)
# after proposals generated, next step is to cut roi region for roi pooling
res = model.simple_test_bboxes(img, img_meta, proposals)
##########################################################################
############################# training ###################################
# inputs = imgs, img_metas, gt_boxes, gt_class_ids
# model((img, img_meta), training=True)
