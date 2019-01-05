
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# In[1]:

import os
import coco
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import json
from PIL import Image
from config import Config
import utils
import model as modellib
import visualize
from model import log

get_ipython().magic('matplotlib inline')

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# local path to mapillary dataset
DATASET_DIR = os.path.join(ROOT_DIR, "mapillary_dataset")

# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[2]:

class MapillaryConfig(coco.CocoConfig):
    """Configuration for training on the mapillary dataset.
    Derives from the base Config class and overrides values specific
    to the mapillary dataset.
    """
    # Give the configuration a recognizable name
    NAME = "mapillary"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    # this MUST be explicitly defined, or will run into index out of bound error
    NUM_CLASSES = 1 + 37  # background + 37 objects

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
#     IMAGE_MIN_DIM = 1024
#     IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
#     RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
config = MapillaryConfig()
config.display()


# ## Notebook Preferences

# In[3]:

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# In[4]:

class MapillaryDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    
    DEBUG = False
    CLASS_MAP = {}

    CLASSES = ['Bird', 'Ground Animal', 'Crosswalk - Plain', 'Person',                'Bicyclist', 'Motorcyclist', 'Other Rider', 'Lane Marking - Crosswalk',               'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',                'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Street Light',                'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light', 'Traffic Sign (Back)',                'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan',                'Motorcycle', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow']
    
    # local path to image folder, choose 'dev', 'training', or 'testing'
    SUBSET_DIR = ""

    # local path to images inside development folder
    IMG_DIR = ""

    # local path to instance annotations inside development folder
    INS_DIR = ""


    def load_mapillary(self, dataset_dir, subset, class_ids=None,
                  class_map=None):
        
        self.SUBSET_DIR = os.path.join(dataset_dir, subset)
        self.IMG_DIR = os.path.join(self.SUBSET_DIR, 'images')
        self.INS_DIR = os.path.join(self.SUBSET_DIR, 'instances')
        
        # load classes, start with id = 1 to account for background "BG"
        class_id = 1
        for label_id, label in enumerate(class_ids):
            if label["instances"] == True and label["readable"] in self.CLASSES:
                self.CLASS_MAP[label_id] = class_id
                
                if (self.DEBUG):
                    print("{}: Class {} {} added".format(label_id, class_id, label["readable"]))
                    
                self.add_class("mapillary", class_id, label["readable"])
                class_id = class_id + 1
                
        # add images 
        file_names = next(os.walk(self.IMG_DIR))[2]
        for i in range(len(file_names)):
            image_path = os.path.join(self.IMG_DIR, file_names[i])
            base_image = Image.open(image_path)
            w, h = base_image.size
            
            if (self.DEBUG):
                print("Image {} {} x {} added".format(file_names[i], w, h))
                
            self.add_image("mapillary", image_id = i,
                          path = file_names[i],
                           width = w,
                           height = h
                          )

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        This function loads the image from a file.
        """
        info = self.image_info[image_id]
        img_path = os.path.join(self.IMG_DIR, info["path"])
        image = Image.open(img_path)
        image_array = np.array(image)
        return image_array

    def image_reference(self, image_id):
        """Return the local directory path of the image."""
        info = self.image_info[image_id]
        img_path = os.path.join(self.IMG_DIR, info["path"])
        return img_path

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        instance_path = os.path.join(self.INS_DIR, info["path"])
        instance_image = Image.open(instance_path.rsplit(".", 1)[0] + ".png")
        
        # convert labeled data to numpy arrays for better handling
        instance_array = np.array(instance_image, dtype=np.uint16)

        instances = np.unique(instance_array)
        instaces_count = instances.shape[0]
        
        label_ids = instances // 256
        label_id_count = np.unique(label_ids).shape[0]
        
        if (self.DEBUG):
            print("There are {} instances, {} classes labelled instances in the image {}."                  .format(instaces_count, label_id_count, info["path"]))
            
        mask = np.zeros([instance_array.shape[0], instance_array.shape[1], instaces_count], dtype=np.uint8)
        mask_count = 0
        loaded_class_ids = []
        for instance in instances:
            label_id = instance//256
            if (label_id in self.CLASS_MAP):
                m = np.zeros((instance_array.shape[0], instance_array.shape[1]), dtype=np.uint8)
                m[instance_array == instance] = 1
                m_size = np.count_nonzero(m == 1)
                
                # only load mask greater than threshold size, 
                # otherwise bounding box with area zero causes program to crash
                if m_size > 32 ** 2:
                    mask[:, :, mask_count] = m
                    loaded_class_ids.append(self.CLASS_MAP[label_id])
                    mask_count = mask_count + 1
                    if (self.DEBUG):
                        print("Non-zero: {}".format(m_size))
                        print("Mask {} created for instance {} of class {} {}"                              .format(mask_count, instance, self.CLASS_MAP[label_id],                                       self.class_names[self.CLASS_MAP[label_id]]))
        mask = mask[:, :, 0:mask_count]
        return mask, np.array(loaded_class_ids)


# In[5]:

# read in config file
with open(os.path.join(DATASET_DIR, 'config.json')) as config_file:
    class_config = json.load(config_file)
# in this example we are only interested in the labels
labels = class_config['labels']
        
# Training dataset
# dataset_train = MapillaryDataset()
# dataset_train.load_mapillary(DATASET_DIR, "training", class_ids = labels)
# dataset_train.prepare()

# Test dataset
dataset_test = MapillaryDataset()
dataset_test.load_mapillary(DATASET_DIR, "AWS_run_3/testing_1024", class_ids = labels) # TODO: change data directory
dataset_test.prepare()

print("mapping: ", class_config["mapping"])
print("version: ", class_config["version"])
print("folder_structure:", class_config["folder_structure"])
print("There are {} classes in the config file".format(len(labels)))
print("There are {} classes in the model".format(len(dataset_test.class_names)))
for i in range(len(dataset_test.class_names)):
    print("    Class {}: {}".format(i, dataset_test.class_names[i]))


# ## Ceate Model

# In[6]:

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=config,
                          model_dir=MODEL_DIR)

log_path = "mapillary20180319T0404"
model_path = os.path.join(MODEL_DIR, log_path, "mask_rcnn_mapillary_0001.h5")

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# ## Evaluation

# In[7]:

# # # Compute VOC-Style mAP @ IoU=0.5
# # Running on 10 images. Increase for better accuracy.
# image_ids = np.random.choice(dataset_test.image_ids, 5)

# APs = []
# for image_id in image_ids:
#     try:
#         # Load image and ground truth data
#         image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#             modellib.load_image_gt(dataset_test, config,
#                                    image_id, use_mini_mask=False)
#         molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
#         # Run object detection
#         results = model.detect([image], verbose=0)
#         r = results[0]
#         # Compute AP
#         AP, precisions, recalls, overlaps =\
#             utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                              r["rois"], r["class_ids"], r["scores"], r['masks'])
#         APs.append(AP)
#     except:
#         print('Bad picture with id:', image_id)
    
#     if len(APs) % 1 == 0:
#         print('iterating: ', len(APs))
#         print("mAP: ", np.mean(APs))
        
# print("mAP: ", np.mean(APs)) # averaged AP over different classes
# # visualize.plot_precision_recall(AP, precisions, recalls)


# In[ ]:

# Compute mAP @ IoU=0.5:0.95, AP50, AP75
# Running on 2 images. Increase for better accuracy.
evaluation_result = log_path + '_eval.txt'
with open(evaluation_result, 'a') as output: 
    output.write("Start Evaluation: \n")
    
images_number = 1024
image_ids = np.random.choice(dataset_test.image_ids, images_number)
APs = []
AP50s = []
AP75s = []

for image_id in image_ids:
    try:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =            modellib.load_image_gt(dataset_test, config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        mAPs, AP = utils.compute_ap_avg(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r['masks'])
        AP50 = mAPs[0]
        AP75 = mAPs[4]
        APs.append(AP)
        AP50s.append(AP50)
        AP75s.append(AP75)
    except:
        print('Bad picture with id:', image_id, dataset_test.image_reference(image_id))

    if len(APs) % 16 == 0:
        print('iterating: {}'.format(len(APs)))
        print("mAP: {}".format(100.*np.mean(APs))) # averaged AP over IoU: 50~95
        print("mAP50: {}".format(100.*np.mean(AP50s)))
        print("mAP75: {}".format(100.*np.mean(AP75s)))
        
        with open(evaluation_result, 'a') as output:  
            output.write('iterating: {} \n'.format(len(APs)))
            output.write("mAP: {}\n".format(100.*np.mean(APs))) # averaged AP over IoU: 50~95
            output.write("mAP50: {}\n".format(100.*np.mean(AP50s)))
            output.write("mAP75: {}\n".format(100.*np.mean(AP75s)))

with open(evaluation_result, 'a') as output: 
    output.write("\nFinal Result:\nmAP: {}\n".format(100.*np.mean(APs))) # averaged AP over IoU: 50~95
    output.write("mAP50: {}\n".format(100.*np.mean(AP50s)))
    output.write("mAP75: {}\n".format(100.*np.mean(AP75s)))


# In[ ]:

evaluation_result


# In[ ]:



