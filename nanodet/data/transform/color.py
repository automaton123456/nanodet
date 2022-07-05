# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import cv2
import imgaug
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np

import cv2
from google.colab.patches import cv2_imshow

def random_brightness(img, delta):
    img += random.uniform(-delta, delta)
    return img


def random_contrast(img, alpha_low, alpha_up):
    img *= random.uniform(alpha_low, alpha_up)
    return img


def random_saturation(img, alpha_low, alpha_up):
    hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] *= random.uniform(alpha_low, alpha_up)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img

def augment_hsv(im, hgain=0.9, sgain=0.9, vgain=0.9):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed

def random_color_map(img):
    #3 in 10
    choice = random.randint(0, 10)
    if choice == 1:
        return cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif choice == 2:    
        return cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    elif choice == 3:    
        return cv2.applyColorMap(img, cv2.COLORMAP_BONE)   
    
    return img


def random_tee(img):    
    #1 in 4
    if random.randint(0,3) == 2:
        return img
        
    b = random.randint(180,255)
    g = random.randint(180,255)
    r = random.randint(180,255)
    line_thickness = random.randint(2,4)
    line_length = random.randint(6,12)

    x = random.randint(0, 320 - line_thickness)
    y  = random.randint(0, 320 - line_length)
    
    offset = random.randint(0,6)
    if offset == 0:
        offset = 0
    elif offset == 5:
        offset = line_thickness
    elif offset == 6:
        offset = 0 - line_thickness
    
    x2 = x

    cv2.line(img, (x, y), (x + offset, y + line_length), (b,g,r), thickness=line_thickness)
    
    return img


def normalize(meta, mean, std):
    img = meta["img"].astype(np.float32)
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    meta["img"] = img
    return meta


def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img

def motion_blur(meta):
    labels = meta['gt_labels']
    gt_bboxes = meta['gt_bboxes']
    all_bboxes = []
    image = meta["img"]
    
    if len(gt_bboxes) == 0:
        return meta
    
    for index,bbox in enumerate(gt_bboxes):
        all_bboxes.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], label=labels[index]))
    
    bbs = BoundingBoxesOnImage(all_bboxes, shape=image.shape)

    rand_angle = random.randrange(-45, 45)
    rand_k = random.randrange(3, 20)
    
    
    #Apply random motion bluring
    aug = iaa.MotionBlur(k=rand_k, angle=rand_angle)   
    
    bbs = aug.augment_bounding_boxes(bbs)
    image = aug.augment(image=image)
    
    bbs = bbs.clip_out_of_image()
    meta["gt_bboxes"] = bbs.to_xyxy_array()
    
    labels = np.array([],dtype=np.int32)

    for box in bbs:
        np.append(labels, box.label)
    
    meta['gt_labels'] = labels
    meta["img"] = image
   
    return meta


def color_aug_and_norm(meta, kwargs):
    img = meta["img"]
    #img = random_color_map(img)
    img = random_tee(img)

    #if "brightness" in kwargs and random.randint(0, 1):
    #    img = random_brightness(img, kwargs["brightness"])

    #if "contrast" in kwargs and random.randint(0, 1):
    #    img = random_contrast(img, *kwargs["contrast"])

    #if "saturation" in kwargs and random.randint(0, 1):
    #    img = random_saturation(img, *kwargs["saturation"])
        
    #Random color using HSV    
    if random.randint(0, 1):
        augment_hsv(img, hgain=0.6, sgain=0.6, vgain=0.6)
    
    #Random contrast 1 in 4
    if random.randint(0, 3) == 0:
        aug = iaa.GammaContrast((0.5, 2.0))
        img = aug.augment(image=img)
        
    #Add random motion blur to images, 1 in 10
    #if random.randint(0, 10) == 0:
        #aug = iaa.MotionBlur(k=[3,32], angle=[-45, 45])
        #img = aug.augment(image=img)
        
        #meta = motion_blur(meta)  
        #img = meta["img"]
        
    img = img.astype(np.float32) / 255    
    
    # cv2.imshow('trans', img)
    # cv2.waitKey(0)
    img = _normalize(img, *kwargs["normalize"])
    meta["img"] = img
    return meta
