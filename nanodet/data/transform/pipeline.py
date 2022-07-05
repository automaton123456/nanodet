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

import functools
import warnings
import random
import imgaug
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np

from typing import Dict, Tuple

from torch.utils.data import Dataset

from .color import color_aug_and_norm
from .warp import ShapeTransform, warp_and_resize

def zoom_to_bbox(meta, bbox_index, dst_shape):
    filename = meta['img_info']['file_name']
    width = meta['img_info']['width']
    height = meta['img_info']['height']
    labels = meta['gt_labels']
    gt_bboxes = meta['gt_bboxes']
    all_bboxes = []
    image = meta["img"]
    
    for index,bbox in enumerate(gt_bboxes):
        all_bboxes.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], label=labels[index]))
        
    if len(all_bboxes) == 0:
        return meta
    
    bbs = BoundingBoxesOnImage(all_bboxes, shape=image.shape)
    
    zoom_bbox = gt_bboxes[bbox_index]
    x1 = zoom_bbox[0]
    y1 = zoom_bbox[1]
    x2 = zoom_bbox[2]
    y2 = zoom_bbox[3]

    tr_x = random.uniform(-0.1,0.1)
    tr_y = random.uniform(-0.1,0.1)

    scale=random.uniform(2,5)

    rotate = random.choice([-90,-90,0,0,90,180,270])
    
    #Translate object of interest to center, then scale, then flip, then translate randomly, then crop to center square
    aug1 = iaa.Affine(translate_px={"x": int(-1 * (((x1+x2) / 2)- (width /2) )), "y": int(-1 * (((y1 + y2)/2)-(height / 2)))})
    aug2 = iaa.Affine(scale=scale)
    aug3 = iaa.Fliplr(0.5)
    aug4 = iaa.Affine(translate_percent={"x": tr_x, "y": tr_y})  
    aug5 = imgaug.augmenters.size.CropToFixedSize(width=320, height=320, position="center")
    aug6 = imgaug.augmenters.geometric.Rotate(rotate)
    
    bbs = aug1.augment_bounding_boxes(bbs)
    image = aug1.augment(image=image)
    
    bbs = aug2.augment_bounding_boxes(bbs)
    image = aug2.augment(image=image)
    
    bbs = aug3.augment_bounding_boxes(bbs)
    image = aug3.augment(image=image)
    
    image = aug4.augment(image=image)
    bbs = aug4.augment_bounding_boxes(bbs)
    
    bbs = aug5.augment_bounding_boxes(bbs)
    image = aug5.augment(image=image)
    
    bbs = aug6.augment_bounding_boxes(bbs)
    image = aug6.augment(image=image)
    
    bbs = bbs.clip_out_of_image()
    meta["gt_bboxes"] = bbs.to_xyxy_array()
    
    labels = np.array([],dtype=np.int32)

    for box in bbs:
        np.append(labels, box.label)
    
    meta['gt_labels'] = labels
    
    meta["img"] = image
    meta["warp_matrix"] = []
   
    return meta


class LegacyPipeline:
    def __init__(self, cfg, keep_ratio):
        warnings.warn(
            "Deprecated warning! Pipeline from nanodet v0.x has been deprecated,"
            "Please use new Pipeline and update your config!"
        )
        
        self.warp = functools.partial(
            warp_and_resize, warp_kwargs=cfg, keep_ratio=keep_ratio
        )
        
        
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, meta, dst_shape):
        meta = self.warp(meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta


class Pipeline:
    """Data process pipeline. Apply augmentation and pre-processing on
    meta_data from dataset.

    Args:
        cfg (Dict): Data pipeline config.
        keep_ratio (bool): Whether to keep aspect ratio when resizing image.

    """

    def __init__(self, cfg: Dict, keep_ratio: bool):
        self.shape_transform = ShapeTransform(keep_ratio, **cfg)
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, dataset: Dataset, meta: Dict, dst_shape: Tuple[int, int]):
        print(self.cfg)
        print(dataset)
        choice = random.randint(0, 8)
        ball_found = -1
        club_found = -1
        
        if choice == 1:
            labels = meta['gt_labels']
            
            if random.randint(0, 1):
                for i, data in enumerate(labels):
                    if data == 1:
                        ball_found = i;
                        break
                        
                if ball_found > -1:
                    meta = zoom_to_bbox(meta, bbox_index=ball_found, dst_shape=dst_shape)
                else:
                    meta = self.shape_transform(meta, dst_shape=dst_shape)
            else:
                for i, data in enumerate(labels):
                    if data == 2:
                        club_found = i;
                        break
                        
                if club_found > -1:        
                    meta = zoom_to_bbox(meta, bbox_index=club_found, dst_shape=dst_shape)
                else:
                    meta = self.shape_transform(meta, dst_shape=dst_shape)                 
        else:
            meta = self.shape_transform(meta, dst_shape=dst_shape)
   
       
        meta = self.color(meta=meta)
        return meta
