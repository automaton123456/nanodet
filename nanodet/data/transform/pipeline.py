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
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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
    all_boxes = []
    
    for bbox in gt_bboxes:
        all_bboxes.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
        
    if len(all_bboxes) == 0:
        return meta
    
    bbs = BoundingBoxesOnImage(all_bboxes, shape=image.shape)
    
    zoom_bbox = bbox[bbox_index]
    x1 = zoom_bbox[0]
    y1 = zoom_bbox[1]
    x2 = zoom_bbox[2]
    y2 = zoom_bbox[3]

    seq = iaa.Sequential([               
        iaa.Affine(
          translate_px={"x": int(-1 * (((x1+x2) / 2)- (width /2) )), "y": int(-1 * (((y1 + y2)/2)-(height / 2)))},
        ),
        iaa.Affine(
            scale=(2,5)
        ),
        iaa.Fliplr(0.5),
        iaa.Affine(
          translate_percent={"x": (-0.1,0.1), "y": (-0.1,0.1)},
        ),
        imgaug.augmenters.size.CropToFixedSize(width=320, height=320, position="center")
    ])
    
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    bbs_aug = bbs_aug.clip_out_of_image()

    meta["gt_bboxes"] = bbs_aug.to_xyxy_array()
    meta["img"] = image
    meta["height"] = 320
    meta["width"] = 320
   
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
        choice = random.randint(0, 10)
        ball_found = -1
        club_found = -1
        
        if 1 == 1: #choice == 1:
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
