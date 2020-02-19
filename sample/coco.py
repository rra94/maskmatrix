import cv2
import math
import numpy as np
import torch
import random
import string
from random import randrange

from config import system_configs
from utils import crop_image, normalize_, color_jittering_, lighting_
from .utils import random_crop, draw_gaussian, gaussian_radius
from models.bbox_transform import crop_and_resize
from pycocotools import mask as maskUtils
from .visualize import display_instances, display_images
from PIL import Image
import albumentations as A
from pdb import set_trace as bp
import time

def data_augs(image,mask ,bbox, categories, height, width, min_max_height, rand_crop, rand_scales, scale_min, scale_max, ratio_min, ratio_max, debug ):
    params = A.BboxParams(format="coco", label_fields=["bbox_ids"])
    if not debug and rand_crop and rand_scales:
        transforms=[A.HorizontalFlip(p = 0.5),A.RandomResizedCrop(height, width, scale=(scale_min, scale_max), ratio=(ratio_min, ratio_max), p=1.0)]
    elif not debug and rand_crop :
        transforms = [A.HorizontalFlip(p = 0.5),A.RandomSizedCrop(min_max_height, height, width, p=1.0)]
    else:
         transforms = [A.HorizontalFlip(p = 0.5),A.CenterCrop(height, width, p=1.0)]     
    augmentation_pipeline = A.Compose(transforms, params)  
    return  augmentation_pipeline(image = image, masks = mask, bboxes=bbox, bbox_ids=np.arange(len(bbox)) )  

def minimize_mask(mask, bbox, minimask_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()
    """
    mini_mask = np.zeros((mask.shape[0],)+minimask_shape, dtype=bool)
    for i in range(mask.shape[0]):
        m = mask[i,:, :]
        x1, y1, x2, y2 = bbox[i][:4].astype(int)
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            print("Invalid bounding box with area of zero")
        m = np.array(Image.fromarray(m).resize(minimask_shape))
#         scipy.misc.imresize(m.astype(float), minimask_shape, interp='bilinear')
#         print(m)
        mini_mask[i,:, :] = m
#         print(m)
    return mini_mask

def mask_keeps(detections, segmentations):
#     print(segmentations.shape)
    n_seg = segmentations.shape[0]
    s =segmentations.reshape(n_seg, -1)
    print(s.shape)
    keeps = np.nonzero(np.sum(segmentations.reshape(n_seg, -1), axis=1)>0)
    print(keeps)
    return detections[keeps], segmentations[keeps]
    
    

def masks_to_dets(cats, masks):
#     print(cats)
    bboxes = []
    masks_final = []
    
    for i,m in enumerate(masks):
#         print(m,i)
#         if m.any() and m.shape[0]>1:
        print(m)
        print(m.shape)
        print(type(m))
        if m.size<=6:
            continue
#         print(m)
#         print("\----------",m[:,0::2])
#         time.sleep(60)
        x1 = np.amin(m[:,0::2])
        y1 = np.amin(m[:,1::2])
        x2 = np.amax(m[:,0::2])
        y2 = np.amax(m[:,1::2])
        if abs(x2-x1) >1 and abs(y2-y1) > 1 :
            bboxes.append([x1,y1,x2,y2,cats[i].item()])
            masks_final.append(m)
    
    bboxes = np.array(bboxes)
    return bboxes, masks_final
        
    
def annToRLE(input_size, segm):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
   """
    h,w = input_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        rle = segm
    return rle
    
def _full_image_crop(image, detections,segmentations):
    detections    = detections.copy()
    segmentations = [seg.copy() for seg in segmentations]
    height, width = image.shape[0:2]
        
    max_hw = max(height, width)
    center = [height // 2, width // 2]
    size   = [max_hw, max_hw]
    
    image, border, offset = crop_image(image, center, size)
    detections[:, 0:4:2] += border[2]
    detections[:, 1:4:2] += border[0]
    for seg in segmentations:
        if seg.size<=6:
            continue
        seg[:,0::2] += border[2] 
        seg[:,1::2] += border[0] 

    return image, detections, segmentations


def _resize_image(image, detections, size, segmentations):
    detections    = detections.copy()
    segmentations = [seg.copy() for seg in segmentations]
    height, width = image.shape[0:2]
#     print(height, width, "")
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))
    height_ratio = new_height / height
    width_ratio  = new_width  / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
# #     print(segmentations[1])
    for seg in segmentations:
        if seg.size <= 6 :
            continue

        seg[:,0::2] *= width_ratio 
        seg[:,1::2] *= height_ratio 

    
    return image, detections, segmentations


def _clip_detections(image, detections, segmentations):
    detections    = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    
#     for seg in segmentations:
#         if seg.size <= 6:
#             continue
#         seg[:, 0::2] = np.clip(seg[:, 0::2], 0, width - 1)
#         seg[:, 1::2] = np.clip(seg[:, 1::2], 0, height - 1)

    keep_inds  = ((detections[:, 2] - detections[:, 0]) >= 1) & \
                 ((detections[:, 3] - detections[:, 1]) >= 1)
#     print(keep_inds)
    detections = detections[keep_inds]
    segmentations = segmentations[keep_inds]
    
    return detections, segmentations



def layer_map_using_ranges(width, height, layer_ranges, fpn_flag=0):
    layers = []
   
    for i, layer_range in enumerate(layer_ranges):
        if fpn_flag ==0:
            if (width >= 0.8 * layer_range[2]) and (width <= 1.3 * layer_range[3]) and (height >= 0.8 * layer_range[0]) and (height <= 1.3 * layer_range[1]):
                layers.append(i)
        else:
            max_dim = max(height, width)
            if max_dim <= 1.3*layer_range[1] and max_dim >= 0.8* layer_range[0]:
                layers.append(i)
    if len(layers) > 0:
        return layers
    else:
        return [len(layer_ranges) - 1]
    

def cutout(image, detections): 
    for detection in detections:
        center = [random.randint(int(detection[1]), int(detection[3])), random.randint(int(detection[0]), int(detection[2]))]
        center_random = [random.randint(0, image.shape[0]), random.randint(0, image.shape[1])]            
        if np.random.uniform() > 0.5:
            width = max(1, int(0.1 * (detection[2] - detection[0]) * (1 + 0.3 * np.random.normal())))
            height = max(1, int((detection[3] - detection[1]) * (1 + np.random.normal())))
        else:
            width = max(1, int( (detection[2] - detection[0]) * (1 + np.random.normal())))
            height = max(1, int(0.1 * (detection[3] - detection[1]) * (1 + 0.3 * np.random.normal())))  

        if np.random.uniform() > 0.5:
            x1 = max(0, center[1] - int(width/2))
            x2 = min(image.shape[1], center[1] + int(width/2))
            y1 = max(0, center[0] - int(height/2))
            y2 = min(image.shape[0], center[0] + int(height/2))          
            image[y1:y2, x1:x2,:] = 0
        else:
            x1 = max(0, center_random[1] - int(width/2))
            x2 = min(image.shape[1], center_random[1] + int(width/2))
            y1 = max(0, center_random[0] - int(height/2))
            y2 = min(image.shape[0], center_random[0] + int(height/2))
            
            image[y1:y2, x1:x2,:] = 0            
           
    return image

def samples_MatrixNetCorners(db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size
    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]
   

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]
    cutout_flag   = db.configs["cutout"]
    max_dim       = db.configs["train_image_max_dim"]
    
    width_thresholds = db.configs["width_thresholds"]
    height_thresholds = db.configs["height_thresholds"]
    layers_range = db.configs["layers_range"]

    
    max_tag_len = 128

    _dict={}
    output_sizes=[]
    # indexing layer map
    for i,l in enumerate(layers_range):
        for j,e in enumerate(l):
            if e !=-1:
                output_sizes.append([input_size[0]//(8*2**(j)), input_size[1]//(8*2**(i))])
                _dict[(i+1)*10+(j+1)]=e
    
    layers_range=[_dict[i] for i in sorted(_dict)]
    
    fpn_flag = set(_dict.keys()) == set([11,22,33,44,55])
    # allocating memory
    print(fpn_flag) 
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    tl_heatmaps = [np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32) for output_size in output_sizes]
    br_heatmaps = [np.zeros((batch_size, categories, output_size[0], output_size[1]), dtype=np.float32) for output_size in output_sizes]
    tl_regrs    = [np.zeros((batch_size, max_tag_len, 2), dtype=np.float32) for output_size in output_sizes]
    center_regrs = [np.zeros((batch_size, max_tag_len, 2), dtype=np.float32) for output_size in output_sizes]
    br_regrs    = [np.zeros((batch_size, max_tag_len, 2), dtype=np.float32) for output_size in output_sizes]

    tl_tags = [np.zeros((batch_size, max_tag_len), dtype=np.int64) for output_size in output_sizes]
    br_tags = [np.zeros((batch_size, max_tag_len), dtype=np.int64) for output_size in output_sizes]

    tag_masks   = [np.zeros((batch_size, max_tag_len), dtype=bool) for output_size in output_sizes]
    tag_lens    = [np.zeros((batch_size, ), dtype=np.int32) for output_size in output_sizes]

    db_size = db.db_inds.size
    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size

        # reading image
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)
        # reading detections
        detections = db.detections(db_ind)
     
             #add seg cutout and crop
        if cutout_flag:
            image = cutout(image, detections)

        if not debug and rand_crop:
            image, detections = random_crop(image, detections, rand_scales, input_size, border=border)
        else:
            image, detections = _full_image_crop(image, detections)
        #resize sw
        image, detections  = _resize_image(image, detections, input_size)
        detections = _clip_detections(image, detections)

        
        # flipping an image randomly
#         if not debug and np.random.uniform() > 0.5:
#             image[:] = image[:, ::-1, :]
#             width    = image.shape[1]
#             detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
        #add seg flip
    
        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)

        images[b_ind] = image.transpose((2, 0, 1))

        for ind, detection in enumerate(detections):
            for olayer_idx in layer_map_using_ranges(detection[2] - detection[0], detection[3] - detection[1],layers_range, fpn_flag):
            
                width_ratio = output_sizes[olayer_idx][1] / input_size[1]
                height_ratio = output_sizes[olayer_idx][0] / input_size[0]

                category = int(detection[-1]) - 1
                xtl, ytl = detection[0], detection[1]
                xbr, ybr = detection[2], detection[3]

                fxtl = (xtl * width_ratio)
                fytl = (ytl * height_ratio)
                fxbr = (xbr * width_ratio)
                fybr = (ybr * height_ratio)

                mx = output_sizes[olayer_idx][1] - 1
                my = output_sizes[olayer_idx][0] - 1
                
                xtl = int(min(round(fxtl), mx))
                ytl = int(min(round(fytl), my))
                xbr = int(min(round(fxbr), mx))
                ybr = int(min(round(fybr), my))                     
                if gaussian_bump:
                    width  = detection[2] - detection[0]
                    height = detection[3] - detection[1]

                    width  = math.ceil(width * width_ratio)
                    height = math.ceil(height * height_ratio)

                    if gaussian_rad == -1:
                        radius = gaussian_radius((height, width), gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = gaussian_rad

                    draw_gaussian(tl_heatmaps[olayer_idx][b_ind, category], [xtl, ytl], radius)
                    draw_gaussian(br_heatmaps[olayer_idx][b_ind, category], [xbr, ybr], radius)

                else:
                    tl_heatmaps[olayer_idx][b_ind, category, ytl, xtl] = 1
                    br_heatmaps[olayer_idx][b_ind, category, ybr, xbr] = 1
                

                tag_ind = tag_lens[olayer_idx][b_ind]
                tl_regrs[olayer_idx][b_ind, tag_ind, :] = [fxtl - xtl, fytl - ytl]
                br_regrs[olayer_idx][b_ind, tag_ind, :] = [fxbr - xbr, fybr - ybr]
                
                center_regrs[olayer_idx][b_ind, tag_ind, :] = [(fxbr - fxtl)/2.0/output_sizes[-1][1],
                                                               (fybr - fytl)/2.0/output_sizes[-1][0]]

                tl_tags[olayer_idx][b_ind, tag_ind] = ytl * output_sizes[olayer_idx][1] + xtl
                br_tags[olayer_idx][b_ind, tag_ind] = ybr * output_sizes[olayer_idx][1] + xbr
                tag_lens[olayer_idx][b_ind] += 1

    for b_ind in range(batch_size):
        for olayer_idx in range(len(tag_lens)):
            tag_len = tag_lens[olayer_idx][b_ind]
            tag_masks[olayer_idx][b_ind, :tag_len] = 1

    images      = [torch.from_numpy(images)]
    tl_heatmaps = [torch.from_numpy(tl) for tl in tl_heatmaps]
    br_heatmaps = [torch.from_numpy(br) for br in br_heatmaps]

    tl_regrs    = [torch.from_numpy(tl) for tl in tl_regrs]
    br_regrs    = [torch.from_numpy(br) for br in br_regrs]
    center_regrs = [torch.from_numpy(c) for c in center_regrs]
    tl_tags     = [torch.from_numpy(tl) for tl in tl_tags]
    br_tags     = [torch.from_numpy(br) for br in br_tags]

    tag_masks   = [torch.from_numpy(tags) for tags in tag_masks]
    return {
        "xs": [images, tl_tags, br_tags],
        "ys": [tl_heatmaps, br_heatmaps, tag_masks, tl_regrs, br_regrs, center_regrs]
    }, k_ind


def samples_MatrixNetAnchors(db, k_ind, data_aug, debug):
    data_rng   = system_configs.data_rng
    batch_size = system_configs.batch_size
    categories   = db.configs["categories"]
    input_size   = db.configs["input_size"]

    border        = db.configs["border"]
    lighting      = db.configs["lighting"]
    rand_crop     = db.configs["rand_crop"]
    rand_color    = db.configs["rand_color"]
    rand_scales   = db.configs["rand_scales"]
    gaussian_bump = db.configs["gaussian_bump"]
    gaussian_iou  = db.configs["gaussian_iou"]
    gaussian_rad  = db.configs["gaussian_radius"]
    base_layer_range = db.configs["base_layer_range"] 
    cutout_flag   = db.configs["cutout"]
    max_dim       = db.configs["train_image_max_dim"]

    width_thresholds = db.configs["width_thresholds"]
    height_thresholds = db.configs["height_thresholds"]
    layers_range = db.configs["layers_range"]
    max_tag_len = 256
    minimask_shape = (56, 56)
    ratios ={}
    _dict={}
    output_sizes=[]
    k=0
    # indexing layer map
    for i,l in enumerate(layers_range):
        for j,e in enumerate(l):
            if e !=-1:
                output_sizes.append([input_size[0]//(8*2**(j)), input_size[1]//(8*2**(i))])
                _dict[(i+1)*10+(j+1)]=e
                ratios[k] = [1/(2**(j))/(base_layer_range[2]), 1/(2**(i))/(base_layer_range[0])]
                k+=1
    
    layers_range=[_dict[i] for i in sorted(_dict)]
    fpn_flag = set(_dict.keys()) == set([11,22,33,44,55]) #creating fpn flag
    
  
    images      = np.zeros((batch_size, 3, input_size[0], input_size[1]), dtype=np.float32)
    anchors_heatmaps = [np.zeros((batch_size, 1, output_size[0], output_size[1]), dtype=np.float32) for output_size in output_sizes]
    detections_batch     = np.zeros((batch_size,200, 7), dtype=np.float32) 
    segmentations_batch =  np.zeros((batch_size,200, minimask_shape[0],minimask_shape[1] ), dtype=np.int32) 
    mask_detections_batch     = np.zeros((batch_size,200, 7), dtype=np.float32) 
    tl_corners_regrs    = [np.zeros((batch_size, max_tag_len, 2), dtype=np.float32) for output_size in output_sizes]
    br_corners_regrs    = [np.zeros((batch_size, max_tag_len, 2), dtype=np.float32) for output_size in output_sizes]
    anchors_tags = [np.zeros((batch_size, max_tag_len), dtype=np.int64) for output_size in output_sizes]
    tag_masks   = [np.zeros((batch_size, max_tag_len), dtype=bool) for output_size in output_sizes]
    tag_lens    = [np.zeros((batch_size, ), dtype=np.int32) for output_size in output_sizes]
    
    
    db_size = db.db_inds.size
#     db_size = 10
    

    for b_ind in range(batch_size):
        if not debug and k_ind == 0:
            db.shuffle_inds()

        db_ind = db.db_inds[k_ind]
        k_ind  = (k_ind + 1) % db_size
        
        image_file = db.image_file(db_ind)
        image      = cv2.imread(image_file)
        
        
        # reading detections
        detections, categories = db.detections(db_ind)
        segmentations = db.segmentations(db_ind)
#         print(detections, categories)
#         print([seg.size for seg in segmentations])
            
#         if not debug and rand_crop:
#             image, detections,segmentations = random_crop(image, detections, rand_scales, input_size, border=border, segmentations=segmentations)
#         else:
#             image, detections, segmentations = _full_image_crop(image, detections, segmentations)
#         image, detections, segmentations = _resize_image(image, detections, input_size, segmentations)   
#         detections, segmentations = _clip_detections(image, detections, segmentations)

        # flipping an image randomly

#         if not debug and np.random.uniform() > 0.5:
#             image[:] = image[:, ::-1, :]
#             width    = image.shape[1]
#             detections[:, [0, 2]] = width - detections[:, [2, 0]] - 1
            
#             for seg in segmentations:
#                 if seg.size<6:
#                     continue
#                 seg[:, 0::2] = width - seg[:, 0::2] - 1
            
        
#         mask_dets, segmentations = masks_to_dets(cats, segmentations)

        segmentations = [maskUtils.decode(annToRLE(image.shape[:2] , rle)) for rle in segmentations]
#         dets = detections[:,0:4].tolist()
#         print(dets)
#         segmentations = np.stack(segmentations, axis=2).astype(np.bool)
#         segmentations=segmentations.transpose((2, 0, 1))
#         segmentations =    np.expand_dims(segmentations, axis=1)
#         categories = detections[:,4].tolist()
        agumented = data_augs(image, segmentations ,detections, categories, input_size[0], input_size[1], input_size[0]-border, rand_crop, True,rand_scales[0], rand_scales[1], 0.8, 1.25, debug)

        image = agumented['image']
        segmentations = agumented['masks']
        detections  = agumented['bboxes']
        bbox_ids = agumented['bbox_ids']
#         print(bbox_ids)

        categories = np.expand_dims(categories, axis=1)
        categories= categories[bbox_ids]
        
    
        detections = np.array(detections)
        detections[:, 2] +=detections[:,0]
        detections[:, 3] +=detections[:,1]
        detections = np.concatenate((np.array(detections), categories), axis=1)

        segmentations_ = np.stack(segmentations, axis=2).astype(np.bool)
        segmentations=segmentations_.transpose((2, 0, 1))
        
        print(0, detections.shape, segmentations.shape )
        
        segmentations = segmentations[bbox_ids]
                                    
        print(1, detections.shape, segmentations.shape )
        
        detections, segmentations  = _clip_detections(image, detections, segmentations)
        print(2, detections.shape, segmentations.shape )
        
        mask_dets, segmentations = mask_keeps(detections, segmentations)
        segmentations = minimize_mask(segmentations, mask_dets, minimask_shape)
        print(3, mask_dets.shape, segmentations.shape )
        
        
        display_instances(image, detections, segmentations_, detections[:,-1], "/home/rragarwal4/matrixnet/imgs/", str(db_ind)+"_reformatted.jpg")

                              
# #         print(len(segmentations))
#         if len(segmentations) >0:
#             segmentations = [maskUtils.decode(annToRLE(image.shape[:2] , rle.tolist())) for rle in segmentations]
#             segmentations_ = np.stack(segmentations, axis=2).astype(np.bool)
#             segmentations=segmentations_.transpose((2, 0, 1))
#             segmentations = minimize_mask(segmentations, mask_dets, minimask_shape)
#         else:
#             segmentations = np.zeros((1,)+minimask_shape)

#         display_instances(image, np.array(mask_dets), segmentations_, np.array(mask_dets[:,-1]), "/home/rragarwal4/matrixnet/imgs/", str(db_ind)+"_reformatted.jpg")

        if not debug:
            image = image.astype(np.float32) / 255.
            if rand_color:
                color_jittering_(data_rng, image)
                if lighting:
                    lighting_(data_rng, image, 0.1, db.eig_val, db.eig_vec)
        images[b_ind] = image.transpose((2, 0, 1))
        
        
        
#         print(np.vstack((detections[1:4,:],mask_dets[1:4,:])), "--------------")
        dets = []
        msks = [] 
        m_dets = []
        
        for ind, detection in enumerate(mask_dets):
            for olayer_idx in layer_map_using_ranges(detection[2] - detection[0], detection[3] - detection[1],layers_range, fpn_flag):
                msks.append(segmentations[ind:ind+1])
                m_dets.append([0] + list(detection) +[olayer_idx])
      
        for ind, detection in enumerate(detections):
            for olayer_idx in layer_map_using_ranges(detection[2] - detection[0], detection[3] - detection[1],layers_range, fpn_flag):
                dets.append([0] + list(detection) +[olayer_idx])
                
                width_ratio = output_sizes[olayer_idx][1] / input_size[1]
                height_ratio = output_sizes[olayer_idx][0] / input_size[0]
                
               # if categories ==1:
                #    category = 0 
                #else:
                
                category = 0
                xtl, ytl = detection[0], detection[1]
                xbr, ybr = detection[2], detection[3]
                
                fxtl = (xtl * width_ratio)
                fytl = (ytl * height_ratio)
                fxbr = (xbr * width_ratio)
                fybr = (ybr * height_ratio)

                mx = output_sizes[olayer_idx][1] - 1
                my = output_sizes[olayer_idx][0] - 1
                
                xc = int(min(round((fxtl+fxbr)/2), mx))
                yc = int(min(round((fytl+fybr)/2), my))

                if gaussian_bump:
                    width  = detection[2] - detection[0]
                    height = detection[3] - detection[1]

                    width  = math.ceil(width * width_ratio)
                    height = math.ceil(height * height_ratio)

                    if gaussian_rad == -1:
                        radius = gaussian_radius((height, width), gaussian_iou)
                        radius = max(0, int(radius))
                    else:
                        radius = gaussian_rad
                    #for RPN cange 1 to categoy
                    draw_gaussian(anchors_heatmaps[olayer_idx][b_ind, 0], [xc, yc], radius)

                else:
                    anchors_heatmaps[olayer_idx][b_ind, 0, yc, xc] = 1

                tag_ind = tag_lens[olayer_idx][b_ind]
                min_y, max_y , min_x, max_x = map(lambda x: x/8/2, base_layer_range)

                tl_corners_regrs[olayer_idx][b_ind, tag_ind, :] = [((xc - fxtl) - (max_x+min_x)/2)/ (max_x-min_x),
                                                                ((yc - fytl) - (max_y+min_y)/2)/ (max_y-min_y)]
                br_corners_regrs[olayer_idx][b_ind, tag_ind, :] = [((fxbr - xc) - (max_x+min_x)/2)/ (max_x-min_x),
                                                                ((fybr - yc) - (max_y+min_y)/2)/ (max_y-min_y)]
                
                anchors_tags[olayer_idx][b_ind, tag_ind] = yc * output_sizes[olayer_idx][1] + xc
                tag_lens[olayer_idx][b_ind] += 1
                
        if len(dets) > 0:
            detections_batch[b_ind][:len(dets),:] = np.array(dets)
        else:
            print("zero dets in image")
#         print(msks)
        if len(msks) > 0:
            msks = np.vstack(msks)
            segmentations_batch[b_ind][:len(msks),:,:] = msks
            m_dets= np.array(m_dets)
#             print(m_dets.shape)
            mask_detections_batch[b_ind][:len(m_dets),:] = m_dets
        else:
            print("zero seg in image")

        
    for b_ind in range(batch_size):
        for olayer_idx in range(len(tag_lens)):
            tag_len = tag_lens[olayer_idx][b_ind]
            tag_masks[olayer_idx][b_ind, :tag_len] = 1
    

    images= [torch.from_numpy(images)]
    anchors_heatmaps = [torch.from_numpy(anchors) for anchors in anchors_heatmaps]
    tl_corners_regrs    = [torch.from_numpy(c) for c in tl_corners_regrs]
    br_corners_regrs    = [torch.from_numpy(c) for c in br_corners_regrs]
    anchors_tags     = [torch.from_numpy(t) for t in anchors_tags]
    tag_masks   = [torch.from_numpy(tags) for tags in tag_masks]
    detections_batch = [torch.from_numpy(detections_batch)]
    mask_detections_batch = [torch.from_numpy(mask_detections_batch)]
    ratios =[ [ [i]+ ratios[i] for i in ratios] for _ in range(batch_size)]
    ratios = [torch.from_numpy(np.array(ratios))]
    segmentations_batch = [torch.from_numpy(segmentations_batch)]

    return {
        "xs": [images, anchors_tags, detections_batch, ratios, segmentations_batch, mask_detections_batch ],
        "ys": [anchors_heatmaps, tl_corners_regrs, br_corners_regrs, tag_masks]
    }, k_ind
def sample_data(db, k_ind, data_aug=True, debug=False):
    return globals()["samples_"+system_configs.model_name](db, k_ind, data_aug, debug)
