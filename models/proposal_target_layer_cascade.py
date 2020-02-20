from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------
import time
import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
#from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch, crop_and_resize
import pdb
from  torchvision.utils import save_image

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.FG_THRESH = 0.7
        self.BG_THRESH_HI = 0.3
        self.BG_THRESH_LO = 0.0
        self.mask_shape = 28
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor([1,1,1,1])
        
    def forward(self, all_rois, gt_boxes, gt_masks, mask_gt_boxes, ratios):
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)
        all_rois[:,:,1:5] = 8 * all_rois[:,:, 1:5]
        all_rois = torch.cat([all_rois, gt_boxes], 1)
        BATCH_SIZE  = gt_boxes.shape[0] #what batch size is this?
        num_images = 1
        rois_per_image = int(256) #int(BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(0.25 * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        
        labels, rois, bbox_target_data_tl, bbox_target_data_br = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image, rois_per_image,  ratios)
        
        target_masks, mask_select, mask_labels = self._masks_assignment(rois , mask_gt_boxes,  gt_masks)

        bbox_targets_tl, bbox_targets_br, bbox_inside_weights = self._get_bbox_regression_labels_pytorch(bbox_target_data_tl , bbox_target_data_br, labels, self._num_classes)
            
        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets_tl, bbox_targets_br, bbox_inside_weights, bbox_outside_weights , target_masks, mask_select, mask_labels

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_tl_data, bbox_target_br_data , labels_batch, num_classes):
        """

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets_tl = bbox_target_tl_data.new(batch_size, rois_per_image, 2).zero_()
        bbox_targets_br = bbox_target_br_data.new(batch_size, rois_per_image, 2).zero_()
        bbox_inside_weights = bbox_target_br_data.new(batch_size, rois_per_image, 4).zero_()

#         mask_clss = mask_labels_data
#         target_mask_select = bbox_target_br_data.new(batch_size, rois_per_image, 1).zero_()
#         target_mask_batch = target_mask_batch_data.new(batch_size, rois_per_image, self.mask_shape, self.mask_shape).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() > 0:
                inds = torch.nonzero(clss[b] > 0).view(-1)
                for i in range(inds.numel()):
                    ind = inds[i]
                    bbox_targets_tl[b, ind, :] = bbox_target_tl_data[b, ind, :]
                    bbox_targets_br[b, ind, :] = bbox_target_br_data[b, ind, :]
                    bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS
            
#             if mask_clss[b].sum() > 0:
#                 m_inds = torch.nonzero(mask_clss[b] > 0).view(-1)
#                 for i in range(m_inds.numel()):
#                     m_ind = m_inds[i]
#                     target_mask_batch[b, m_ind, :] = target_mask_batch_data[b, m_ind, :] 
#                     target_mask_select[b, m_ind]  =mask_select_data [b, m_ind]

        return bbox_targets_tl, bbox_targets_br, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois, ratios):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 7
        assert gt_rois.size(2) == 7

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets_tl, targets_br  = bbox_transform_batch(ex_rois, gt_rois, ratios)

       # if TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
        #    targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
         #               / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets_tl, targets_br


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, ratios):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        
        batch_size = overlaps.size(0)
        
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        labels = gt_boxes[:,:,5].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        
        #print(labels) 
        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 7).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 7).zero_()
        ratios_batch = torch.from_numpy(np.zeros((batch_size,rois_per_image, 2), dtype=np.float32))
        target_mask_batch = all_rois.new(batch_size, rois_per_image, self.mask_shape, self.mask_shape).zero_()
        
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):
            fg_inds = torch.nonzero(max_overlaps[i] >= self.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()
#             print(fg_num_rois, "FGRIS")
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < self.BG_THRESH_HI) &
                                    (max_overlaps[i] >= self.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
            elif fg_num_rois > 0 and bg_num_rois == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0
            
            rois_batch[i] = all_rois[i][keep_inds]        
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]
    
        bbox_target_data_tl , bbox_target_data_br  = self._compute_targets_pytorch(
                rois_batch, gt_rois_batch, ratios)

        return labels_batch, rois_batch, bbox_target_data_tl, bbox_target_data_br
    
    def _masks_assignment(self, rois, mask_gt_boxes,  gt_masks):
        
        rois = rois.clone().detach()
        
        overlaps = bbox_overlaps_batch(rois, mask_gt_boxes)
        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        batch_size, rois_per_image = rois.shape[:2]
#         print(max_overlaps, gt_assignment)
        offset = torch.arange(0, batch_size)*mask_gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        labels = mask_gt_boxes[:,:,5].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)       
        target_mask_batch = rois.new(batch_size, rois_per_image, self.mask_shape, self.mask_shape).zero_()
        mask_select = rois.new(batch_size, rois_per_image).zero_() 
        labels_batch = labels.new(batch_size, rois_per_image).zero_()

        for i in range(batch_size):
#             fg_inds =  (max_overlaps[i] >= 0.6).view(-1) * (max_overlaps[i] <= 0.8).view(-1)
            fg_inds =  (max_overlaps[i] >= 0.7).view(-1)
#             print(fg_inds)

#             print(fg_inds)
            if torch.sum(fg_inds) > 0:
                gt_rois_batch = torch.index_select(mask_gt_boxes[i][:,1:5], 0, gt_assignment[i])
                mask_select[i] = fg_inds
                norm_boxes = rois[i][:,1:5].clone().detach()
                norm_gt = gt_rois_batch.clone().detach()
#                 print("------",norm_boxes)
                norm_boxes[:,0:2] -= norm_gt[:,0:2]
                norm_boxes[:,2:4] -= norm_gt[:,0:2]
                dh = norm_gt[:,2:3]-norm_gt[:, 0:1]#+1e-10
                dw = norm_gt[:,3:4]-norm_gt[:, 1:2]#+1e-10
#                 print(gt_rois_batch)
                labels_batch[i].copy_(labels[i])
                roi_masks = torch.index_select(gt_masks[i], 0, gt_assignment[i])
                
#                 print(mask_select)
                norm_boxes = torch.cat([56*norm_boxes[:, 0:1]/dh,56*norm_boxes[:, 1:2]/dw ,56*norm_boxes[:, 2:3]/dh,56*norm_boxes[:, 3:4]/dw ] ,dim=1).clone().detach()
                norm_boxes = norm_boxes* fg_inds.unsqueeze(1).expand_as(norm_boxes).float()
                
#                 print(roi_masks.shape,norm_boxes.shape)
#                 save_image(roi_masks.float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_brforecrop.jpg",5)
#                 print(norm_boxes[fg_inds])
                masks = crop_and_resize(roi_masks  , norm_boxes, self.mask_shape).clone().detach()
                target_mask_batch[i] = masks
#                 save_image(masks.float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_aftercrop.jpg",5)
#             time.sleep(2)
            return target_mask_batch, mask_select, labels_batch

