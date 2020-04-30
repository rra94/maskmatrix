import math
import torch
import torch.nn as nn
import torchvision.ops.roi_align as roi_align
import torch.nn.functional as F
from .py_utils.loss_utils import _regr_loss, _neg_loss
from torch.autograd import Variable
from .resnet_features import resnet152_features, resnet50_features, resnet18_features, resnet101_features, resnext101_32x8d, wide_resnet101_2
from .py_utils.utils import conv1x1, conv3x3
from .matrixnet import _sigmoid, MatrixNet, _gather_feat, _tranpose_and_gather_feat, _topk, _nms
from .proposal_target_layer_cascade import _ProposalTargetLayer
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from sample.visualize import display_instances, display_images
from  torchvision.utils import save_image

class SubNet(nn.Module):

    def __init__(self, mode, depth=4,
                 base_activation=F.relu,
                 output_activation=F.sigmoid):
        super(SubNet, self).__init__()
        self.depth = depth
        self.base_activation = base_activation
        self.output_activation = output_activation

        self.subnet_base = nn.ModuleList([conv3x3(256, 256, padding=1)
                                          for _ in range(depth)])
        if mode == 'tl_corners':
            self.subnet_output = conv3x3(256, 2, padding=1)
        if mode == 'br_corners':
            self.subnet_output = conv3x3(256, 2, padding=1)    
        if mode == 'RPN': 
            self.subnet_output = conv3x3(256, 1, padding=1)

    def forward(self, x):
        for layer in self.subnet_base:
            x = self.base_activation(layer(x))
        x = self.subnet_output(x)
        return x   

    
class MatrixNetAnchors(nn.Module):
    def __init__(self, classes, resnet, layers):
        super(MatrixNetAnchors, self).__init__()
        self.classes = classes
        self.resnet = resnet
        if self.resnet == "resnext101_32x8d":
            _resnet = resnext101_32x8d(pretrained=True)
        elif self.resnet == "resnet101":
            _resnet = resnet101_features(pretrained =True)
        elif self.resnet == "resnet50":
            _resnet = resnet50_features(pretrained =True)
        elif self.resnet == "resnet152":
            _resnet = resnet152_features(pretrained =True)
        try: 
            self.matrix_net = MatrixNet(_resnet, layers)
        except : 
            print("ERROR: ivalid resnet")
            sys.exit()

        self.subnet_tl_corners_regr = SubNet(mode='tl_corners')
        self.subnet_br_corners_regr = SubNet(mode='br_corners')
        self.subnet_anchors_heats = SubNet(mode='RPN')

    def forward(self, x):
        features = self.matrix_net(x)
        anchors_tl_corners_regr = [self.subnet_tl_corners_regr(feature) for feature in features]
        anchors_br_corners_regr = [self.subnet_br_corners_regr(feature) for feature in features]
        anchors_heatmaps = [_sigmoid(self.subnet_anchors_heats(feature)) for feature in features]
        return features, anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr
    
# class SamePad2d(nn.Module):
#     """Mimics tensorflow's 'SAME' padding.
#     """
#     def __init__(self, kernel_size, stride):
#         super(SamePad2d, self).__init__()
#         self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
#         self.stride = torch.nn.modules.utils._pair(stride)

#     def forward(self, input):
#         in_width = input.size(2)
#         in_height = input.size(3)
#         out_width = math.ceil(float(in_width) / float(self.stride[0]))
#         out_height = math.ceil(float(in_height) / float(self.stride[1]))
#         pad_along_width = ((out_width - 1) * self.stride[0] +
#                            self.kernel_size[0] - in_width)
#         pad_along_height = ((out_height - 1) * self.stride[1] +
#                             self.kernel_size[1] - in_height)
#         pad_left = math.floor(pad_along_width / 2)
#         pad_top = math.floor(pad_along_height / 2)
#         pad_right = pad_along_width - pad_left
#         pad_bottom = pad_along_height - pad_top
#         return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

#     def __repr__(self):
#         return self.__class__.__name__

    
class MaskLayer(nn.Module):
    def __init__(self, depth, pool_size, num_classes ):
        super(MaskLayer, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.num_classes = num_classes
#         self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = conv3x3(self.depth, 256, stride=1, padding =1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = conv3x3(256, 256, stride=1,padding =1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = conv3x3(256, 256, stride=1 ,padding =1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = conv3x3(256, 256, stride=1,padding =1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = conv1x1(256, num_classes, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x ):
#         x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x
        

class model(nn.Module):
    def __init__(self, db):
        super(model, self).__init__()
        self.classes = db.configs["categories"]
        resnet  = db.configs["backbone"]
        layers  = db.configs["layers_range"]
        self.POOLING_SIZE = 7
        self.nchannels = 256
        self.MASK_SIZE = 28
        linearfiltersize = self.nchannels * (self.POOLING_SIZE-1)*(self.POOLING_SIZE-1)
        self.rpn = MatrixNetAnchors(self.classes, resnet, layers)
        self._decode = _decode
        self.proposals_generators = ProposalGenerator(db)
        self.proposal_target_layer = _ProposalTargetLayer(self.classes) #80 or 81
        self.RCNN_roi_align = RoIAlignMatrixNet(self.POOLING_SIZE, self.POOLING_SIZE)
        
        self.RCNN_head = nn.Sequential(nn.Linear(linearfiltersize, 1024),
                                             nn.ReLU(),
                                             nn.Linear(1024, 1024),
                                             nn.ReLU())
        self.RCNN_cls_score = nn.Sequential(nn.Linear(1024, 1024),
                                             nn.ReLU(),nn.Linear(1024, self.classes)) # 80 or 81
        self.RCNN_bbox_pred_tl = nn.Sequential(nn.Linear(1024, 1024),
                                             nn.ReLU(),nn.Linear(1024, 2))
        self.RCNN_bbox_pred_br = nn.Sequential(nn.Linear(1024, 1024),
                                             nn.ReLU(),nn.Linear(1024, 2))
        #80 classes
        self.RCNN_mask = MaskLayer( 256, self.MASK_SIZE, self.classes-1)

    def _train(self, *xs):
        image = xs[0][0]
        gt_rois = xs[2][0]
        anchors_inds = xs[1]
        ratios = xs[3][0]
        gt_masks = xs[4][0]
        mask_gt_rois = xs[5][0]
        
        features, anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr = self.rpn.forward(image)
        rois = self.proposals_generators(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr)
        
        rois, rois_label, bbox_targets_tl, bbox_targets_br , bbox_inside_weights, bbox_outside_weights , target_mask, mask_select, mask_labels =self.proposal_target_layer(rois, gt_rois, gt_masks, mask_gt_rois, ratios)
        
        #prints
#         print(gt_masks[0].unsqueeze(1).shape)
        
#         x = gt_masks[0].float()
#         save_image(x.unsqueeze(1), "/home/rragarwal4/matrixnet/imgs/gt.jpg",5)
#         save_image(target_mask[0][mask_select[0].bool()].float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target.jpg",32)
#         time.sleep(30)
        
#         '(torch.sum(target_mask[0]))
#         save_image(target_mask[0].float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target.jpg",32)

#         g = gt_masks[0].data.cpu().numpy()
#         t = target_mask[0].data.cpu().numpy()
# #         print(g)
        
#         display_images( [g[i].astype(int) for i in range(200)], "/home/rragarwal4/matrixnet/imgs/", str(time.time())+"gt.jpg" )
#         display_images( [t[i].astype(int) for i in range(200)], "/home/rragarwal4/matrixnet/imgs/", str(time.time())+"target.jpg" )

        

#         save_image(target_mask.view(-1,28,28).float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_masks_bs_before_select.jpg",5)
#         ps = (mask_select >0).view(-1)
#         if (torch.sum(ps) > 0):
#             save_image(target_mask.view(-1,28,28)[ps].float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_masks_bs_after_select.jpg",5)

        _, inds = torch.topk(rois[:,:, 6], rois.size(1))
        rois = _gather_feat(rois, inds)
        rois_label =  rois_label.gather(1, inds)
        
#         print (target_mask.shape)
        target_mask = _gather_feat(target_mask, inds)
        mask_select =  mask_select.gather(1, inds)
        mask_labels =  mask_labels.gather(1, inds)
        
        bbox_targets_tl = _gather_feat(bbox_targets_tl, inds)
        bbox_targets_br = _gather_feat(bbox_targets_br, inds)
        bbox_inside_weights = _gather_feat(bbox_inside_weights, inds)
        bbox_outside_weights = _gather_feat(bbox_outside_weights, inds)
        
# #         save_image(target_mask.view(-1,28,28).float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_masks_before.jpg",5)
# #         print(mask_select.view(-1))
        
# #         inds = inds.clone().detach()
        


#         save_image(target_mask.view(-1,28,28).float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_masks_before_select.jpg",5)
#         ps = (mask_select >0).view(-1)
#         if (torch.sum(ps) > 0):
#             save_image(target_mask.view(-1,28,28)[ps].float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_masks_after_select.jpg",5)
# #         print(mask_select.view(-1))
              
#         print(torch.sum(target_mask[0]))
#         save_image(target_mask[0].float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target.jpg",32)

#         time.sleep(5)

        pooled_masks, pooled_feat, batch_size, nroi,c, h, w = self.RCNN_roi_align(features,rois)
        pooled_feat = self.RCNN_head(pooled_feat)
        bbox_pred_tl = self.RCNN_bbox_pred_tl(pooled_feat)
        bbox_pred_tl = bbox_pred_tl.view(batch_size, nroi, 2)
        bbox_pred_br = self.RCNN_bbox_pred_br(pooled_feat)
        bbox_pred_br = bbox_pred_br.view(batch_size, nroi, 2)
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, dim = 1)
        cls_score = cls_score.view(batch_size, nroi, -1)
        cls_prob = cls_prob.view(batch_size, nroi, -1)
        
        pooled_masks = pooled_masks.view(batch_size*nroi, self.nchannels,self.POOLING_SIZE*2 ,self.POOLING_SIZE*2 )
        masks_preds = self.RCNN_mask(pooled_masks)
        masks_preds = masks_preds.view(batch_size, nroi,self.classes-1, self.MASK_SIZE ,self.MASK_SIZE )
        

        for ind in range(len(anchors_inds)):
            anchors_tl_corners_regr[ind] = _tranpose_and_gather_feat(anchors_tl_corners_regr[ind], anchors_inds[ind])
            anchors_br_corners_regr[ind] = _tranpose_and_gather_feat(anchors_br_corners_regr[ind], anchors_inds[ind])
        
        return anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, cls_score, cls_prob , bbox_pred_tl,bbox_pred_br,bbox_targets_tl, bbox_targets_br, rois_label, bbox_inside_weights, bbox_outside_weights,masks_preds , target_mask, mask_select, mask_labels, self.classes

    def _test(self, *xs, **kwargs):
        image = xs[0][0]
        features, anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr = self.rpn.forward(image)
        rois = self.proposals_generators(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr)

        rois[:,:,1:5] = rois[:,:,1:5] * 8
        
        _, inds = torch.topk(rois[:,:, 6], rois.size(1))
        
        rois = _gather_feat(rois, inds)

        
        pooled_masks, pooled_feat, batch_size, nroi,c, h, w = self.RCNN_roi_align(features,rois)
        
        pooled_feat = self.RCNN_head(pooled_feat)
        bbox_pred_tl = self.RCNN_bbox_pred_tl(pooled_feat)
        bbox_pred_tl = bbox_pred_tl.view(batch_size, nroi, 2)
        bbox_pred_br = self.RCNN_bbox_pred_br(pooled_feat)
        bbox_pred_br = bbox_pred_br.view(batch_size, nroi, 2)        
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, dim = 1)
        cls_score = cls_score.view(batch_size, nroi, -1)
        cls_prob = cls_prob.view(batch_size, nroi, -1)
        
        bboxes_decoded = self._decode(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, bbox_pred_tl,bbox_pred_br, cls_score ,cls_prob,  **kwargs)
#         print(bboxes_decoded[0][1:100][1:5], "ddddd")
        batch_size, prenms_nroi, _ = bboxes_decoded.shape

    
        bboxes_for_masks=bboxes_decoded.new(batch_size, prenms_nroi, 7).zero_()
# #         print(bboxes_decoded[0:,1:4,:])

        _, inds = torch.topk(bboxes_decoded[:,:, 6], bboxes_decoded.size(1))
        bboxes_decoded = _gather_feat(bboxes_decoded, inds)
        
        bboxes_for_masks[:,:,0:1] = bboxes_decoded[:,:,4:5] #score
        bboxes_for_masks[:,:,1:5] = bboxes_decoded[:,:,0:4]*8 #bbox cords
        bboxes_for_masks[:,:,5:6] = bboxes_decoded[:,:,7:8] #classes
        bboxes_for_masks[:,:,6:7] = bboxes_decoded[:,:,6:7] #layer
        
#         print("-----",bboxes_for_masks[0][1:100])
#         bboxes_for_masks_post_nms = bboxes_for_masks.new(batch_size, 100, 7 ).zero_()
        
#         bboxes_decoded_post_nms = bboxes_decoded.new(batch_size, 100, 8 ).zeros_()

#         for b_ind in batch_size:
#             keeps = ops.nms(bboxes_for_masks[bind][:,1:5], bboxes_for_masks[b_ind][:,5], iou_threshold=0.5)
#             keeps = keeps[:100]
#             bboxes_for_masks_post_nms[b_ind] = bboxes_for_masks[b_ind][keeps, :]
#             bboxes_decoded_post_nms[b_ind] = bboxes_decoded_post_nms[b_ind][keeps, :]
            
        pooled_masks, _, batch_size, nroi,c, h, w = self.RCNN_roi_align(features, bboxes_for_masks)       
            
#         pooled_masks, _, batch_size, nroi,c, h, w = self.RCNN_roi_align(features,bboxes_for_masks_post_nms)
#         pooled_masks, _, batch_size, nroi,c, h, w = self.RCNN_roi_align(features,bboxes_for_masks)
        
        pooled_masks = pooled_masks.view(batch_size*prenms_nroi, self.nchannels,self.POOLING_SIZE*2 ,self.POOLING_SIZE*2 )
        masks_preds = self.RCNN_mask(pooled_masks)
        masks_preds = masks_preds.view(batch_size, prenms_nroi,self.classes-1, self.MASK_SIZE ,self.MASK_SIZE )

#         masks_preds = masks_preds.view(batch_size, prenms_nroi,self.MASK_SIZE ,self.MASK_SIZE, self.classes-1 )      
#         mask_preds = mask_preds[:,:,:,bboxes_decoded_post_nms[:,:,5]]
#         masks_preds = masks_preds[:,:,:,bboxes_for_masks[:,:,5].long()]
        
#         print(bboxes_for_masks.shape)\\


        return bboxes_decoded, masks_preds
    
    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

    
class MatrixNetAnchorsLoss(nn.Module):
    def __init__(self, corner_regr_weight=1, center_regr_weight=0.1, focal_loss=_neg_loss):
        super(MatrixNetAnchorsLoss, self).__init__()
        self.corner_regr_weight = corner_regr_weight
        self.center_regr_weight = center_regr_weight
        self.focal_loss  = focal_loss
        self.regr_loss   = _regr_loss
        
    def forward(self, outs, targets):
        # focal loss
        focal_loss = 0
        corner_regr_loss = 0
        
        anchors_heats = outs[0]
        anchors_tl_corners_regrs = outs[1]
        anchors_br_corners_regrs = outs[2]

        gt_anchors_heat = targets[0]
        gt_tl_corners_regr = targets[1]
        gt_br_corners_regr = targets[2]
        gt_mask = targets[3]
        
        numf = 0
        numr = 0
        for i in range(len(anchors_heats)):
            floss, num = self.focal_loss([anchors_heats[i]], gt_anchors_heat[i])
            focal_loss += floss
            numf += num
            rloss, num = self.regr_loss(anchors_br_corners_regrs[i], gt_br_corners_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss
            
            rloss, num = self.regr_loss(anchors_tl_corners_regrs[i], gt_tl_corners_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss            

        if numr > 0:
            corner_regr_loss = corner_regr_loss / numr
           
        if numf > 0:
            focal_loss = focal_loss / numf
        
        #classification and prediction loss
        rois, cls_score, cls_prob , bbox_pred_tl,bbox_pred_br,bbox_targets_tl, bbox_targets_br, rois_label, bbox_inside_weights, bbox_outside_weights, masks_preds, target_mask, mask_select, mask_labels, nclasses = outs[3:]
        
        
#         save_image(target_mask.view(-1,28,28).float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_masks_before_select.jpg",5)
#         ps = (mask_select >0).view(-1)
#         if (torch.sum(ps) > 0):
#             save_image(target_mask.view(-1,28,28)[ps].float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_masks_after_select.jpg",5)
# #         print(mask_select.view(-1))
#         time.sleep(5)
        
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0 
        mrcnn_mask_loss = 0
        
        RCNN_loss_bbox = self._compute_RCNN_loss_bbox(bbox_pred_tl,bbox_pred_br, bbox_targets_tl, bbox_targets_br, bbox_inside_weights, bbox_outside_weights)
        rois_label = rois_label.flatten().long()
        RCNN_loss_cls = F.cross_entropy(cls_score.view(-1, nclasses), rois_label)
#         print(RCNN_loss_cls)
        mask_labels = mask_labels.flatten().long()
        mrcnn_mask_loss = self._compute_mrcnn_mask_loss(target_mask, mask_labels, masks_preds, mask_select)

        loss = focal_loss + corner_regr_loss + RCNN_loss_bbox +  RCNN_loss_cls
        if mrcnn_mask_loss > 0:
            loss += mrcnn_mask_loss
        return loss.unsqueeze(0)
    
    def _compute_RCNN_loss_bbox(self, bbox_pred_tl,bbox_pred_br, bbox_targets_tl, bbox_targets_br, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):        
        sigma_2 = sigma ** 2
        box_diff_tl = bbox_pred_tl - bbox_targets_tl
        box_diff_br =  bbox_pred_br - bbox_targets_br
        in_box_diff = bbox_inside_weights * torch.cat([box_diff_tl, box_diff_br], dim = 2)
        #print(in_box_diff.shape)
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box
        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)
            loss_box = loss_box.mean()
        return loss_box
    
    def _compute_mrcnn_mask_loss(self,target_masks, target_class_ids, pred_masks, mask_select):
        
        
        batch_size , nrois, nclasses, h, w = pred_masks.shape
#         print(pred_masks.shape, target_masks.shape)
        target_masks = target_masks.view(batch_size*nrois,h,w )
#         save_image(target_masks.float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_masks.jpg",5)
#         print(mask_select.view(-1))

#         print(mask_select.view(-1))        
        pred_masks = pred_masks. view(batch_size*nrois,nclasses, h,w  )
        if target_class_ids.size():
                positive_ix = (mask_select > 0).view(-1)
#                 print(positive_ix.shape)
                if torch.sum(positive_ix) > 0:
                    positive_class_ids = target_class_ids[positive_ix.clone().detach()].long().view(-1)-1

                    y_true = target_masks[positive_ix]
                    y_pred = pred_masks[positive_ix]
#                     print(y_true.shape)
#                     save_image(y_true.float().unsqueeze(1),  "/home/rragarwal4/matrixnet/imgs/target_after.jpg",5)
            
                    y_onehot = torch.FloatTensor(positive_class_ids.shape[0], nclasses).type_as(positive_class_ids)
                    y_onehot.zero_()

                    y_onehot.scatter_(1, positive_class_ids.view(-1,1), 1)
                    y_onehot=torch.nonzero(y_onehot.view(-1))
                    y_pred_final = y_pred.view(-1, h,w)[y_onehot,:,:]
                    y_pred_final = y_pred_final.squeeze(1)
                    loss = F.binary_cross_entropy(y_pred_final, y_true)
                else:
                    loss = 0
        else:
            loss = 0
#         time.sleep(4)    
        return loss
    
loss = MatrixNetAnchorsLoss()


class RoIAlignMatrixNet(nn.Module):
    def __init__ (self, aligned_height, aligned_width):
        super(RoIAlignMatrixNet, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)

    def forward(self, features, rois):
        batch_pooled_feats = []
        batch_pooled_masks = []
        batch_size,_,height_0, width_0 = features[0].size()
        for b in range(batch_size):
            pooled_feats = []
            pooled_masks = []
            for i in range(len(features)-1,-1,-1):
                keep_inds = (rois[b][:,6] == i)
                if (torch.sum(keep_inds) == 0):
                    continue
                roi = rois[b][keep_inds]
                rois_cords = self.resize_rois(roi[:,1:5], features[i],height_0, width_0)
                x =  roi_align(features[i][b:b+1], [rois_cords], output_size=(self.aligned_width, self.aligned_height))               
                x = F.avg_pool2d(x, kernel_size=2, stride=1)
                y =  roi_align(features[i][b:b+1], [rois_cords], output_size=(self.aligned_width*2, self.aligned_height*2))
                
                pooled_feats.append(x)
                pooled_masks.append(y)
                
            pooled_feats = torch.cat(pooled_feats, dim =0)
            pooled_feats = torch.unsqueeze(pooled_feats, dim = 0)
            batch_pooled_feats.append(pooled_feats)
            
            pooled_masks = torch.cat(pooled_masks, dim =0)
            pooled_feats = torch.unsqueeze(pooled_masks, dim = 0)
            batch_pooled_masks.append(pooled_masks)
            
        batch_pooled_feats = torch.cat(batch_pooled_feats, dim=0)
        batch_size , n_roi, c, h, w = batch_pooled_feats.size()
        batch_pooled_feats=batch_pooled_feats.view(batch_size*n_roi,-1)
        
        batch_pooled_masks = torch.cat(batch_pooled_masks, dim=0)
        batch_pooled_masks=batch_pooled_masks.view(batch_size*n_roi,-1)
        
        #print(batch_pooled_feats.size())
        return batch_pooled_masks, batch_pooled_feats, batch_size, n_roi, c ,h ,w

    def resize_rois(self, rois_cords, layer,height_0, width_0):        
        _, _,  height, width = layer.size()
        width_scale  =width/(width_0*8)
        height_scale = height/(height_0*8)
        for i in range(rois_cords.shape[1]):
            if i%2 == 0:
                rois_cords[:,i] *= width_scale
            else:
                rois_cords[:,i] *= height_scale
        return rois_cords
          
class ProposalGenerator(nn.Module):
    def __init__ (self, db):
        super(ProposalGenerator, self).__init__() 
        self.K = 5000
        self.num_dets = 10000
        self.base_layer_range = db.configs["base_layer_range"]
                
    def forward(self, anchors_heats, corners_tl_regrs, corners_br_regrs):
        top_k = self.K 
        batch, cat, height_0, width_0 = anchors_heats[0].size()
        layer_detections = [None for i in range(len(anchors_heats))]
       # with torch.no_grad(): 
        for i in range(len(anchors_heats)):
            anchors_heat = anchors_heats[i]
            corners_tl_regr = corners_tl_regrs[i]
            corners_br_regr = corners_br_regrs[i]
            batch, cat, height, width = anchors_heat.size()
            height_scale = height_0 / height
            width_scale = width_0 / width
            K = min(top_k, width * height)
            anchors_scores, anchors_inds, anchors_clses, anchors_ys, anchors_xs = _topk(anchors_heat, K=K)
            anchors_ys = anchors_ys.view(batch, K, 1)
            anchors_xs = anchors_xs.view(batch, K, 1)

            if corners_br_regr is not None:
                corners_tl_regr = _tranpose_and_gather_feat(corners_tl_regr, anchors_inds)
                corners_tl_regr = corners_tl_regr.view(batch, K, 1, 2)
                corners_br_regr = _tranpose_and_gather_feat(corners_br_regr, anchors_inds)
                corners_br_regr = corners_br_regr.view(batch, K, 1, 2)
                min_y, max_y, min_x, max_x = map(lambda x:x/8/2,self.base_layer_range) #This is the range of object sizes within the layers 
            # We devide by 2 since we want to compute the distances from center to corners.
                tl_xs = anchors_xs -  (((max_x - min_x) * corners_tl_regr[..., 0]) + (max_x + min_x)/2) 
                tl_ys = anchors_ys -  (((max_y - min_y) * corners_tl_regr[..., 1]) + (max_y + min_y)/2)
                br_xs = anchors_xs +  (((max_x - min_x) * corners_br_regr[..., 0]) + (max_x + min_x)/2)
                br_ys = anchors_ys +  (((max_y - min_y) * corners_br_regr[..., 1]) + (max_y + min_y)/2)
            bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
            scores    = anchors_scores.view(batch, K, 1)
            width_inds  = (br_xs < tl_xs)
            height_inds = (br_ys < tl_ys)
            scores[width_inds]  = -1
            scores[height_inds] = -1
            bboxes = bboxes.view(batch, -1, 4)
            bboxes[:, :, 0] *= width_scale
            bboxes[:, :, 1] *= height_scale
            bboxes[:, :, 2] *= width_scale
            bboxes[:, :, 3] *= height_scale 
            with torch.no_grad():
                layer_detections[i] = torch.cat([scores , bboxes], dim =2).data
                rights= torch.tensor([[[0,i]]*layer_detections[i].size(1)]*batch, dtype=torch.float, device = bboxes.device)
                layer_detections[i] = torch.cat([layer_detections[i] , rights],  dim =2)
        detections = torch.cat(layer_detections, dim = 1 )

        top_scores, top_inds = torch.topk(detections[:, :, 0],min(self.num_dets, detections.shape[1] ))
        detections = _gather_feat(detections, top_inds)
        return detections
        
    def  backward(self,top,propagate_down,bottom):
        pass

def _decode(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, bbox_pred_tl,bbox_pred_br, cls_score, cls_prob, 
   K=2000, kernel=1,layers_range = None,dist_threshold=0.2,
        output_kernel_size = None, output_sizes = None, input_size=None, base_layer_range=None
        ):
        dets = rois.clone().detach()
        dets[:,:,1:5] = dets[:,:,1:5]/8 
        ratios =[]
        k=0
        for i,l in enumerate(layers_range):
            for j,e in enumerate(l):
                if e !=-1:
                    ratios.append( [k,1/(2**(j))/(base_layer_range[2]/8), 1/(2**(i))/(base_layer_range[0]/8)])
                    k+=1
        
        ratios= torch.from_numpy(np.array(ratios)).type_as(dets)
        layers = dets[:,:,6].clone().detach().long()
        batches, nrois, _ = dets.shape
        layers_h = layers.new(batches, nrois, 3).zero_().float()
        for b in range(batches):
            layers_h[b] = torch.index_select(ratios, 0, layers[b].view(-1))

        targets_tl = torch.cat([bbox_pred_tl[:,:,0:1] / layers_h[:,:,1:2] , bbox_pred_tl[:,:,1:2] / layers_h[:,:,2:3] ] ,dim=2)
        targets_br = torch.cat([bbox_pred_br[:,:,0:1] / layers_h[:,:,1:2],  bbox_pred_br[:,:,1:2] / layers_h[:,:,2:3] ] ,dim=2)
        
        dets[:,:,1:2] =  dets[:,:,1:2] - targets_tl[:,:,0:1]
        dets[:,:,2:3] =  dets[:,:,2:3] - targets_tl[:,:,1:2]
        dets[:,:,3:4] =  dets[:,:,3:4] - targets_br[:,:,0:1]
        dets[:,:,4:5] =  dets[:,:,4:5] - targets_br[:,:,1:2]
        
        
        batch, rois, classes = cls_prob.size()
        cls_prob = cls_prob[:,:, 1:]
        batch, rois, classes = cls_prob.size()
        topk_scores, topk_inds = torch.topk(cls_prob.data.contiguous().view(batch, -1 ),  K)
        topk_clses = topk_inds % (classes)
        topk_inds = (topk_inds / (classes)).long()
        inds = topk_inds.unsqueeze(2).expand(batch, K, dets.size(2))
        dets = torch.gather(dets, 1, inds)
        dets[:,:,0] =  torch.sqrt(topk_scores[:,:] * dets[:,:,0])
        dets[:,:,5] = topk_clses
        dets_return = torch.cat([dets[:,:,1:5], dets[:,:, 0:1], dets[:,:, 0:1], dets[:,:,6:7], dets[:,:,5:6]], dim =2)
        return dets_return

