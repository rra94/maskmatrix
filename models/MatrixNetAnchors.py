import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .py_utils.loss_utils import _regr_loss, _neg_loss
from torch.autograd import Variable
from .resnet_features import resnet152_features, resnet50_features, resnet18_features, resnet101_features, resnext101_32x8d, wide_resnet101_2
from .py_utils.utils import conv1x1, conv3x3
from  torchvision.utils import save_image
from .matrixnet import _sigmoid, MatrixNet, _gather_feat, _tranpose_and_gather_feat, _topk, _nms
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch, crop_and_resize
import pdb
import time

import torchvision.ops.roi_pool as roipool

from  torchvision.ops import nms as nms


class SubNet(nn.Module):

    def __init__(self, mode, classes=80, depth=4,
                 base_activation=F.relu,
                 output_activation=F.sigmoid):
        super(SubNet, self).__init__()
        self.classes = classes
        self.depth = depth
        self.base_activation = base_activation
        self.output_activation = output_activation

        self.subnet_base = nn.ModuleList([conv3x3(256, 256, padding=1)
                                          for _ in range(depth)])

        if mode == 'corners':
            self.subnet_output = conv3x3(256, 4, padding=1)
            
        if mode == 'tl_corners':
            self.subnet_output = conv3x3(256, 2, padding=1)
        if mode == 'br_corners':
            self.subnet_output = conv3x3(256, 2, padding=1)
            
        elif mode == 'classes':
            # add an extra dim for confidence
            self.subnet_output = conv3x3(256, self.classes, padding=1)

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
        self.subnet_anchors_heats = SubNet(mode='classes')

    def forward(self, x):
        features = self.matrix_net(x)
        anchors_tl_corners_regr = [self.subnet_tl_corners_regr(feature) for feature in features]
        anchors_br_corners_regr = [self.subnet_br_corners_regr(feature) for feature in features]
        anchors_heatmaps = [_sigmoid(self.subnet_anchors_heats(feature)) for feature in features]
        return  anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, features

class MaskLayer(nn.Module):
    def __init__(self, depth, num_classes):
        super(MaskLayer, self).__init__()
        self.depth = depth
#         self.pool_size = pool_size
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
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = conv3x3(256, 256, stride=1 ,padding =1)
        self.bn5 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv6 = conv1x1(256, num_classes, stride=1)        
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
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)        
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)        
        x = self.conv6(x)
        x = self.sigmoid(x)
        return x
        
class model(nn.Module):
    def __init__(self, db):
        super(model, self).__init__()
        classes = db.configs["categories"]
        resnet  = db.configs["backbone"]
        layers  = db.configs["layers_range"]
        self.base_layer_range  = db.configs["base_layer_range"]
        self.net = MatrixNetAnchors(classes, resnet, layers)
        self.Xmask = MaskLayer( 256, classes)

        self._decode = _decode
        
    def _train(self, *xs):

        image = xs[0][0]
        anchors_inds = xs[1]
        seg_inds = xs[2]
        seg_tag_masks = xs[3]
        
        outs = self.net.forward(image)

        all_pred_masks = [None for i in range(len(anchors_inds))]
        for ind in range(len(anchors_inds)):
            outs[1][ind] = _tranpose_and_gather_feat(outs[1][ind], anchors_inds[ind])
            outs[2][ind] = _tranpose_and_gather_feat(outs[2][ind], anchors_inds[ind])
            
            pred_masks = None
            
            with torch.no_grad():
                
                min_y, max_y, min_x, max_x = map(lambda x:int(x/8/2),self.base_layer_range)

                xc = seg_inds[ind] % outs[0][ind].shape[3]
                yc = seg_inds[ind] // outs[0][ind].shape[3]
                
#                 print(xcyc[ind].view(-1,2)[:5,:], xc.view(-1)[:5], yc.view(-1)[:3])

                x1 = (xc - max_x).unsqueeze(2)
                y1 = (yc - max_y).unsqueeze(2)
                x2 = (xc + max_x).unsqueeze(2)
                y2 = (yc + max_y).unsqueeze(2)             
                
                b_inds = torch.arange(xc.shape[0]).view(-1,1).repeat(1,xc.shape[1]).type_as(xc).unsqueeze(2)
                
#                 print(b_inds.view(-1))
#                 print(b_inds.shape, x1.shape)
                
                select_inds = seg_tag_masks[ind].view(-1)
        
                if torch.sum(select_inds) > 0:
                    dets = torch.cat([b_inds, x1,y1,x2,y2], dim =2).reshape(-1,5).float()[select_inds]
                    
                
                    
            if torch.sum(select_inds) > 0:
                pred_masks = roipool(outs[3][ind], dets, (2 * max_y + 1, 2 * max_x + 1))
            
            all_pred_masks[ind] = pred_masks
#         print(all_pred_masks[0].shape)    
        if all_pred_masks.count(None) != len(all_pred_masks):
            all_pred_masks = self.Xmask(torch.cat([i for i in all_pred_masks if i != None], dim = 0))
        else:
            all_pred_masks = None
            
                             
        return outs[0], outs[1], outs[2], all_pred_masks

    def _test(self, *xs, **kwargs):
        image = xs[0][0]
        
        outs = self.net.forward(image)
        return self._decode(*outs, self.Xmask, **kwargs)

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
        mask_loss = 0
        
        
        anchors_heats = outs[0]
        anchors_tl_corners_regrs = outs[1]
        anchors_br_corners_regrs = outs[2]
        pred_masks = outs[3]

        gt_anchors_heat = targets[0]
        gt_tl_corners_regr = targets[1]
        gt_br_corners_regr = targets[2]
        gt_mask = targets[3]
        
        gt_seg_masks = targets[4]
        gt_mask_labels = targets[5]
        gt_seg_tag_masks = targets[6]
        
        numf = 0
        numr = 0
        num= 0 
        num_positive_masks = 0
        for i in range(len(anchors_heats)):
                             
            floss, num = self.focal_loss([anchors_heats[i]], gt_anchors_heat[i])
            focal_loss += floss
            numf += num
#             print( gt_mask[i].shape)
            rloss, num = self.regr_loss(anchors_br_corners_regrs[i], gt_br_corners_regr[i], gt_mask[i])
            numr += num
            corner_regr_loss += rloss
            
            rloss, num = self.regr_loss(anchors_tl_corners_regrs[i], gt_tl_corners_regr[i], gt_mask[i])
            numr += num
         
            corner_regr_loss += rloss
                  
        if  pred_masks != None:
            gt_masks_all_layers = [None for i in range(len(anchors_heats))]
            gt_classes_all_layers = [None for i in range(len(anchors_heats))]

            for i in range(len(anchors_heats)):
                if torch.sum(gt_seg_tag_masks[i]>0) :
                    positive_ix = (gt_seg_tag_masks[i] > 0).view(-1)

                    target_masks = gt_seg_masks[i].view(-1,gt_seg_masks[i].shape[2],gt_seg_masks[i].shape[3])
                    gt_masks_all_layers[i] = target_masks[positive_ix]

                    gt_classes_all_layers[i] = gt_mask_labels[i].view(-1)[positive_ix]

            gt_masks_all_layers = torch.cat([i for i in gt_masks_all_layers if i != None], dim = 0)


            gt_classes_all_layers = torch.cat([i for i in gt_classes_all_layers if i != None], dim = 0).view(-1).long()
   
            y_onehot = torch.FloatTensor(pred_masks.shape[0], pred_masks.shape[1]).type_as(gt_classes_all_layers)
            y_onehot.zero_()
            y_onehot.scatter_(1, gt_classes_all_layers.view(-1,1), 1)
            y_onehot=torch.nonzero(y_onehot.view(-1))
            
            pred_masks = pred_masks.view(-1, pred_masks.shape[2],pred_masks.shape[3])[y_onehot,:,:].squeeze(1)
            
            save_image(gt_masks_all_layers.float().unsqueeze(1),  "./imgs/target+" + str(i) + "_after.jpg",5)
            save_image(pred_masks.float().unsqueeze(1),  "./imgs/pred+" + str(i) + "_after.jpg",5)

            mask_loss += F.binary_cross_entropy(pred_masks, gt_masks_all_layers)
            
            time.sleep(10)
            
        if numr > 0:
            corner_regr_loss = corner_regr_loss / numr
           
        if numf > 0:
            focal_loss = focal_loss / numf
            
        loss = (focal_loss + corner_regr_loss) + mask_loss
        return loss.unsqueeze(0)
    
loss = MatrixNetAnchorsLoss()

def _decode(
    anchors_heats, corners_tl_regrs, corners_br_regrs,features, Xmask,
    K=100, kernel=1, dist_threshold=0.2, num_dets=1000,layers_range = None,
    output_kernel_size = None, output_sizes = None, input_size=None, base_layer_range=None
):
    top_k = K
    batch, cat, height_0, width_0 = anchors_heats[0].size()
    min_y, max_y, min_x, max_x = map(lambda x:int(x/8/2), base_layer_range)
    
    detections_batch =[]
    predmask_batch = []
    
    for b_ind in range(batch):
        layer_dets =[]
        masks_dets_in_layer = []
        boxes_without_scaling_layer = []
        for i in range(len(anchors_heats)):
            anchors_heat = anchors_heats[i][b_ind]
            corners_tl_regr = corners_tl_regrs[i][b_ind]
            corners_br_regr = corners_br_regrs[i][b_ind]
#             print(anchors_heat.shape, corners_tl_regr.shape)
            
            cat, height, width = anchors_heat.size()
            height_scale = height_0 / height
            width_scale = width_0 / width
            
            K = min(K,height*width)
            anchors_scores, anchors_inds, anchors_clses, anchors_ys, anchors_xs = _topk(anchors_heat, K )
#             print(anchors_scores.shape)
            anchors_ys = anchors_ys.view(K, 1)
            anchors_xs = anchors_xs.view(K, 1)
        
            if corners_br_regr is not None:
                corners_tl_regr = _tranpose_and_gather_feat(corners_tl_regr, anchors_inds)
                corners_tl_regr = corners_tl_regr.view(K, 1, 2)
                corners_br_regr = _tranpose_and_gather_feat(corners_br_regr, anchors_inds)
                corners_br_regr = corners_br_regr.view(K, 1, 2)

                tl_xs = anchors_xs -  (((max_x - min_x) * corners_tl_regr[..., 0]) + (max_x + min_x)/2) 
                tl_ys = anchors_ys -  (((max_y - min_y) * corners_tl_regr[..., 1]) + (max_y + min_y)/2)
                br_xs = anchors_xs +  (((max_x - min_x) * corners_br_regr[..., 0]) + (max_x + min_x)/2)
                br_ys = anchors_ys +  (((max_y - min_y) * corners_br_regr[..., 1]) + (max_y + min_y)/2)
                
                tl_xs = torch.clamp(tl_xs, 0, width)
                tl_ys = torch.clamp(tl_ys, 0, height)
                br_xs = torch.clamp(br_xs, 0, width)
                br_ys = torch.clamp(br_ys, 0, height)
                
            bboxes = torch.cat((tl_xs, tl_ys, br_xs, br_ys), dim=1)

            scores    = anchors_scores.view(K, 1)
            
            mask_x1 = (anchors_xs - max_x)
            mask_y1 = (anchors_ys - max_y)
            mask_x2 = (anchors_xs + max_x)
            mask_y2 = (anchors_ys + max_y)

            layer_name  = torch.tensor([i]).repeat(mask_x1.shape[0],1).type_as(mask_x1)
            
            mask_bboxes =  torch.cat((mask_x1, mask_y1, mask_x2, mask_y2,layer_name), dim=1)
           
            width_inds  = (br_xs < tl_xs)
            height_inds = (br_ys < tl_ys)

            scores[width_inds]  = -1
            scores[height_inds] = -1
            scores = scores.view(-1)

            scores, inds = torch.topk(scores, min(num_dets, scores.shape[0]))
            scores = scores.unsqueeze(1)

            bboxes = bboxes.view(-1, 4)
            bboxes = _gather_feat(bboxes, inds)
            mask_bboxes =  _gather_feat(mask_bboxes, inds)
            
            layer_name = layer_name[inds]
            
            clses  = anchors_clses.contiguous().view(-1, 1)
            clses  = _gather_feat(clses, inds).float()
            
            boxes_without_scaling =  torch.cat([bboxes, scores,scores,layer_name, clses], dim=1)
           
            
            bboxes[:, 0] *= width_scale
            bboxes[:, 1] *= height_scale
            bboxes[:, 2] *= width_scale
            bboxes[:, 3] *= height_scale
            
            
            dets_in_layer = torch.cat([bboxes, scores,scores,layer_name, clses], dim=1)
            layer_dets.append(dets_in_layer)
            boxes_without_scaling_layer.append(boxes_without_scaling)
            masks_dets_in_layer.append(mask_bboxes)
                
        detections = torch.cat(layer_dets, dim = 0)
        mask_bboxes = torch.cat(masks_dets_in_layer, dim = 0)
        boxes_without_scaling  = torch.cat(boxes_without_scaling_layer, dim = 0)
        
#         top_scores, top_inds = torch.topk(detections[:, 4], 300)
                
        
#         mask_bboxes = _gather_feat(mask_bboxes, top_inds)
#         dets = _gather_feat(detections, top_inds)
#         boxes_without_scaling =  _gather_feat(boxes_without_scaling, top_inds)
        
        #nms
        keeps = nms(detections[:,1:5],detections[:, 4], iou_threshold=0.5)
        keeps=keeps[:300]
        dets =_gather_feat(detections, keeps)
        boxes_without_scaling =  _gather_feat(boxes_without_scaling, keeps)
        mask_bboxes = _gather_feat(mask_bboxes, keeps)

  
        _, mask_inds = torch.topk(mask_bboxes[:, -1], mask_bboxes.size(0))
        mask_bboxes = _gather_feat(mask_bboxes, mask_inds)
        dets = _gather_feat(dets, mask_inds)
        boxes_without_scaling = _gather_feat(boxes_without_scaling, mask_inds)
        detections_batch.append(dets)
        
        target_classes = dets[:, -1]

        
        all_pred_masks = []
        
        for i in range(len(features)-1, -1,-1):
            keeps = (mask_bboxes[:, -1] == i)
            pred_masks = roipool(features[i][b_ind][None,:,:,:], [mask_bboxes[keeps][:,0:4].float()], (2 * max_y + 1, 2 * max_x + 1))
            all_pred_masks.append( pred_masks)
        
        
        all_pred_masks = torch.cat([i for i in all_pred_masks if i != None], dim = 0)
        
                                   
        all_pred_masks = Xmask(all_pred_masks)

        y_onehot = torch.FloatTensor(all_pred_masks.shape[0], all_pred_masks.shape[1]).type_as(target_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, target_classes.view(-1,1).long(), 1)
        y_onehot=torch.nonzero(y_onehot.view(-1))  
#         pdb.set_trace()
        all_pred_masks = all_pred_masks.view(-1, all_pred_masks.shape[2],all_pred_masks.shape[3])[y_onehot,:,:].squeeze(1)
        _, h,w = all_pred_masks.shape              
        save_image(all_pred_masks.float().unsqueeze(1),  "./imgs/target+" + str(i) + "bc_predicted.jpg",5)
        
        
        boxes_without_scaling[:,0:2] -= mask_bboxes[:,0:2]
        boxes_without_scaling[:,2:4] -= mask_bboxes[:,0:2]

        norm_boxes = (36. - 1.) / (9. - 1.) * boxes_without_scaling[:,:4] # multiplying by the box upsampling ratios (2 deconvs)
        
        all_pred_masks = crop_and_resize (all_pred_masks, norm_boxes ,h)                                     
        save_image(all_pred_masks.float().unsqueeze(1),  "./imgs/target+" + str(i) + "ac_predicted.jpg",5)
        predmask_batch.append(all_pred_masks)
        
    predmask_batch = torch.cat(predmask_batch, dim =0)
    detections_batch = torch.cat(detections_batch, dim =0)
   
    predmask_batch = predmask_batch.view(batch, -1, predmask_batch.shape[-2], predmask_batch.shape[-1])
    detections_batch = detections_batch.view(batch, -1, detections_batch.shape[-1])
#     pdb.set_trace()
    return detections_batch,predmask_batch
                                                      
            
                                                      
                                                      
                                                      
       
                                                      
                                                      
                                                      
                                                      
                                                      



