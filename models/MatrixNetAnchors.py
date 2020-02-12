import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .py_utils.loss_utils import _regr_loss, _neg_loss
from torch.autograd import Variable
from .resnet_features import resnet152_features, resnet50_features, resnet18_features, resnet101_features, resnext101_32x8d, wide_resnet101_2
from .py_utils.utils import conv1x1, conv3x3
from .matrixnet import _sigmoid, MatrixNet, _gather_feat, _tranpose_and_gather_feat, _topk, _nms
from .proposal_target_layer_cascade import _ProposalTargetLayer
import numpy as np
import torchvision.ops.roi_align as roi_align
import matplotlib.pyplot as plt
import time
import cv2

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
    
    
class model(nn.Module):
    def __init__(self, db):
        super(model, self).__init__()
        self.classes = db.configs["categories"]
        resnet  = db.configs["backbone"]
        layers  = db.configs["layers_range"]
        POOLING_SIZE = 7
        nchannels = 256
        linearfiltersize = nchannels * (POOLING_SIZE-1)*(POOLING_SIZE-1)
        self.rpn = MatrixNetAnchors(self.classes, resnet, layers)
        self._decode = _decode
        self.proposals_generators = ProposalGenerator(db)
        self.proposal_target_layer = _ProposalTargetLayer(self.classes) #80 or 81
        self.RCNN_roi_align = RoIAlignMatrixNet(POOLING_SIZE, POOLING_SIZE)
        
        self.RCNN_cls_score = nn.Sequential(nn.Linear(linearfiltersize, 1024),
                                             nn.ReLU(),
                                             nn.Linear(1024, self.classes)) # 80 or 81
        self.RCNN_bbox_pred_tl = nn.Sequential(nn.Linear(linearfiltersize, 1024),
                                             nn.ReLU(),
                                             nn.Linear(1024, 2))
        self.RCNN_bbox_pred_br = nn.Sequential(nn.Linear(linearfiltersize, 1024),
                                             nn.ReLU(),
                                             nn.Linear(1024, 2))

    def _train(self, *xs):
        image = xs[0][0]
        gt_rois = xs[2][0]
        anchors_inds = xs[1]
        ratios = xs[3][0]
        features, anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr = self.rpn.forward(image)
        rois = self.proposals_generators(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr)
        rois, rois_label, bbox_targets_tl, bbox_targets_br , bbox_inside_weights, bbox_outside_weights =self.proposal_target_layer(rois, gt_rois, ratios)
        
        
#         fimage = image[0] * 255
#         fimage = fimage.clone().cpu().detach().numpy()
#         fimage = fimage.transpose((1, 2, 0)).astype(np.uint8).copy() 
# #         print(rois[0].shape)
#         for i,bbox in enumerate(rois[0]):     
#             cat_size  = cv2.getTextSize("     ", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
#             color     = np.random.random((3, )) * 0.6 + 0.4
#             color     = color * 255
#             color     = color.astype(np.int32).tolist()
            
#             bbox  = bbox[1:7].clone().cpu().detach().numpy().astype(np.int32)
#             if rois_label[0,i].item() == 0:
#                    continue
#             if bbox[1] - cat_size[1] - 2 < 0:
#                 cv2.rectangle(fimage,
#                     (bbox[0], bbox[1] + 2),
#                     (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
#                     color, -1
#                 )
#                 cv2.putText(fimage, str(rois_label[0,i].item()) + "l"+ str(bbox[-1]),
#                     (bbox[0], bbox[1] + cat_size[1] + 2), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
#                 )
#             else:
#                 cv2.rectangle(fimage, 
#                     (bbox[0], bbox[1] - cat_size[1] - 2),
#                     (bbox[0] + cat_size[0], bbox[1] - 2),
#                     color, -1
#                 )
#                 cv2.putText(fimage, str(rois_label[0,i].item()) + "l"+ str(bbox[-1]),
#                     (bbox[0], bbox[1] - 2), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
#                 )
#             cv2.rectangle(fimage,
#                 (bbox[0], bbox[1]),
#                 (bbox[2], bbox[3]),
#                 color, 2
#             )          
#         debug_file = "/h/arashwan/maskmatrix/rois/"+ str(time.time()) +".jpg"
#         cv2.imwrite(debug_file,fimage)

        _, inds = torch.topk(rois[:,:, 6], rois.size(1))
        rois = _gather_feat(rois, inds)
        rois_label =  rois_label.gather(1, inds)
        bbox_targets_tl = _gather_feat(bbox_targets_tl, inds)
        bbox_targets_br = _gather_feat(bbox_targets_br, inds)
        bbox_inside_weights = _gather_feat(bbox_inside_weights, inds)
        bbox_outside_weights = _gather_feat(bbox_outside_weights, inds)
        
        pooled_feat, batch_size, nroi,c, h, w = self.RCNN_roi_align(features,rois)
        bbox_pred_tl = self.RCNN_bbox_pred_tl(pooled_feat)
        bbox_pred_tl = bbox_pred_tl.view(batch_size, nroi, 2)
        
        bbox_pred_br = self.RCNN_bbox_pred_br(pooled_feat)
        bbox_pred_br = bbox_pred_br.view(batch_size, nroi, 2)
        
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, dim = 1)
        cls_score = cls_score.view(batch_size, nroi, -1)
        cls_prob = cls_prob.view(batch_size, nroi, -1)
        
        for ind in range(len(anchors_inds)):
            anchors_tl_corners_regr[ind] = _tranpose_and_gather_feat(anchors_tl_corners_regr[ind], anchors_inds[ind])
            anchors_br_corners_regr[ind] = _tranpose_and_gather_feat(anchors_br_corners_regr[ind], anchors_inds[ind])

        return anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, cls_score, cls_prob , bbox_pred_tl,bbox_pred_br,bbox_targets_tl, bbox_targets_br, rois_label, bbox_inside_weights, bbox_outside_weights, self.classes

    def _test(self, *xs, **kwargs):
        image = xs[0][0]
        features, anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr = self.rpn.forward(image)
        rois = self.proposals_generators(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr)
        rois[:,:,1:5] = rois[:,:,1:5] * 8
        
        _, inds = torch.topk(rois[:,:, 6], rois.size(1))
        rois = _gather_feat(rois, inds)
        
        #rois, rois_label, bbox_targets, bbox_inside_weights, bbox_outside_weights =self.proposal_target_layer(rois, gt_rois)
        pooled_feat, batch_size, nroi,c, h, w = self.RCNN_roi_align(features,rois)

        
        bbox_pred_tl = self.RCNN_bbox_pred_tl(pooled_feat)
        bbox_pred_tl = bbox_pred_tl.view(batch_size, nroi, 2)
        
        bbox_pred_br = self.RCNN_bbox_pred_br(pooled_feat)
        bbox_pred_br = bbox_pred_br.view(batch_size, nroi, 2)
        
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, dim = 1)
        cls_score = cls_score.view(batch_size, nroi, -1)
        cls_prob = cls_prob.view(batch_size, nroi, -1)
        decoded = self._decode(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, bbox_pred_tl,bbox_pred_br, cls_score ,cls_prob,  **kwargs)
        return decoded
    
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
        rois, cls_score, cls_prob, bbox_pred_tl,bbox_pred_br, bbox_targets_tl, bbox_targets_br, rois_label, bbox_inside_weights, bbox_outside_weights, nclasses = outs[3:]
        RCNN_loss_cls = 0
        RCNN_loss_bbox = self._smooth_l1_loss(bbox_pred_tl,bbox_pred_br, bbox_targets_tl, bbox_targets_br, bbox_inside_weights, bbox_outside_weights)
        print(RCNN_loss_bbox)
        RCNN_loss_cls = F.cross_entropy(cls_score.view(-1, nclasses), rois_label.flatten().long())
        loss = (focal_loss + corner_regr_loss) + RCNN_loss_bbox +  RCNN_loss_cls
        return loss.unsqueeze(0)
    
    def _smooth_l1_loss(self, bbox_pred_tl,bbox_pred_br, bbox_targets_tl, bbox_targets_br, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):        
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
    
loss = MatrixNetAnchorsLoss()


class RoIAlignMatrixNet(nn.Module):
    def __init__ (self, aligned_height, aligned_width):
        super(RoIAlignMatrixNet, self).__init__()
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)

    def forward(self, features, rois):
        batch_pooled_feats=[]
        batch_size,_,height_0, width_0 = features[0].size()
        for b in range(batch_size):
            pooled_feats = []
            for i in range(len(features)-1,-1,-1):
                keep_inds = (rois[b][:,6] == i)
                if (torch.sum(keep_inds) == 0):
                    continue
                roi = rois[b][keep_inds]
                rois_cords = self.resize_rois(roi[:,1:5], features[i],height_0, width_0)
#                 #print(rois_cords.shape) caused illegal memory error. converting to list seems to work -1/30
                x =  roi_align(features[i][b:b+1], [rois_cords], output_size=(self.aligned_width, self.aligned_height))               
                x = F.avg_pool2d(x, kernel_size=2, stride=1)
                pooled_feats.append(x)

            pooled_feats = torch.cat(pooled_feats, dim =0)
            pooled_feats = torch.unsqueeze(pooled_feats, dim = 0)
            batch_pooled_feats.append(pooled_feats)
        batch_pooled_feats = torch.cat(batch_pooled_feats, dim=0)
        batch_size , n_roi, c, h, w = batch_pooled_feats.size()
        batch_pooled_feats=batch_pooled_feats.view(batch_size*n_roi,-1)
        #print(batch_pooled_feats.size())
        return batch_pooled_feats, batch_size, n_roi, c ,h ,w

    def resize_rois(self, rois_cords, layer,height_0, width_0):        
        _, _,  height, width = layer.size()
        width_scale  =width/(width_0*8)
        height_scale = height/(height_0*8)
        rois_cords[:,0] *= width_scale
        rois_cords[:,2] *= width_scale
        rois_cords[:,1] *= height_scale
        rois_cords[:,3] *= height_scale
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
                    ratios.append( [k,1/(2**(j)), 1/(2**(i))])
                    k+=1
        
        ratios= torch.from_numpy(np.array(ratios)).type_as(dets)
        layers = dets[:,:,6].clone().detach().long()
        batches, nrois, _ = dets.shape
        layers_h = layers.new(batches, nrois, 3).zero_().float()
        for b in range(batches):
            layers_h[b] = torch.index_select(ratios, 0, layers[b].view(-1))

        targets_tl = torch.cat([bbox_pred_tl[:,:,0:1] / layers_h[:,:,1:2] , bbox_pred_tl[:,:,1:2] / layers_h[:,:,2:3] ] ,dim=2)
        targets_br = torch.cat([bbox_pred_br[:,:,0:1] / layers_h[:,:,1:2],  bbox_pred_br[:,:,1:2] / layers_h[:,:,2:3] ] ,dim=2)
        
        dets[:,:,1:2] =  dets[:,:,1:2] + targets_tl[:,:,0:1]
        dets[:,:,2:3] =  dets[:,:,2:3] + targets_tl[:,:,1:2]
        dets[:,:,3:4] =  dets[:,:,3:4] + targets_br[:,:,0:1]
        dets[:,:,4:5] =  dets[:,:,4:5] + targets_br[:,:,1:2]
        
        
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
        dets_return = torch.cat([dets[:,:,1:5], dets[:,:, 0:1], dets[:,:, 0:1], dets[:,:,0:1], dets[:,:,5:6]], dim =2)
        return dets_return