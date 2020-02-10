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
        
        self.RCNN_cls_score = nn.Linear(linearfiltersize, self.classes) # 80 or 81
        self.RCNN_bbox_pred = nn.Linear(linearfiltersize, 4 )

    def _train(self, *xs):
        image = xs[0][0]
        gt_rois = xs[2][0]
        anchors_inds = xs[1]
        features, anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr = self.rpn.forward(image)
        rois = self.proposals_generators(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr)
        rois, rois_label, bbox_targets, bbox_inside_weights, bbox_outside_weights =self.proposal_target_layer(rois, gt_rois)
        pooled_feat, batch_size, nroi,c, h, w = self.RCNN_roi_align(features,rois)
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        bbox_pred = bbox_pred.view(batch_size, nroi, 4)        
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.sigmoid(cls_score)
        cls_score = cls_score.view(batch_size, nroi, -1)
        cls_prob = cls_prob.view(batch_size, nroi, -1)
        for ind in range(len(anchors_inds)):
            anchors_tl_corners_regr[ind] = _tranpose_and_gather_feat(anchors_tl_corners_regr[ind], anchors_inds[ind])
            anchors_br_corners_regr[ind] = _tranpose_and_gather_feat(anchors_br_corners_regr[ind], anchors_inds[ind])

        return anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, cls_score, cls_prob , bbox_pred,bbox_targets, rois_label, bbox_inside_weights, bbox_outside_weights, self.classes

    def _test(self, *xs, **kwargs):
        image = xs[0][0]
        #gt_rois = xs[2][0]
        features, anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr = self.rpn.forward(image)
        rois = self.proposals_generators(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr)
        #print(rois.shape)
        rois[:,:,1:5] = rois[:,:,1:5] * 8
        #rois, rois_label, bbox_targets, bbox_inside_weights, bbox_outside_weights =self.proposal_target_layer(rois, gt_rois)
        pooled_feat, batch_size, nroi,c, h, w = self.RCNN_roi_align(features,rois)
        
        #print(rois.size(), "fgsgrs")
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        bbox_pred = bbox_pred.view(batch_size, nroi, 4)
        cls_score = self.RCNN_cls_score(pooled_feat)
        #cls_prob = F.softmax(cls_score, 1)
        cls_prob = F.sigmoid(cls_score)
        cls_score = cls_score.view(batch_size, nroi, -1)
        cls_prob = cls_prob.view(batch_size, nroi, -1)
        decoded = self._decode(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, cls_score ,cls_prob,  **kwargs)
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
        #print(len(outs))
        
        anchors_heats = outs[0]
        anchors_tl_corners_regrs = outs[1]
        anchors_br_corners_regrs = outs[2]
        #print(anchors_heats[1].shape)

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
        rois, cls_score, cls_prob, bbox_pred, bbox_targets, rois_label, bbox_inside_weights, bbox_outside_weights, nclasses = outs[3:]
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        #print(bbox_inside_weights.shape)
        #print(rois_label.shape, cls_score.shape)
        
        #RCNN_loss_cls = F.cross_entropy(cls_score.view(-1, nclasses), rois_label.flatten().long())
        y_onehot = torch.FloatTensor(rois_label.shape[0] * rois_label.shape[1], nclasses).cuda()
        y_onehot.zero_()
        y_onehot.scatter_(1, rois_label.view(-1, 1).long(), 1)
        RCNN_loss_cls = F.binary_cross_entropy(cls_prob.view(-1, nclasses), y_onehot)
        #RCNN_loss_cls = F.binary_cross_entropy(cls_score, rois_label)
        #RCNN_loss_bbox = self._smooth_l1_loss(bbox_pred,  torch.reshape(bbox_targets, bbox_pred.size()), bbox_inside_weights.view(-1,4), bbox_outside_weights.view(-1,4)) 
        #print("cls loss: {}".format(RCNN_loss_cls.item()))
        loss = (focal_loss + corner_regr_loss) + RCNN_loss_bbox +  RCNN_loss_cls
        #(RCNN_loss_cls, focal_loss,  corner_regr_loss )
        return loss.unsqueeze(0)
    #change to get bbox loss 
    def _smooth_l1_loss(self,bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):        
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
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
        
        #batch_size = rois.size(0)
        batch_size,_,height_0, width_0 = features[0].size()
        #print(rois.size(), "roi")
        for b in range(batch_size):
            pooled_feats = []
            pooled_inds = []
            #print(rois)
            for i in range(len(features)):
                #print(features[i].shape, rois[b].shape)
                #print(rois[b][:,6])
                keep_inds = (rois[b][:,6] == i)
                
                #:w
                #print(keep_inds.device)
                #Print(keep_inds)
                if (torch.sum(keep_inds) == 0):
                    continue
                inds = keep_inds.nonzero() 
                roi = rois[b][keep_inds]
                #print(roi.shape)
                rois_cords = self.resize_rois(roi[:,1:5], features[i],height_0, width_0)
                #print (features[i].shape)
                #print(rois_cords.shape) caused illegal memory error. converting to list seems to work -1/30
                x =  roi_align(features[i][b:b+1], [rois_cords], output_size=(self.aligned_width, self.aligned_height))
                x = F.avg_pool2d(x, kernel_size=2, stride=1)
                pooled_feats.append(x)
                pooled_inds.append(inds)
                #print(x.size())
                #print(pooled_feats[0].size())
            pooled_feats = torch.cat(pooled_feats, dim =0)
            pooled_inds = torch.cat(pooled_inds, dim =0)
            #print(pooled_inds.size(),pooled_feats.size() )
            #pooled_inds = pooled_inds.squeeze().expand(pooled_feats.size())
            #print(pooled_inds.size(),pooled_feats.size() )
            pooled_feats[:,:,:,:] = pooled_feats[pooled_inds.squeeze(),:,:,:].clone().detach()
            pooled_feats = torch.unsqueeze(pooled_feats, dim = 0)
            batch_pooled_feats.append(pooled_feats)
        #print(batch_pooled_feats[0].size())
        batch_pooled_feats = torch.cat(batch_pooled_feats, dim=0)
        #print(batch_pooled_feats.size())
        #batch_pooled_feats.squeeze_(0)
        #print(batch_pooled_feats.size())
        batch_size , n_roi, c, h, w = batch_pooled_feats.size()
        batch_pooled_feats=batch_pooled_feats.view(batch_size*n_roi,-1)
        #print(batch_pooled_feats.size())
        return batch_pooled_feats, batch_size, n_roi, c ,h ,w

    def resize_rois(self, rois_cords, layer,height_0, width_0):
        
        _, _,  height, width = layer.size()
        width_scale  =width/(width_0*8)
        height_scale = height/(height_0*8)
        #print(rois_cords)
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
        #print(anchors_heats[0].size(), "anchorsize)")
        top_k = self.K 
        batch, cat, height_0, width_0 = anchors_heats[0].size()
        #print(batch, "hdrhdrhrsddr")
        layer_detections = [None for i in range(len(anchors_heats))]
       # with torch.no_grad(): 
        for i in range(len(anchors_heats)):
            anchors_heat = anchors_heats[i]
            corners_tl_regr = corners_tl_regrs[i]
            corners_br_regr = corners_br_regrs[i]
            batch, cat, height, width = anchors_heat.size()
            height_scale = height_0 / height
            width_scale = width_0 / width
            #print(height_scale, width_scale)
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
            #scores = scores.view(batch, -1)
            #print(scores.shape , "score")
            #scores, inds = torch.topk(scores, min(self.num_dets, scores.shape[1]))
            #scores = scores.unsqueeze(2)
            bboxes = bboxes.view(batch, -1, 4)
            #bboxes = _gather_feat(bboxes, inds)
            #clses  = anchors_clses.contiguous().view(batch, -1, 1)
            #clses  = _gather_feat(clses, inds).float()
            bboxes[:, :, 0] *= width_scale
            bboxes[:, :, 1] *= height_scale
            bboxes[:, :, 2] *= width_scale
            bboxes[:, :, 3] *= height_scale
            #print(bboxes.shape, scores.shape, "yo")
            #if i == 0
            #print(scores) 
            with torch.no_grad():
                #layer_detections[i] = torch.tensor(np.zeros((batch,bboxes.size(1), 7)) ,dtype = torch.float, requires_grad = False, device = bboxes.device) 
                layer_detections[i] = torch.cat([scores , bboxes], dim =2).data
                rights= torch.tensor([[[0,i]]*layer_detections[i].size(1)]*batch, dtype=torch.float, device = bboxes.device)
                #print(rights)
                #print(rights.size(), layer_detections[i].size())
                layer_detections[i] = torch.cat([layer_detections[i] , rights],  dim =2)
                #layer_detections[i].*requires_grad  = False
                #layer_detections[i][:,5] = 0
                #layer_detections[i][ :, 6] = i  #fix
                 #fic
                #print(layer_detections[i].size())
               
                #layer_detections[i][] = bboxe

                #else:
            #    detections = torch.cat([detections, torch.cat([scores, bboxes,i], dim=2)], dim = 1)
        detections = torch.cat(layer_detections, dim = 1 )
    #print(detections.shape
        #print(detections.shape, "dets.shape")
        #print(detections.shape)
        top_scores, top_inds = torch.topk(detections[:, :, 0],min(self.num_dets, detections.shape[1] ))
        detections = _gather_feat(detections, top_inds)
        #print(detections.shape, "yp")
        return detections
        
    def  backward(self,top,propagate_down,bottom):
        pass



def _decode(anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, cls_score, cls_prob, 
   K=2000, kernel=1,layers_range = None,dist_threshold=0.2,
        output_kernel_size = None, output_sizes = None, input_size=None, base_layer_range=None
        ):
        dets = rois.data
        dets[:,:,1:5] = dets[:,:,1:5]/8 
        batch, rois, classes = cls_prob.size()
        #print(batch)
        cls_prob = cls_prob[:,:, 1:]
        #print ("class_props: ", cls_prob[0,:5, 1:classes]*10000)
        #print(dets[:,:,1:5])
        #print(cls_prob.shape, "ddd")
        batch, rois, classes = cls_prob.size()
        #print(cls_prob.data.view(batch, rois, 1 ).size(), "tsffdsfds")
        topk_scores, topk_inds = torch.topk(cls_prob.data.contiguous().view(batch, -1 ),  K)
        print(topk_scores)
        #print("topk inds: ", topk_inds[0,:100])
        topk_clses = topk_inds % (classes)
        print(topk_clses[0,:100])
        topk_inds = (topk_inds / (classes)).long()
        #tester= topk_inds[0][1]
        #print(dets[0,tester, :])
        inds = topk_inds.unsqueeze(2).expand(batch, K, dets.size(2))
        #print(topk_scores[0,0])
        #print(cls_prob[0,topk_inds[0,0],topk_clses[0,0]])
        #print(dets[0,topk_inds[0,0],:])
        
        #print(topk_scores, "ghdghrdsgsrgse")
        dets = torch.gather(dets, 1, inds)
        #print(dets[0,0,:])
        #print('---')
        #print(dets[0,1,:])
        #tops = torch.cat([topk_inds, topk_clses], dim =1) 
        #topk_clses = topk_inds % (classes)
        #print(topk_inds.shape,topk_clses.shape, "hdghdh")
        #print(dets.shape)
        #print(topk_inds, topk_clses)
        #$dets = dets[:
        #print(dets.shape)
        #print(dets.shape)
        #clses  = cls_prob.contiguous().view(batch, -1, 1)
        #clses  = _gather_feat(clses,topk_inds).float()   
        #print(clses.shape, "fjfjf")
        #print(dets.shape, "sffs")
        dets[:,:,0] =  topk_scores[:,:] #* dets[:,:,0]
        #print ("Top scores: ",topk_scores[:1,:100])
        #print( "Top classes: ", topk_clses[:1,:100])
        #dets [:,:, 1:5] = dets[:,:,1:5] 
        dets[:,:,5] = topk_clses
        dets_return = torch.cat([dets[:,:,1:5], dets[:,:, 0:1], dets[:,:, 0:1], dets[:,:,0:1], dets[:,:,5:6]], dim =2)
        #print(dets_return[:,:, 0:4])
        return dets_return
                
def _decode_(
   anchors_heatmaps, anchors_tl_corners_regr, anchors_br_corners_regr, rois, cls_score , bbox_pred,bbox_targets, rois_label, bbox_inside_weights, bbox_outside_weights, nclasses
    ,K=2000, kernel=1, dist_threshold=0.2, num_dets=1000,layers_range = None,
    output_kernel_size = None, output_sizes = None, input_size=None, base_layer_range=None
):
    top_k = K
    batch, cat, height_0, width_0 = anchors_heats[0].size()
    
    for i in range(len(anchors_heats)):    
        anchors_heat = anchors_heats[i]
        #print(anchors_heat.shape)
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
            
            min_y, max_y, min_x, max_x = map(lambda x:x/8/2,base_layer_range) #This is the range of object sizes within the layers 
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

        scores = scores.view(batch, -1)

        scores, inds = torch.topk(scores, min(num_dets, scores.shape[1]))
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 4)
        bboxes = _gather_feat(bboxes, inds)

        #clses  =
        #clses  = 

        bboxes[:, :, 0] *= width_scale
        bboxes[:, :, 1] *= height_scale
        bboxes[:, :, 2] *= width_scale
        bboxes[:, :, 3] *= height_scale
        
        
        if i == 0:
            detections = torch.cat([bboxes, scores,scores,scores, clses], dim=2)
        else:
            detections = torch.cat([detections, torch.cat([bboxes, scores,scores,scores, clses], dim=2)], dim = 1)
   
    top_scores, top_inds = torch.topk(detections[:, :, 4],min(10000, detections.shape[1] ))
    detections = _gather_feat(detections, top_inds)
    return detections



