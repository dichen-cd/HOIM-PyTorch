from collections import OrderedDict

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .resnet_backbone import resnet_backbone
from ..loss import HOIMLoss


class FasterRCNN_HOIM(GeneralizedRCNN):
    """
    Implements HOIM model based on Faster R-CNN.
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label (person id) for each ground-truth box
                                        ranging from 1 -> N, unlabled persons are marked as 5555
    The model returns a Dict[Tensor] during training, containing the classification, regression and re-ID
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - scores (Tensor[N]): the scores for each prediction
        - embeddings (Tensor[N, d]): the embedding for each prediction
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or a OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
                           num_classes = 2 for person search (background & pedestrians)
        num_pids (int): number of labeled persons in dataset.
                        5532 for CUHK-SYSU, 482 for PRW
        num_cq_size (int): circular queue of OIM loss.
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        feat_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of feat_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    """

    def __init__(self, backbone,
                 num_classes=None, num_pids=5532, num_cq_size=5000,
                 # transform parameters
                 min_size=900, max_size=1500,
                 image_mean=None, image_std=None,
                 # Anchor settings:
                 anchor_scales=None, anchor_ratios=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, feat_head=None, box_predictor=None,
                 box_score_thresh=0.0, box_nms_thresh=0.4, box_detections_per_img=300,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.1,
                 box_batch_size_per_image=128, box_positive_fraction=0.5,
                 bbox_reg_weights=None,
                 # ReID parameters
                 embedding_head=None, reid_loss=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                'backbone should contain an attribute out_channels '
                'specifying the number of output channels (assumed to be the '
                'same for all the levels)')

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    'num_classes should be None when box_predictor is specified')
        else:
            if box_predictor is None:
                raise ValueError('num_classes should not be None when box_predictor'
                                 'is not specified')

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            if anchor_scales is None:
                anchor_scales = ((32, 64, 128, 256, 512),)
            if anchor_ratios is None:
                anchor_ratios = ((0.5, 1.0, 2.0),)
            rpn_anchor_generator = AnchorGenerator(
                anchor_scales, anchor_ratios)

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = self._set_rpn(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['feat_res4'],
                output_size=14,
                sampling_ratio=2)

        if feat_head is None:
            raise ValueError('feat_head should be specified manually.')
            # resolution = box_roi_pool.output_size[0]
            # representation_size = 2048
            # # ConvHead should be part of the backbone
            # # feat_head = TwoMLPHead(
            # #     out_channels * resolution ** 2,
            # #     representation_size)

        if box_predictor is None:
            box_predictor = CoordRegressor(
                2048,
                num_classes)

        if embedding_head is None:
            embedding_head = ReIDEmbeddingProj(
                featmap_names=['feat_res4', 'feat_res5'],
                in_channels=[1024, 2048],
                dim=256)

        if reid_loss is None:
            reid_loss = HOIMLoss(
                256, num_pids, num_cq_size,
                0.5, 30.0)

        roi_heads = self._set_roi_heads(
            embedding_head, reid_loss,
            box_roi_pool, feat_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

        super(FasterRCNN_HOIM, self).__init__(
            backbone, rpn, roi_heads, transform)

    def _set_roi_heads(self, *args):
        return HOIMRoIHeads(*args)

    def _set_rpn(self, *args):
        return RegionProposalNetwork(*args)

    def ex_feat(self, images, targets, mode='det'):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result: (tuple(Tensor)): list of 1 x d embedding for the RoI of each image

        """
        if mode == 'det':
            return self.ex_feat_by_roi_pooling(images, targets)
        elif mode == 'reid':
            return self.ex_feat_by_img_crop(images, targets)

    def ex_feat_by_roi_pooling(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals = [x['boxes'] for x in targets]

        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)
        rcnn_features = self.roi_heads.feat_head(roi_pooled_features)
        embeddings = self.roi_heads.embedding_head(rcnn_features)
        return embeddings.split(1, 0)

    def ex_feat_by_img_crop(self, images, targets):
        assert len(images) == 1, 'Only support batch_size 1 in this mode'

        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)),
                             targets[0]['boxes'][0].tolist())
        input_tensor = images.tensors[:, :, y1:y2 + 1, x1:x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.feat_head(features)
        embeddings = self.roi_heads.embedding_head(rcnn_features)
        return embeddings.split(1, 0)


class HOIMRoIHeads(RoIHeads):

    def __init__(self, embedding_head, reid_loss, *args, **kwargs):
        super(HOIMRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = embedding_head
        self.reid_loss = reid_loss

    @property
    def feat_head(self):  # this name is better
        return self.box_head

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, \
                    'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, \
                    'target labels must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, \
                        'target keypoints must of float type'

        labels = None
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)

        roi_pooled_features = \
            self.box_roi_pool(features, proposals, image_shapes)
        rcnn_features = self.feat_head(roi_pooled_features)
        box_regression = self.box_predictor(rcnn_features['feat_res5'])
        embeddings_ = self.embedding_head(rcnn_features)
        class_score, loss_detection, loss_reid = \
            self.reid_loss(embeddings_, labels)

        result, losses = [], {}
        if self.training:
            det_labels = [y.clamp(0, 1) for y in labels]
            loss_box_reg = \
                coord_regression_loss(class_score, box_regression,
                                      det_labels, regression_targets)

            losses = dict(loss_detection=loss_detection,
                          loss_box_reg=loss_box_reg,
                          loss_reid=loss_reid)
        else:
            boxes, scores, embeddings, labels = \
                self.postprocess_detections(class_score, box_regression, embeddings_,
                                            proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings[i],
                    )
                )
        # Mask and Keypoint losses are deleted deleted
        return result, losses

    def postprocess_detections(self, pred_scores, box_regression, embeddings_, proposals, image_shapes):
        device = pred_scores.device
        num_classes = pred_scores.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings_.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(pred_boxes, pred_scores, pred_embeddings, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            # embeddings are already personized.

            # batch everything, by making every class prediction be a separate
            # instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = boxes[
                inds], scores[inds], labels[inds], embeddings[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels


class CoordRegressor(nn.Module):
    """
    bounding box regression layers, without classification layer.
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
                           default = 2 for pedestrian detection
    """

    def __init__(self, in_channels, num_classes=2, RCNN_bbox_bn=True):
        super(CoordRegressor, self).__init__()
        if RCNN_bbox_bn:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes),
                nn.BatchNorm1d(4 * num_classes))
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.cls_score = nn.Linear(in_channels, num_classes)  # useless here.

        init.normal_(self.bbox_pred[0].weight, std=0.01)
        init.normal_(self.bbox_pred[1].weight, std=0.01)
        init.constant_(self.bbox_pred[0].bias, 0)
        init.constant_(self.bbox_pred[1].bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def coord_regression_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()
    return box_loss


class ReIDEmbeddingProj(nn.Module):

    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256):
        super(ReIDEmbeddingProj, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = map(int, in_channels)
        self.dim = int(dim)

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(
                nn.Linear(in_chennel, indv_dim),
                nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

    def forward(self, featmaps):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
        '''
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            return F.normalize(self.projectors[k](v))
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(
                    self.projectors[k](v)
                )
            return F.normalize(torch.cat(outputs, dim=1))

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


def get_hoim_model(pretrained_backbone=True,
                   num_features=256, num_pids=5532,
                   num_cq_size=5000, num_bg_size=5000,
                   oim_momentum=0.5, oim_scalar=30.0, **kwargs):
    backbone, conv_head = resnet_backbone('resnet50', pretrained_backbone)
    hoim = HOIMLoss(num_features, num_pids, num_cq_size, num_bg_size,
                    oim_momentum, oim_scalar,
                    omega_decay=0.99,
                    dynamic_lambda=True)
    coord_fc = CoordRegressor(2048, num_classes=2)
    model = FasterRCNN_HOIM(backbone,
                            feat_head=conv_head,
                            reid_loss=hoim,
                            box_predictor=coord_fc,
                            **kwargs)
    return model
