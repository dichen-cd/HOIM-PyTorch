import os.path as osp
import huepy as hue

import numpy as np
import torch
from torch.backends import cudnn

import sys
sys.path.append('./')
from configs import args_faster_rcnn_hoim

from lib.datasets import get_data_loader
from lib.model.faster_rcnn_hoim import get_hoim_model
from lib.utils.misc import lazy_arg_parse, Nestedspace, \
    resume_from_checkpoint
from lib.utils.evaluator import inference, detection_performance_calc


if __name__ == '__main__':
    arg_parser = args_faster_rcnn_hoim()
    new_args = lazy_arg_parse(arg_parser)

    args = Nestedspace()
    args.load_from_json(osp.join(new_args.path, 'args.json'))
    args.from_dict(new_args.to_dict())

    device = torch.device(args.device)
    cudnn.benchmark = False

    print(hue.info(hue.bold(hue.lightgreen(
        'Working directory: {}'.format(args.path)))))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gallery_loader, probe_loader = get_data_loader(args, train=False)

    model = get_hoim_model(pretrained_backbone=False,
                           num_features=args.num_features, num_pids=args.num_pids,
                           num_cq_size=args.num_cq_size, num_bg_size=args.num_bg_size,
                           oim_momentum=args.train.oim_momentum, oim_scalar=args.oim_scalar,
                           min_size=args.train.min_size, max_size=args.train.max_size,
                           anchor_scales=(args.anchor_scales,), anchor_ratios=(args.anchor_ratios,),
                           # RPN parameters
                           rpn_pre_nms_top_n_test=args.test.rpn_pre_nms_top_n,
                           rpn_post_nms_top_n_test=args.test.rpn_post_nms_top_n,
                           rpn_nms_thresh=args.test.rpn_nms_thresh,
                           # Box parameters
                           box_nms_thresh=args.test.nms,
                           box_detections_per_img=args.test.rpn_post_nms_top_n,
                           )
    model.to(device)

    args.resume = osp.join(args.path, 'checkpoint.pth')
    args, model, _, _ = resume_from_checkpoint(args, model)

    name_to_boxes, all_feats, probe_feats = \
        inference(model, gallery_loader, probe_loader, device)

    print(hue.run('Evaluating detections:'))
    precision, recall = detection_performance_calc(gallery_loader.dataset,
                                                   name_to_boxes.values(),
                                                   det_thresh=0.01)

    print(hue.run('Evaluating search: '))
    gallery_size = 100 if args.dataset == 'CUHK-SYSU' else -1
    ret = gallery_loader.dataset.search_performance_calc(
        gallery_loader.dataset, probe_loader.dataset,
        name_to_boxes.values(), all_feats, probe_feats,
        det_thresh=0.5, gallery_size=gallery_size)
