import socket
from datetime import datetime
import os.path as osp
import huepy as hue
import numpy as np

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('./')
from configs import args_faster_rcnn_hoim

from lib.datasets import get_data_loader
from lib.model.faster_rcnn_hoim import get_hoim_model
from lib.utils.misc import Nestedspace, resume_from_checkpoint, \
    get_optimizer, get_lr_scheduler
from lib.utils.distributed import init_distributed_mode, is_main_process
from lib.utils.trainer import get_trainer
from lib.utils.serialization import mkdir_if_missing


def main(args):
    if args.distributed:
        init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if is_main_process():
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        args.path = osp.join(
            args.path, current_time + '_' + socket.gethostname())
        mkdir_if_missing(args.path)
        print(hue.info(hue.bold(hue.lightgreen(
            'Working directory: {}'.format(args.path)))))
        if args.train.use_tfboard:
            tfboard = SummaryWriter(log_dir=args.path)
        args.export_to_json(osp.join(args.path, 'args.json'))
    else:
        tfboard = None

    train_loader = get_data_loader(args, train=True)

    model = get_hoim_model(pretrained_backbone=True,
                           num_features=args.num_features, num_pids=args.num_pids,
                           num_cq_size=args.num_cq_size, num_bg_size=args.num_bg_size,
                           oim_momentum=args.train.oim_momentum, oim_scalar=args.oim_scalar,
                           min_size=args.train.min_size, max_size=args.train.max_size,
                           anchor_scales=(args.anchor_scales,), anchor_ratios=(
                               args.anchor_ratios,),
                           # RPN parameters
                           rpn_pre_nms_top_n_train=args.train.rpn_pre_nms_top_n,
                           rpn_post_nms_top_n_train=args.train.rpn_post_nms_top_n,
                           # rpn_pre_nms_top_n_test=args.test.rpn_pre_nms_top_n,
                           # rpn_post_nms_top_n_test=args.test.rpn_post_nms_top_n,
                           rpn_nms_thresh=args.train.rpn_nms_thresh,
                           rpn_fg_iou_thresh=args.train.rpn_positive_overlap,
                           rpn_bg_iou_thresh=args.train.rpn_negative_overlap,
                           rpn_batch_size_per_image=args.train.rpn_batch_size,
                           rpn_positive_fraction=args.train.rpn_fg_fraction,
                           # Box parameters
                           box_score_thresh=args.train.fg_thresh,
                           # box_nms_thresh=args.test.nms, # inference only
                           box_detections_per_img=args.train.rpn_post_nms_top_n,  # use all
                           box_fg_iou_thresh=args.train.bg_thresh_hi,
                           box_bg_iou_thresh=args.train.bg_thresh_lo,
                           box_batch_size_per_image=args.train.rcnn_batch_size,
                           box_positive_fraction=args.train.fg_fraction,  # for proposals
                           bbox_reg_weights=args.train.box_regression_weights,
                           )
    model.to(device)

    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)

    if args.apex:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    model_without_ddp = model
    if args.distributed:
        if args.apex:
            from apex.parallel import DistributedDataParallel, convert_syncbn_model
            model = convert_syncbn_model(model)
            model = DistributedDataParallel(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.resume is not None:
        args, model_without_ddp, optimizer, lr_scheduler = resume_from_checkpoint(
            args, model_without_ddp, optimizer, lr_scheduler)

    trainer = get_trainer(args, model, model_without_ddp, train_loader,
                          optimizer, lr_scheduler, device, tfboard)

    trainer.run(train_loader, max_epochs=args.train.epochs)

    if is_main_process():
        tfboard.close()


if __name__ == '__main__':
    arg_parser = args_faster_rcnn_hoim()
    args = arg_parser.parse_args(namespace=Nestedspace())

    main(args)
