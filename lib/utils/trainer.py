import torch
from torch.nn.utils import clip_grad_norm_
from apex import amp
from ignite.engine.engine import Engine, Events
import huepy as hue
import time
import os.path as osp

from .misc import ship_data_to_cuda, lucky_bunny, warmup_lr_scheduler
from .distributed import reduce_dict, is_main_process
from .logger import MetricLogger
from .serialization import save_checkpoint


def get_trainer(args, model, model_without_ddp, train_loader, optimizer, lr_scheduler, device, tfboard):

    def _update_model(engine, data):
        images, targets = ship_data_to_cuda(data, device)

        loss_dict = model(images, targets)

        losses = args.train.w_RPN_loss_cls * loss_dict['loss_objectness'] \
            + args.train.w_RPN_loss_box * loss_dict['loss_rpn_box_reg'] \
            + args.train.w_RCNN_loss_bbox * loss_dict['loss_box_reg'] \
            + args.train.w_RCNN_loss_cls * loss_dict['loss_detection'] \
            + args.train.w_OIM_loss_oim * loss_dict['loss_reid']

        # reduce losses over all GPUs for logging purposes
        if engine.state.iteration % args.train.disp_interval == 0:
            loss_dict_reduced = reduce_dict(loss_dict)
            losses_reduced = args.train.w_RPN_loss_cls * loss_dict_reduced['loss_objectness'] \
                + args.train.w_RPN_loss_box * loss_dict_reduced['loss_rpn_box_reg'] \
                + args.train.w_RCNN_loss_bbox * loss_dict_reduced['loss_box_reg'] \
                + args.train.w_RCNN_loss_cls * loss_dict_reduced['loss_detection'] \
                + args.train.w_OIM_loss_oim * loss_dict_reduced['loss_reid']
            loss_value = losses_reduced.item()
            state = dict(loss_value=loss_value,
                         lr=optimizer.param_groups[0]['lr'])
            state.update(loss_dict_reduced)
        else:
            state = None

        optimizer.zero_grad()
        if args.apex:
            with amp.scale_loss(losses, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses.backward()
        if args.train.clip_gradient > 0:
            clip_grad_norm_(model.parameters(), args.train.clip_gradient)
        optimizer.step()

        return state

    trainer = Engine(_update_model)

    @trainer.on(Events.STARTED)
    def _init_run(engine):
        engine.state.epoch = args.train.start_epoch
        engine.state.iteration = args.train.start_epoch * len(train_loader)

        if args.debug:
            def _get_rcnn_fg_bg_ratio(module, input, output):
                if engine.state.iteration % args.train.disp_interval == 0:
                    targets = torch.cat(input[1])
                    num_fg = targets.gt(0).sum()
                    num_bg = targets.eq(0).sum()
                    debug_info = {'num_fg': num_fg,
                                  'num_bg': num_bg}
                    debug_info_reduced = reduce_dict(debug_info, average=False)
                    engine.state.debug_info = {k: v.item()
                                               for k, v in debug_info_reduced.items()}
            model_without_ddp.roi_heads.reid_loss.register_forward_hook(
                _get_rcnn_fg_bg_ratio)

    @trainer.on(Events.EPOCH_STARTED)
    def _init_epoch(engine):
        if engine.state.epoch == 1 and args.train.lr_warm_up:
            warmup_factor = 1. / 1000
            warmup_iters = len(train_loader) - 1
            engine.state.sub_scheduler = warmup_lr_scheduler(
                optimizer, warmup_iters, warmup_factor)
        lucky_bunny(engine.state.epoch)
        lr_scheduler.step()
        engine.state.metric_logger = MetricLogger()

    @trainer.on(Events.ITERATION_STARTED)
    def _init_iter(engine):
        if engine.state.epoch == 1 and args.train.lr_warm_up:  # epoch start from 1
            engine.state.sub_scheduler.step()
        if engine.state.iteration % args.train.disp_interval == 0:
            engine.state.start = time.time()

    @trainer.on(Events.ITERATION_COMPLETED)
    def _post_iter(engine):
        if engine.state.iteration % args.train.disp_interval == 0:
            # Update logger
            batch_time = time.time() - engine.state.start
            engine.state.metric_logger.update(batch_time=batch_time)
            engine.state.metric_logger.update(**engine.state.output)
            if hasattr(engine.state, 'debug_info'):
                engine.state.metric_logger.update(**engine.state.debug_info)
            # Print log on console
            step = (engine.state.iteration - 1) % len(train_loader) + 1
            engine.state.metric_logger.print_log(engine.state.epoch, step,
                                                 len(train_loader))
            # Record log on tensorboard
            if args.train.use_tfboard and is_main_process():
                for k, v in engine.state.metric_logger.meters.items():
                    if 'loss' in k:
                        k = k.replace('loss_', 'Loss/')
                    if 'num' in k:
                        tfboard.add_scalars('Debug/fg_bg_ratio', {k: v.avg},
                                            engine.state.iteration)
                    else:
                        tfboard.add_scalar(k, v.avg, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _post_epoch(engine):
        if is_main_process():
            save_name = osp.join(args.path, 'checkpoint.pth')
            save_checkpoint({
                'epoch': engine.state.epoch,
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, save_name)
            print(hue.good('save model: {}'.format(save_name)))

            tfboard.export_scalars_to_json(
                osp.join(args.path, 'all_scalar.json'))

    return trainer
