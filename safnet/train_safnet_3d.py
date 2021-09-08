#!/usr/bin/env python
import os
import os.path as osp
import sys
import argparse
import logging
import time
import socket
import warnings
import pdb

import open3d  # import before torch
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Assume that the script is run at the root directory
_ROOT_DIR = os.path.abspath(osp.dirname(__file__) + '/..')
sys.path.insert(0, _ROOT_DIR)

from common.solver.build import build_optimizer, build_scheduler
from common.nn.freezer import Freezer
from common.utils.checkpoint import CheckpointerV2
from common.utils.logger import setup_logger
from common.utils.metric_logger import MetricLogger
from common.utils.torch_util import set_random_seed
from common.utils.sampler import IterationBasedBatchSampler

from safnet.models.build import build_model_mvpnet_3d
from safnet.data.build import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch 3D Deep Learning Training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument('--similarity', type=str, default='point', help='cache directory')
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def train(similarity,cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # It is recommended not to modify this section.
    # ---------------------------------------------------------------------------- #
    logger = logging.getLogger('safnet.train')

    # build model
    set_random_seed(cfg.RNG_SEED)
    model, loss_fn, train_metric, val_metric = build_model_mvpnet_3d(similarity,cfg)

    # net_3d_ckpt_path = 'late_fusion/0706_attention_nn_bidirection_image/protect/model_best_0.6855.pth'


    if net_3d_ckpt_path:
        checkpoint = torch.load(net_3d_ckpt_path, map_location=torch.device("cpu"))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() }#if (k in model_dict) and ('net_2d' in k)}
        # pdb.set_trace()
        model_dict.update(pretrained_dict)

        # pretrained_model_dict = state_dict(checkpoint['model'])

        # pdb.set_trace()
        # checkpoint = torch.load(net_2d_ckpt_path, map_location=torch.device("cpu"))
        # pretrained_dict = {k: v for k, v in checkpoint['model'].items() if (k in model_dict) and ('net_2d' in k)}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict,False)

    logger.info('Build model:\n{}'.format(str(model)))
    num_params = sum(param.numel() for param in model.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model).cuda()
    elif num_gpus == 1:
        model = model.cuda()
    else:
        raise NotImplementedError('Not support cpu training now.')

    reg_loss = Regularization(model, 0.00001, p=1).cuda()

    # build optimizer
    # model_cfg = cfg.MODEL[cfg.MODEL.TYPE]
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer = CheckpointerV2(model,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  save_dir=output_dir,
                                  logger=logger,
                                  max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    # path = '0315_batchsize/model_best.pth'
    # path = '0429_chamfer_loss/model_012000.pth'
    # checkpoint_data = checkpointer.load(path, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)

    checkpoint_data = checkpointer.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build freezer
    # if cfg.TRAIN.FROZEN_PATTERNS:
    #     freezer = Freezer(model, cfg.TRAIN.FROZEN_PATTERNS)
    #     # freezer.freeze(verbose=True)  # sanity check
    # else:
    #     freezer = None

    if True:
        freezer2 = Freezer(model, ("module:net_2d", "net_2d"))
        # pdb.set_trace()
        freezer3 = Freezer(model, ("module:similarity", "similarity"))
        # freezer5 = Freezer(model, ("module:net_2d", "net_2d"))
        # freezer4 = Freezer(model, ("module:similarity", "similarity"))
        # freezer2.freeze(verbose=True)  # sanity check
        # freezer3.freeze(verbose=True)
        # freezer3 = None
    else:
        freezer2 = None
        freezer3 = None
    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    set_random_seed(cfg.RNG_SEED)
    train_dataloader = build_dataloader(cfg, mode='train')
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val') if val_period > 0 else None

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writier = SummaryWriter(tb_dir)
    else:
        summary_writier = None

    # ---------------------------------------------------------------------------- #
    # Train
    # Customization begins here.
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data.get('iteration', 0)
    # start_iteration = 0
    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = checkpoint_data.get(best_metric_name, None)
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    if not isinstance(train_metric, (list, tuple)):
        train_metric = [train_metric]
    if not isinstance(val_metric, (list, tuple)):
        val_metric = [val_metric]
    train_metric_logger = MetricLogger(delimiter='  ')
    train_metric_logger.add_meters(train_metric)
    val_metric_logger = MetricLogger(delimiter='  ')
    val_metric_logger.add_meters(val_metric)

    # wrap the dataloader
    batch_sampler = train_dataloader.batch_sampler
    train_dataloader.batch_sampler = IterationBasedBatchSampler(batch_sampler, max_iteration, start_iteration)

    def setup_train():
        # set training mode
        model.train()
        loss_fn.train()
        # loss_fn2.train()
        # loss_fn3.train()
        # freeze parameters/modules optionally
        # if freezer is not None:
        #     freezer.freeze()
        # if freezer2 is not None:
        #     freezer2.freeze(verbose=False)
        if freezer3 is not None:
            freezer3.freeze(verbose=False)
        #     freezer3.freeze()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model.eval()
        loss_fn.eval()
        # loss_fn2.eval()
        # loss_fn3.eval()
        # reset metric
        val_metric_logger.reset()

    setup_train()
    end = time.time()
    for iteration, data_batch in enumerate(train_dataloader, start_iteration):
        is_training = False
        data_time = time.time() - end
        # copy data from cpu to gpu
        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
        # forward
        # if iteration >=10000:
        if False:
            if iteration % 1000 == 0:
                print(iteration)
                if ((iteration / 1000) % 2) == 0:
                    print('train_point')
                    freezer2.freeze(verbose=False)  #image
                    freezer3.unfreeze(verbose=False) #point
                else:
                    print('train_image')
                    freezer2.unfreeze(verbose=False)
                    freezer3.freeze(verbose=False)
        preds,recon_loss,depth_loss = model(data_batch)
        # update losses
        optimizer.zero_grad()
        loss_dict = loss_fn(preds, data_batch)
        # loss_dict['recon_loss'] = recon_loss
        loss_dict['depth_loss'] = depth_loss.mean()
        # loss_dict['reg_loss'] = reg_loss(model)
        if is_training:
            print('is training')
            loss_dict['recon_loss'] = recon_loss.mean()
        # data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
        # loss_dict2 = loss_fn2(preds, data_batch)
        # loss_dict3 = loss_fn3(preds, data_batch)
        # pdb.set_trace()
        # total_loss = sum(loss_dict.values())+reg_loss(model)
        if "seg_loss_2d" in loss_dict.keys():
            if is_training:
                total_loss = (loss_dict['seg_loss_3d']
                              + 0.7*loss_dict['seg_loss_2d_3d']
                              # + 0.3*loss_dict['seg_loss_2d']
                              + 0*loss_dict['recon_loss']
                              + 0.3*loss_dict['depth_loss']
                              # + 0.01*loss_dict['reg_loss']
                              )
            else:
                total_loss = (loss_dict['seg_loss_3d']
                              + 0.2 * loss_dict['seg_loss_2d_3d']
                              + 0.1 * loss_dict['seg_loss_2d']
                              + 0.8 * loss_dict['seg_loss_point_branch']
                              + 0.1*loss_dict['depth_loss']
                              # + 0.01*loss_dict['reg_loss']
                              )
                # + 0.1 * loss_dict['recon_loss'])
        else:
            total_loss = sum(loss_dict.values()) + reg_loss(model)
        # It is slightly faster to update metrics and meters before backward
        with torch.no_grad():
            train_metric_logger.update(loss=total_loss, **loss_dict)
            for metric in train_metric:
                metric.update_dict(preds, data_batch)

        # backward
        total_loss.backward()
        if cfg.OPTIMIZER.MAX_GRAD_NORM > 0:
            # CAUTION: built-in clip_grad_norm_ clips the total norm.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.OPTIMIZER.MAX_GRAD_NORM)
        optimizer.step()

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)
        cur_iter = iteration + 1

        # log
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD) == 0:
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writier is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writier.add_scalar('train/' + name, meter.global_avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data['iteration'] = cur_iter
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save('model_{:06d}'.format(cur_iter), **checkpoint_data)

        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            end = time.time()
            with torch.no_grad():
                for iteration_val, data_batch in enumerate(val_dataloader):
                    is_training = False
                    data_time = time.time() - end
                    # copy data from cpu to gpu
                    data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}
                    # forward
                    preds,recon_loss,depth_loss = model(data_batch)
                    # update losses and metrics
                    loss_dict = loss_fn(preds, data_batch)
                    # loss_dict['recon_loss'] = recon_loss.mean()
                    loss_dict['depth_loss'] = depth_loss.mean()
                    total_loss = sum(loss_dict.values())
                    # update metrics and meters
                    val_metric_logger.update(loss=total_loss, **loss_dict)
                    for metric in val_metric:
                        metric.update_dict(preds, data_batch)

                    batch_time = time.time() - end
                    val_metric_logger.update(time=batch_time, data=data_time)
                    end = time.time()

                    if cfg.VAL.LOG_PERIOD > 0 and iteration_val % cfg.VAL.LOG_PERIOD == 0:
                        logger.info(
                            val_metric_logger.delimiter.join(
                                [
                                    'iter: {iter:4d}',
                                    '{meters}',
                                    'max mem: {memory:.0f}',
                                ]
                            ).format(
                                iter=iteration,
                                meters=str(val_metric_logger),
                                memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                            )
                        )

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writier is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writier.add_scalar('val/' + name, meter.global_avg, global_step=cur_iter)

            # best validation
            if cfg.VAL.METRIC in val_metric_logger.meters:
                cur_metric = val_metric_logger.meters[cfg.VAL.METRIC].global_avg
                if best_metric is None \
                        or ('loss' not in cfg.VAL.METRIC and cur_metric > best_metric) \
                        or ('loss' in cfg.VAL.METRIC and cur_metric < best_metric):
                    best_metric = cur_metric
                    checkpoint_data['iteration'] = cur_iter
                    checkpoint_data[best_metric_name] = best_metric
                    checkpointer.save('model_best', tag=False, **checkpoint_data)

            # restore training
            setup_train()

        # since pytorch v1.1.0, lr_scheduler is called after optimization.
        if scheduler is not None:
            scheduler.step()
        end = time.time()

    logger.info('Best val-{} = {}'.format(cfg.VAL.METRIC, best_metric))
    return model


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from common.config import purge_cfg
    from safnet.config.mvpnet_3d import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    output_dir = 'late_fusion/20210301_pointconv_new_image'
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs', 'outputs'))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('safnet', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    from common.utils.misc import collect_env_info
    logger.info('Collecting env info (might take some time)\n' + collect_env_info())

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    assert cfg.TASK == 'mvpnet_3d'
    train(args.similarity,cfg, output_dir, run_name)


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):

        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):

        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):

        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):

        reg_loss = 0
        for name, w in weight_list:
            if 'weight' in name:
                l2_reg = torch.norm(w, p=p)
                reg_loss = reg_loss + l2_reg
                # print(name)
                # break
        # pdb.set_trace()
        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):

        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")

if __name__ == '__main__':
    main()
