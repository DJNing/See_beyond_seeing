import argparse
import datetime
import glob
from itertools import cycle
import os
# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
from test import repeat_eval_ckpt, eval_single_ckpt, vis_single_ckpt
# from eval_utils import eval_utils
import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter
import time
from pcdet import models
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from eval_utils import eval_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/root/dj/code/CenterPoint-KITTI/tools/cfgs/inhouse_models/RaDetSSDv2.yaml', help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=1, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='initial_pct_0401', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default='/root/dj/code/CenterPoint-KITTI/output/centerpoint_radar_car/no_aug/ckpt/checkpoint_epoch_13.pth', help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default='/root/dj/code/CenterPoint-KITTI/output/RaDetSSDv2/initial_pct_0401/ckpt/checkpoint_epoch_20.pth', help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', default=False, help='')
    parser.add_argument('--freeze_part', default=True, help='load head params only and freeze them during training')
    parser.add_argument('--result_dir', type=str, default=None, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    # output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir = cfg.ROOT_DIR / 'output' / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    

    log_file = output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    log_str = 'CKPT PATH for this experiment: ' + str(ckpt_dir)
    logger.info('**' * 30)
    logger.info(log_str)
    logger.info('**' * 30)
    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard_eval')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    # cfg.MODEL['DISABLE_ATTACH'] = True
    # cfg.DATA_CONFIG['DEBUG'] = cfg.MODEL.get('DEBUG', False)
    cfg.DATA_CONFIG['DEBUG'] = True
    # cfg.DATA_CONFIG['USE_ATTACH'] = cfg.get('USE_ATTACH', False)
    cfg.DATA_CONFIG['USE_ATTACH'] = True
    cfg.MODEL['DEBUG'] = True
    cfg.MODEL['CLASS_NAMES'] = cfg.CLASS_NAMES
    cfg.MODEL['USE_POOLING_WEIGHT'] = True
    cfg.MODEL.BACKBONE_3D['USE_POOLING_WEIGHT'] = True
    cfg.MODEL.ATTACH_NETWORK.BACKBONE_3D['USE_POOLING_WEIGHT'] = True
    cfg.MODEL.FEAT_AUG['DEBUG'] = True
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set, tb_log=tb_log)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    init_mem_usage = torch.cuda.memory_allocated(torch.device('cuda'))

    model.cuda()


    # optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)


    model_mem_usage = torch.cuda.memory_allocated(torch.device('cuda'))
    
    model_size = model_mem_usage - init_mem_usage

    model_size = model_size / 1024 / 1024 # in MB
    logger.info('model size is %.4f MB' % model_size)
    logger.info(model)

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )

    ckpt_tag = args.pretrained_model.split('/')[-1].split('.')[0]
    eval_output_dir = output_dir / 'eval' / ckpt_tag
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    # eval_single_ckpt(model, test_loader, args, \
    # eval_output_dir, logger=logger, epoch_id=args.epochs, \
    # reload=False, save_to_file=args.save_to_file,\
    #     result_dir=args.result_dir, save_centers=True)
    # args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs

    vis_single_ckpt(model, test_loader, args, \
    eval_output_dir, logger=logger, epoch_id=args.epochs, \
    reload=False, save_to_file=args.save_to_file,\
        result_dir=args.result_dir, save_centers=True)
    # repeat_eval_ckpt(
    #     model.module if dist_train else model,
    #     test_loader, args, eval_output_dir, logger, ckpt_dir,
    #     dist_test=dist_train
    # )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
