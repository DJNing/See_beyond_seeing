import argparse
import datetime
import glob
from itertools import cycle
import os
from pathlib import Path
from re import S
from test import repeat_eval_ckpt, eval_single_ckpt
import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/root/dj/code/CenterPoint-KITTI/tools/cfgs/inhouse_models/IA-SSD.yaml', help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=1, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='debug_new', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
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
    parser.add_argument('--freeze_part', type=bool, default=False, help='load head params only and freeze them during training')
    parser.add_argument('--eval_epoch', type=int, default=1, help='number of epoch for eval once')
    parser.add_argument('--eval_save', type=bool, default=True, help='save best eval model during training')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='whether to use multiple gpu for training')

    args = parser.parse_args()
    print(args.freeze_part)
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
    

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
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
        logger.info('visble GPUs count: %d' % total_gpus)
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None
    cfg.DATA_CONFIG['USE_ATTACH'] = cfg.get('USE_ATTACH', False)
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

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set, tb_log=tb_log)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.cuda()
    
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)
    torch.autograd.set_detect_anomaly(True)
    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    # print(type(args.freeze_part))
    # print(args.freeze_part)
    if args.freeze_part == True:
        # print('freeze_part is True')
        if cfg.get('FREEZE_MODE', None) is None:
            raise ValueError
        else:
            cfg.MODEL['FREEZE_MODE'] = cfg.FREEZE_MODE
    if args.pretrained_model is not None:
        logger.info('===> loading pretrained model %s '%args.pretrained_model)
        if args.freeze_part:
        #     if cfg.MODEL.get('MULTIBACKBONE', False):
        #         model.load_params_from_file_singlebranch(filename=args.pretrained_model, to_cpu=dist, logger=logger, id=cfg.FREEZE_MODE)
        #     else:
            model.load_params_from_file_dynamic(filename=args.pretrained_model, to_cpu=dist, logger=logger, id=cfg.FREEZE_MODE)
            # cfg.MODEL['FREEZE_MODE'] = cfg.FREEZE_MODE
        else:
            model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

    if cfg.get('USE_ATTACH', False):
        logger.info('===> Loading ckpt for attached model')
        model.load_ckpt_to_attach(cfg.MODEL.ATTACH_NETWORK.CKPT_FILE, logger)

    if cfg.get('BACKBONE_CKPT', False):
        logger.info('===> Loading ckpt for main backbone')
        for temp_dict in cfg.BACKBONE_CKPT:
            bb_id = temp_dict.ID
            ckpt_file = temp_dict.FILE_PATH
            model.load_backbone_params(ckpt_file, logger, backbone_id=bb_id)
        

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            logger.info('===> Resuming training from ckpt: %s ' % ckpt_list[-1])
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if cfg.get('USE_ATTACH', False):
        model.freeze_attach(logger)
        
    freeze_mode = model.model_cfg.get('FREEZE_MODE', None)
    # all attribute related operation should be perform before parallel wrapper

    if dist_train:
        logger.info('distributed training')
        logger.info('cfg.LOCAL_RANK = ', cfg.LOCAL_RANK)
        logger.info('device count %d ' % torch.cuda.device_count())
        logger.info([cfg.LOCAL_RANK % torch.cuda.device_count()])
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        # batch_size=args.batch_size,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        test_loader,
        cfg,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger,
        eval_epoch=args.eval_epoch,
        eval_output_dir=eval_output_dir,
        save_best_eval=args.eval_save,
        freeze_mode=freeze_mode
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    args.start_epoch = max(args.epochs - 10, 0)  # Only evaluate the last 10 epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
