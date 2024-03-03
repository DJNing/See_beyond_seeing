from distutils.log import debug
from genericpath import exists
import pickle
import time

import numpy as np
import torch
import tqdm
# import ipdb
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (
            metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def vis_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None,
                   runtime_gt=False, save_best_eval=False, best_mAP_3d=0.0, save_centers=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_output_dir = result_dir / 'best_eval'
    if save_best_eval:
        best_model_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    frame_ids = []

    attach_pw_dir = final_output_dir/'attach_pw'
    main_pw_dir = final_output_dir/'main_pw'
    attach_pw_dir.mkdir(exist_ok=True)
    main_pw_dir.mkdir(exist_ok=True)

    for i, batch_dict in enumerate(tqdm.tqdm(dataloader)):
        load_data_to_gpu(batch_dict)

        # with torch.no_grad():
        # calculate the transfer loss, backpropagate to the pooling weights layer
        # set requires_grad for pooling weights module
        start_run_time = time.time()
        pred_dicts, ret_dict = model(batch_dict)
        duration = time.time() - start_run_time
        frame_ids += list(batch_dict['frame_id'])
        attach_pw_dict = pred_dicts[0]['attach_pw_dict']
        main_pw_dict = pred_dicts[0]['main_pw_dict']
        save_name = str(frame_ids[-1]) + '.npy'
        attach_fname = str(attach_pw_dir / save_name)
        main_fname = str(main_pw_dir / save_name)
        np.save(attach_fname, attach_pw_dict)
        np.save(main_fname, main_pw_dict)

            
def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None,
                   runtime_gt=False, save_best_eval=False, best_mAP_3d=0.0, save_centers=False):
    result_dir.mkdir(parents=True, exist_ok=True)
    # ipdb.set_trace()
    final_output_dir = result_dir / 'final_result' / 'data'
    # import ipdb
    # ipdb.set_trace()
    # print('=================================')
    # print('save to file: %s' % save_to_file)
    # print('=================================')
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    # import ipdb
    # ipdb.set_trace()
    best_model_output_dir = result_dir / 'best_eval'
    if save_best_eval:
        best_model_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    frame_ids = []
    sum_duration = 0

    # collect centers, centers_origin for visualization
    center_dict = {}
    center_origin_dict = {}
    ip_dict = {}
    det_dict = {}
    match_dict = {}
    lidar_center_dict = {}
    lidar_preds_dict = {}
    radar_preds_dict = {}
    radar_label_dict = {}
    init_flag = False
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            start_run_time = time.time()
            pred_dicts, ret_dict = model(batch_dict)
            duration = time.time() - start_run_time
            frame_ids += list(batch_dict['frame_id'])

            if hasattr(model, 'vis'):
                vis = model.vis
            else:
                vis = False
            if hasattr(model, 'debug'):
                debug = model.debug
            else:
                debug = False
            save_center = save_centers & ('centers' in batch_dict)
            # import ipdb
            # ipdb.set_trace()
            if save_center:
                centers = batch_dict['centers'].cpu().numpy()
                centers_origin = batch_dict['centers_origin'].cpu().numpy()
                points = batch_dict['points'].cpu().numpy()

                center_dict[frame_ids[-1]] = centers
                center_origin_dict[frame_ids[-1]] = centers_origin
                ip_dict[frame_ids[-1]] = points
                # pointwise classification
                if debug or vis:
                    ipdb.set_trace()
                    radar_idx = batch_dict['radar_idx'].cpu().numpy().reshape([-1, 1])
                    lidar_idx = batch_dict['lidar_idx'].cpu().numpy().reshape([-1, 1])
                    mask = batch_dict['mask'].cpu().numpy().reshape([-1, 1])
                    matches = np.concatenate((radar_idx, lidar_idx, mask), axis=1)
                    lidar_center = batch_dict['lidar_centers'].cpu().numpy()
                    lidar_preds = batch_dict['lidar_preds'][2]
                    radar_cls_label = batch_dict['sa_ins_labels']
                    radar_preds = batch_dict['sa_ins_preds'][2]
                    # print('saving debug result')

                    match_dict[frame_ids[-1]] = matches
                    lidar_center_dict[frame_ids[-1]] = lidar_center
                    lidar_preds_dict[frame_ids[-1]] = lidar_preds
                    radar_preds_dict[frame_ids[-1]] = radar_preds
                    radar_label_dict[frame_ids[-1]] = radar_cls_label

        disp_dict = {}
        sum_duration += duration
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        det_dict[frame_ids[-1]] = annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    peak_memory = torch.cuda.max_memory_allocated() / 1024  # convert to KByte

    logger.info('Peak memory usage: %.4f KB.' % peak_memory)
    peak_memory = peak_memory / 1024  # convert to MByte
    logger.info('Peak memory usage: %.4f MB.' % peak_memory)
    logger.info('Average run time per scan: %.4f ms' % (sum_duration / len(dataloader.dataset) * 1000))
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    if save_to_file:
        logger.info('******************Saving result to dir: ' + str(result_dir) + '**********************')

        with open(result_dir / 'result.pkl', 'wb') as f:
            pickle.dump(det_annos, f)

    # gt pkl
    import copy

    gt_dict = {}
    try:
        for info in dataset.kitti_infos:
            frame_id = copy.deepcopy(info['point_cloud']['lidar_idx'])
            gt_anno = copy.deepcopy(info['annos'])
            gt_dict[frame_id] = gt_anno
            pass
    except Exception:
        logger.info('no available gt annos, running as testing')

        if save_to_file:
            with open(result_dir / 'gt.pkl', 'wb') as f:
                pickle.dump(gt_dict, f)

            # save detection
            with open(result_dir / 'dt.pkl', 'wb') as f:
                pickle.dump(det_dict, f)

            # save frame ids
            with open(result_dir / 'frame_ids.txt', 'w') as f:
                for id in frame_ids:
                    f.write(str(id) + ',')

        import sys
        sys.exit()

    gt_annos = []
    for id in frame_ids:
        gt_annos += [gt_dict[id]]
    
    # ipdb.set_trace()
    if save_to_file:
        with open(result_dir / 'gt.pkl', 'wb') as f:
            pickle.dump(gt_dict, f)

        # save detection
        with open(result_dir / 'dt.pkl', 'wb') as f:
            pickle.dump(det_dict, f)

        # save frame ids
        with open(result_dir / 'frame_ids.txt', 'w') as f:
            for id in frame_ids:
                f.write(str(id) + ',')
    # ipdb.set_trace()
    if save_center:

        save_name_list = (
            'centers', 'centers_origin', 'points', 'match', 'lidar_center', 'lidar_preds', 'radar_preds', 'radar_label')
        save_dict_list = (
            center_dict, center_origin_dict, ip_dict, match_dict, lidar_center_dict, lidar_preds_dict, radar_preds_dict,
            radar_label_dict)
        '''
        center_dict = {}
        center_origin_dict = {}
        ip_dict = {}
        match_dict = {}
        lidar_center_dict = {}
        lidar_preds_dict = {}
        radar_preds_dict = {}
        radar_label_dict = {}
        '''
        # # save centers 
        # with open(result_dir / 'centers.pkl', 'wb') as f:
        #     pickle.dump(center_dict, f)
        # # save centers_origin
        # with open(result_dir / 'centers_origin.pkl', 'wb') as f:
        #     pickle.dump(center_origin_dict, f)
        # # save input points
        # with open(result_dir / 'points.pkl', 'wb') as f:
        #     pickle.dump(ip_dict, f)

        for i, name in enumerate(save_name_list):
            save_data = save_dict_list[i]
            save_name = result_dir / (name + '.pkl')
            with open(save_name, 'wb') as f:
                pickle.dump(save_data, f)


    try:
        eval_results = dataset.evaluation(
            det_annos, class_names, gt_annos=gt_annos,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )
    except:
        eval_results = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path=final_output_dir
        )

    logger.info('*************** Evaluation Summary of EPOCH %s *****************' % epoch_id)
    
    # log_kitti_result(eval_results, logger, ret_dict)
    current_epoch_mAP_3d = log_vod_result(eval_results, logger, ret_dict)
    # save gt, prediction, final points origin, final points new coordinate
    
    if save_best_eval and current_epoch_mAP_3d > best_mAP_3d:
        logger.info('>>>>>> Saving best mAP_3d model save to %s <<<<<<' % result_dir)
        ckpt_name = best_model_output_dir / 'best_epoch_checkpoint'
        save_checkpoint(
            checkpoint_state(model, None, epoch_id, None), filename=ckpt_name,
        )
        logger.info('>>>>>>> current best mAP_3d result is: <<<<<<<')
        log_vod_result(eval_results, logger, ret_dict)
    

    logger.info('****************Evaluation done.*****************')
    return ret_dict

def log_kitti_result(eval_results, logger, ret_dict):
    logger.info('*************   kitti official evaluation script   *************')
    kitti_evaluation_result = eval_results['kitti_eval']
    logger.info("Results: \n"
                f"Entire annotated area: \n"
                f"Car: {kitti_evaluation_result['entire_area']['Car_3d_all']} \n"
                f"Pedestrian: {kitti_evaluation_result['entire_area']['Pedestrian_3d_all']} \n"
                f"Cyclist: {kitti_evaluation_result['entire_area']['Cyclist_3d_all']} \n"
                f"mAP: {(kitti_evaluation_result['entire_area']['Car_3d_all'] + kitti_evaluation_result['entire_area']['Pedestrian_3d_all'] + kitti_evaluation_result['entire_area']['Cyclist_3d_all']) / 3}")
    ret_dict['mAP_3d_kitti'] = (kitti_evaluation_result['entire_area']['Car_3d_all'] + kitti_evaluation_result['entire_area']['Pedestrian_3d_all'] + kitti_evaluation_result['entire_area']['Cyclist_3d_all']) / 3

def log_vod_result(eval_results, logger, ret_dict):
    logger.info('*************   vod official evaluation script   *************')
    vod_evaluation_result = eval_results['vod_eval']
    logger.info("Results: \n"
        f"Entire annotated area: \n"
        f"Car: {vod_evaluation_result['entire_area']['Car_3d_all']} \n"
        f"Pedestrian: {vod_evaluation_result['entire_area']['Pedestrian_3d_all']} \n"
        f"Cyclist: {vod_evaluation_result['entire_area']['Cyclist_3d_all']} \n"
        f"mAP: {(vod_evaluation_result['entire_area']['Car_3d_all'] + vod_evaluation_result['entire_area']['Pedestrian_3d_all'] + vod_evaluation_result['entire_area']['Cyclist_3d_all']) / 3} \n"
        f"Driving corridor area: \n"
        f"Car: {vod_evaluation_result['roi']['Car_3d_all']} \n"
        f"Pedestrian: {vod_evaluation_result['roi']['Pedestrian_3d_all']} \n"
        f"Cyclist: {vod_evaluation_result['roi']['Cyclist_3d_all']} \n"
        f"mAP: {(vod_evaluation_result['roi']['Car_3d_all'] + vod_evaluation_result['roi']['Pedestrian_3d_all'] + vod_evaluation_result['roi']['Cyclist_3d_all']) / 3} \n"
        )


    current_epoch_mAP_3d = (vod_evaluation_result['entire_area']['Car_3d_all'] + vod_evaluation_result['entire_area']['Pedestrian_3d_all'] + vod_evaluation_result['entire_area']['Cyclist_3d_all']) / 3
    ret_dict['mAP_3d'] = current_epoch_mAP_3d
    
    ret_dict['mAP_3d_vod'] = (vod_evaluation_result['entire_area']['Car_3d_all'] + vod_evaluation_result['entire_area']['Pedestrian_3d_all'] + vod_evaluation_result['entire_area']['Cyclist_3d_all']) / 3
    return current_epoch_mAP_3d

def log_inhouse_result(eval_results, logger, ret_dict):
    logger.info('*************   vod official evaluation script   *************')
    vod_evaluation_result = eval_results['vod_eval']
    logger.info("Results: \n"
        f"Entire annotated area: \n"
        f"Car: {vod_evaluation_result['entire_area']['Car_3d_all']} \n"
        f"Pedestrian: {vod_evaluation_result['entire_area']['Pedestrian_3d_all']} \n"
        f"Cyclist: {vod_evaluation_result['entire_area']['Cyclist_3d_all']} \n"
        f"mAP: {(vod_evaluation_result['entire_area']['Car_3d_all'] + vod_evaluation_result['entire_area']['Pedestrian_3d_all'] + vod_evaluation_result['entire_area']['Cyclist_3d_all']) / 3} \n"
        f"Driving corridor area: \n"
        f"Car: {vod_evaluation_result['roi']['Car_3d_all']} \n"
        f"Pedestrian: {vod_evaluation_result['roi']['Pedestrian_3d_all']} \n"
        f"Cyclist: {vod_evaluation_result['roi']['Cyclist_3d_all']} \n"
        f"mAP: {(vod_evaluation_result['roi']['Car_3d_all'] + vod_evaluation_result['roi']['Pedestrian_3d_all'] + vod_evaluation_result['roi']['Cyclist_3d_all']) / 3} \n"
        )


    current_epoch_mAP_3d = (vod_evaluation_result['entire_area']['Car_3d_all'] + vod_evaluation_result['entire_area']['Pedestrian_3d_all'] + vod_evaluation_result['entire_area']['Cyclist_3d_all']) / 3
    ret_dict['mAP_3d'] = current_epoch_mAP_3d
    
    ret_dict['mAP_3d_vod'] = (vod_evaluation_result['entire_area']['Car_3d_all'] + vod_evaluation_result['entire_area']['Pedestrian_3d_all'] + vod_evaluation_result['entire_area']['Cyclist_3d_all']) / 3
    return current_epoch_mAP_3d

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


if __name__ == '__main__':
    pass
