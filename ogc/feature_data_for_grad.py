from detectron2.modeling import build_model
import torch
import os
from .memory import update_memory_prefix, get_prefix_matrix
from hvpl.data.datasets.builtin import register_all_coco_video_mem, register_all_ytvis_2019_mem, register_all_ytvis_2021_mem, register_all_ovis_mem
from hvpl import (
    build_combined_loader,
    build_detection_train_loader,
)
import numpy as np
import pickle
from hvpl.data.dataset_mapper_mem import YTVISDatasetMapper, CocoClipDatasetMapper
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog


def get_feature_for_grad(cfg):
    try:
        _root_data = os.getenv("DETECTRON2_DATASETS", "datasets")
        register_all_ovis_mem(_root_data, cfg)
        register_all_ytvis_2019_mem(_root_data, cfg)
        register_all_ytvis_2021_mem(_root_data, cfg)
        register_all_coco_video_mem(_root_data, cfg)
    except:
        pass

    cfg.defrost()
    cfg.INPUT.SAMPLING_FRAME_NUM = 4

    mappers = []
    for d_i, dataset_name in enumerate(cfg.DATASETS.TRAIN_MEM):
        if dataset_name.startswith('mem_coco'):
            mappers.append(
                CocoClipDatasetMapper(
                    cfg, is_train=True, is_tgt=(d_i == len(cfg.DATASETS.TRAIN_MEM) - 1), src_dataset_name=dataset_name
                )
            )
        elif dataset_name.startswith('mem_ytvis') or dataset_name.startswith('mem_ovis'):
            mappers.append(
                YTVISDatasetMapper(cfg, is_train=True, is_tgt=(d_i == len(cfg.DATASETS.TRAIN_MEM) - 1),
                                   src_dataset_name=dataset_name)
            )
        else:
            raise NotImplementedError
    assert len(mappers) > 0, "No dataset is chosen!"

    if len(mappers) == 1:
        mapper = mappers[0]
        data_loader = build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN_MEM[0])
    else:

        loaders = []
        for mapper, dataset_name in zip(mappers, cfg.DATASETS.TRAIN_MEM):
            dataset = DatasetCatalog.get(dataset_name)
            if len(dataset) == 0:
                print(dataset_name, 'is empty')
            else:
                loader = build_detection_train_loader(cfg, mapper=mapper, dataset_name=dataset_name)
                loaders.append(loader)

        if len(loaders) > 1:
            data_loader = build_combined_loader(cfg, loaders, cfg.DATASETS.DATASET_RATIO)
        else:
            data_loader = loader

    cfg.defrost()
    cfg.CONT.TASK = cfg.CONT.TASK-1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
    num_prompts = cfg.CONT.NUM_PROMPTS
    if cfg.CONT.TASK == 0:
        cfg.CONT.NUM_PROMPTS = 0

    model = build_model(cfg)
    cfg.CONT.TASK = cfg.CONT.TASK + 1

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
    if num_prompts != 0:
        cfg.CONT.NUM_PROMPTS = num_prompts

    cfg.INPUT.SAMPLING_FRAME_NUM = 5
    cfg.freeze()

    checkpointer = DetectionCheckpointer(
        model,
        cfg.OUTPUT_DIR,

    )

    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    device = torch.device

    prefix_rep_frame = get_prefix_matrix(data_loader, model, device)


    if cfg.CONT.TASK > 1:
        load_path = cfg.OUTPUT_ROOT + "/" + cfg.TASK_NAME + "/" + cfg.NAME + f"/step{cfg.CONT.TASK - 1}/grad_data_frame.pkl"
        with open(load_path, 'rb') as f:
            feature_prefix_frame = pickle.load(f)
    else:
        feature_prefix_frame = {}


    threshold_frame = cfg.CONT.THRESHOLD   
    feature_prefix_frame = update_memory_prefix(prefix_rep_frame, threshold_frame, feature_prefix_frame)

    feature_prefix_gt = {0: {}}

    for layer in feature_prefix_frame:
        for item in feature_prefix_frame[layer]:
            temp_feature = feature_prefix_frame[layer][item].reshape(feature_prefix_frame[layer][item].shape[0], -1)
            Uf = torch.Tensor(np.dot(temp_feature, temp_feature.transpose()))  #.to(device)
            print('g', layer, item, Uf.size())  ##torch.Size([256, 256]) <class 'dict'>
            feature_prefix_gt[layer][item] = Uf

    del model
    del data_loader

    torch.cuda.empty_cache()

    print(type(feature_prefix_gt))

    save_path = os.path.join(cfg.OUTPUT_DIR, 'grad_data_frame.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(feature_prefix_frame, f)


    return feature_prefix_gt  





