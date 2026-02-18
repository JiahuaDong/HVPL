import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from torch.cuda.amp import autocast

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from hvpl import add_vita_config
from predictor import VisualizationDemo

################################################
from detectron2.engine import default_setup, default_argument_parser
from continual import add_continual_config
from hvpl.data.datasets.builtin import register_all_coco_video, register_all_ytvis_2019, register_all_ytvis_2021, register_all_ovis
################################################

# constants
WINDOW_NAME = "vita video demo"


'''
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg
'''

def get_parser():
    parser = argparse.ArgumentParser(description="vita demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/youtubevis_2019/vita_R50_bs8.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'"
        "this will be treated as frames of a video",
    )
    parser.add_argument(
        "--output",
        default="visiual_results/youtube2019",  ################################################
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--save-frames",
        default=False,
        help="Save frame level image outputs.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


#######################################################################
def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.NAME = "Exp"

    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_vita_config(cfg)

    add_continual_config(cfg)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN_MEM = tuple(f"mem_{dataset}" for dataset in cfg.DATASETS.TRAIN)  ####
    #print(cfg.DATASETS.TRAIN_MEM)
    #cfg.MODEL.DETR.NUM_CLASSES = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
    #####This is important.

    if cfg.CONT.MODE == 'overlap':
        cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0]}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-ov"
    elif cfg.CONT.MODE == "disjoint":
        cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0]}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-dis"
    else:
        cfg.TASK_NAME = f"{cfg.DATASETS.TRAIN[0]}_{cfg.CONT.BASE_CLS}-{cfg.CONT.INC_CLS}-seq"
    if cfg.CONT.ORDER_NAME is not None:
        cfg.TASK_NAME += "-" + cfg.CONT.ORDER_NAME
    cfg.OUTPUT_ROOT = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "/" + cfg.TASK_NAME + "/" + cfg.NAME + "/step" + str(cfg.CONT.TASK)


    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    #setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")

    return cfg


#################################################################################################

def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False




if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    #args = get_parser().parse_args()
    ############################

    args = default_argument_parser().parse_args()
    args.confidence_threshold = 0.0
    args.save_frames = True
    args.output = "visiual_results/youtube2019"
    args.input = "/home/yinh/mydata/VITA-main/datasets/ytvis_2019/valid/JPEGImages/0a49f5265b/*.jpg"
    args.video_input = None

    #############################
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    #############################
    if hasattr(cfg, 'CONT') and cfg.CONT.TASK > 0:
        cfg.defrost()
        if cfg.CONT.WEIGHTS is None:  # load from last step
            cfg.MODEL.WEIGHTS = cfg.OUTPUT_ROOT + "/" + cfg.TASK_NAME + "/" + cfg.NAME + f"/step{cfg.CONT.TASK - 1}/model_final.pth"
        else:  # load from cfg
            cfg.MODEL.WEIGHTS = cfg.CONT.WEIGHTS
        cfg.freeze()
    if hasattr(cfg, 'CONT') and cfg.CONT.TASK == 0:
        cfg.defrost()
        cfg.MODEL.WEIGHTS = cfg.CONT.WEIGHTS
        cfg.freeze()

        _root_data = os.getenv("DETECTRON2_DATASETS", "datasets")
        register_all_ovis(_root_data, cfg, train=False)
        register_all_ytvis_2019(_root_data, cfg, train=False)
        register_all_ytvis_2021(_root_data, cfg, train=False)
        register_all_coco_video(_root_data, cfg)
    ##############################

    demo = VisualizationDemo(cfg, conf_thres=args.confidence_threshold)

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    if args.input:
        print(len(args.input))

        #if len(args.input) == 1:

        input_images = glob.glob(args.input)    #os.path.expanduser(args.input[0] )   ########################
        print(args.input)
        assert args.input, "The input path(s) was not found"

        vid_frames = []
        for path in input_images:
            print(path)
            img = read_image(path, format="BGR")
            vid_frames.append(img)

        start_time = time.time()
        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames)
        logger.info(
            "detected {} instances per frame in {:.2f}s".format(
                len(predictions["pred_scores"]), time.time() - start_time
            )
        )

        if args.output:
            if args.save_frames:
                for path, _vis_output in zip(args.input, visualized_output):
                    out_filename = os.path.join(args.output, os.path.basename(path))
                    _vis_output.save(out_filename)

            H, W = visualized_output[0].height, visualized_output[0].width

            cap = cv2.VideoCapture(-1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(args.output, "visualization.mp4"), fourcc, 10.0, (W, H), True)
            for _vis_output in visualized_output:
                frame = _vis_output.get_image()[:, :, ::-1]
                out.write(frame)
            cap.release()
            out.release()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        
        vid_frames = []
        while video.isOpened():
            success, frame = video.read()
            if success:
                vid_frames.append(frame)
            else:
                break

        start_time = time.time()
        with autocast():
            predictions, visualized_output = demo.run_on_video(vid_frames)
        logger.info(
            "detected {} instances per frame in {:.2f}s".format(
                len(predictions["pred_scores"]), time.time() - start_time
            )
        )

        if args.output:
            if args.save_frames:
                for idx, _vis_output in enumerate(visualized_output):
                    out_filename = os.path.join(args.output, f"{idx}.jpg")
                    _vis_output.save(out_filename)

            H, W = visualized_output[0].height, visualized_output[0].width

            cap = cv2.VideoCapture(-1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(args.output, "visualization.mp4"), fourcc, 10.0, (W, H), True)
            for _vis_output in visualized_output:
                frame = _vis_output.get_image()[:, :, ::-1]
                out.write(frame)
            cap.release()
            out.release()
