import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
#from detectron2.data.datasets.coco import register_coco_instances
from .Additional_coco_increment import register_coco_instances
from .Additional_coco_increment_mem import register_coco_instances_mem
from hvpl.data.datasets.ytvis_mem import register_ytvis_instances_mem

from .ovis import _get_ovis_instances_meta
from hvpl.data.datasets.ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta
)

from .ytvis_val import register_ytvis_instances_val


# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "split_dataset/YouTube-2019/train_split.json"),

    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "ytvis_2019_val_all_frames": ("ytvis_2019/valid_all_frames/JPEGImages",
                        "ytvis_2019/valid_all_frames.json"),
}


_PREDEFINED_SPLITS_YTVIS_2019_val = {
    "ytvis_2019_val": ("ytvis_2019/train/JPEGImages",
                       "split_dataset/YouTube-2019/test_split.json"),
}

# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "split_dataset/YouTube-2021/train_split.json"),

    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

_PREDEFINED_SPLITS_YTVIS_2021_val = {
    "ytvis_2021_val": ("ytvis_2021/train/JPEGImages",
                       "split_dataset/YouTube-2021/test_split.json"),
}


# ====    Predefined splits for OVIS    ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train/JPEGImages", ##########################
                   "split_dataset/OVIS/train_split.json"),  #####################

    "ovis_test": ("ovis/test/JPEGImages",   ###############
                  "ovis/test.json"),   ################
}


_PREDEFINED_SPLITS_OVIS_val = {
    "ovis_val": ("ovis/train/JPEGImages",  #####################
                 "split_dataset/OVIS/test_split.json"),  ###########
}

_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/annotations/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/annotations/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/annotations/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/annotations/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/annotations/coco2ovis_val.json"),
}



# ==== Predefined splits for YTVIS 2019 ===========
_MEM_PREDEFINED_SPLITS_YTVIS_2019 = {
    "mem_ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "split_dataset/YouTube-2019/train_split.json"),
    "mem_ytvis_2019_val": ("ytvis_2019/train/JPEGImages",
                       "split_dataset/YouTube-2019/test_split.json"),
    "mem_ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
    "mem_ytvis_2019_val_all_frames": ("ytvis_2019/valid_all_frames/JPEGImages",
                        "ytvis_2019/valid_all_frames.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_MEM_PREDEFINED_SPLITS_YTVIS_2021 = {
    "mem_ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "split_dataset/YouTube-2021/train_split.json"),
    "mem_ytvis_2021_val": ("ytvis_2021/train/JPEGImages",
                       "split_dataset/YouTube-2021/test_split.json"),
    "mem_ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}


# ====    Predefined splits for OVIS    ===========
_MEM_PREDEFINED_SPLITS_OVIS = {
    "mem_ovis_train": ("ovis/train/JPEGImages",
                   "split_dataset/OVIS/train_split.json"),
    "mem_ovis_val": ("ovis/train/JPEGImages",
                 "split_dataset/OVIS/test_split.json"),
    "mem_ovis_test": ("ovis/test/JPEGImages",
                  "ovis/test.json"),
}


_MEM_PREDEFINED_SPLITS_COCO_VIDEO = {
    "mem_coco2ytvis2019_train": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
    "mem_coco2ytvis2019_val": ("coco/val2017", "coco/annotations/coco2ytvis2019_val.json"),
    "mem_coco2ytvis2021_train": ("coco/train2017", "coco/annotations/coco2ytvis2021_train.json"),
    "mem_coco2ytvis2021_val": ("coco/val2017", "coco/annotations/coco2ytvis2021_val.json"),
    "mem_coco2ovis_train": ("coco/train2017", "coco/annotations/coco2ovis_train.json"),
    "mem_coco2ovis_val": ("coco/val2017", "coco/annotations/coco2ovis_val.json"),
}




def register_all_ytvis_2019(root, cfg, train):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )

def register_all_ytvis_2019_val(root, cfg, train):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019_val.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances_val(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )

def register_all_ytvis_2021(root, cfg, train):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )

def register_all_ytvis_2021_val(root, cfg, train):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021_val.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances_val(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )

def register_all_coco_video(root, cfg):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
        )


def register_all_ovis(root, cfg, train):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )

def register_all_ovis_val(root, cfg, train):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS_val.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances_val(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
            train,
        )


def register_all_ytvis_2019_mem(root, cfg):
    for key, (image_root, json_file) in _MEM_PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances_mem(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,

        )


def register_all_ytvis_2021_mem(root, cfg):
    for key, (image_root, json_file) in _MEM_PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances_mem(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,

        )


def register_all_coco_video_mem(root, cfg):
    for key, (image_root, json_file) in _MEM_PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances_mem(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,
        )


def register_all_ovis_mem(root, cfg):
    for key, (image_root, json_file) in _MEM_PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances_mem(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            cfg,

        )




#if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    #_root = os.getenv("DETECTRON2_DATASETS", "datasets")
    #register_all_ovis(_root, cfg)
    #register_all_ytvis_2019(_root, cfg)
    #register_all_ytvis_2021(_root, cfg)
    #register_all_coco_video(_root, cfg)
