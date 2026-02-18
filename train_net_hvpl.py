
import copy
import itertools
import logging
import os
import warnings
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Set
import torch

warnings.filterwarnings("ignore")

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)
from detectron2.engine import (
    DefaultTrainer,
    TrainerBase,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model, default_writers
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from fvcore.nn.precise_bn import get_bn_modules


from mask2former import add_maskformer2_config
from continual import add_continual_config

from ogc.hook_grad import AMPTrainer_grad, SimpleTrainer_grad
from ogc.feature_data_for_grad import get_feature_for_grad

from hvpl import (
    CocoClipDatasetMapper,
    YTVISDatasetMapper,
    YTVISEvaluator,
    add_vita_config,
    build_combined_loader,
    build_detection_test_loader,
    build_detection_train_loader,
)
from hvpl.data.datasets.builtin import (
    register_all_coco_video,
    register_all_ovis,
    register_all_ovis_val,
    register_all_ytvis_2019,
    register_all_ytvis_2019_val,
    register_all_ytvis_2021,
    register_all_ytvis_2021_val,
)


class Trainer(TrainerBase):
    """
    Extension of the Trainer class adapted to HVPL.

    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        #For OGC
        feature_prefix_gt = get_feature_for_grad(cfg) if (cfg.CONT.OGC and cfg.CONT.TASK > 0) else None
        

        model = self.build_model(cfg)

        self.optimizer = optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        self.model = model = create_ddp_model(model, broadcast_buffers=False)

        #For OGC
        if cfg.CONT.OGC and cfg.CONT.TASK > 0:
            feature_prefix_gt[0]['key'] = feature_prefix_gt[0]['key'].to(self.model.device)


        TrainerClass = AMPTrainer_grad if cfg.SOLVER.AMP.ENABLED else SimpleTrainer_grad


        self._trainer = TrainerClass(
            model, 
            data_loader, 
            optimizer, 
            cfg, 
            feature_prefix_gt=feature_prefix_gt
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

        self._last_eval_results = None


    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1


        """
        ECLIPSE: initialization
        """
        if self.cfg.CONT.NUM_PROMPTS > 0:
            self.model.module.copy_prompt_embed_weights()
            self.model.module.copy_mask_embed_weights()
            self.model.module.copy_no_obj_weights()
            self.model.module.copy_prompt_trans_decoder_weights()


    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)


    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results


    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        elif evaluator_type == "ytvis":
            evaluator_list.append(YTVISEvaluator(dataset_name, cfg, True, output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        else:
            raise NotImplementedError


    @classmethod
    def build_train_loader(cls, cfg):

        try:
            _root_data = os.getenv("DETECTRON2_DATASETS", "datasets")
            register_all_ovis(_root_data, cfg, train=True)
            register_all_ytvis_2019(_root_data, cfg, train=True)
            register_all_ytvis_2021(_root_data, cfg, train=True)
            register_all_coco_video(_root_data, cfg)
        except:
            pass

        mappers = []
        for d_i, dataset_name in enumerate(cfg.DATASETS.TRAIN):
            print(dataset_name)
            if dataset_name.startswith('coco'):
                mappers.append(
                    CocoClipDatasetMapper(
                        cfg, is_train=True, is_tgt=(d_i==len(cfg.DATASETS.TRAIN)-1), src_dataset_name=dataset_name
                    )
                )
            elif dataset_name.startswith('ytvis') or dataset_name.startswith('ovis'):
                mappers.append(
                    YTVISDatasetMapper(cfg, is_train=True, is_tgt=(d_i==len(cfg.DATASETS.TRAIN)-1), src_dataset_name=dataset_name)
                )
            else:
                raise NotImplementedError
        assert len(mappers) > 0, "No dataset is chosen!"

        if len(mappers) == 1:
            mapper = mappers[0]
            return build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])
        else:

            loaders = []
            for mapper, dataset_name in zip(mappers, cfg.DATASETS.TRAIN):
                dataset = DatasetCatalog.get(dataset_name)
                if len(dataset) == 0:
                    print(dataset_name, 'is empty')
                else:
                    loader = build_detection_train_loader(cfg, mapper=mapper, dataset_name=dataset_name)
                    loaders.append(loader)

            if len(loaders) > 1:
                combined_data_loader = build_combined_loader(cfg, loaders, cfg.DATASETS.DATASET_RATIO)
                return combined_data_loader

            else:
                return loader


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):

        try:
            _root_data = os.getenv("DETECTRON2_DATASETS", "datasets")
            register_all_ytvis_2019_val(_root_data, cfg, train=False)
            register_all_ytvis_2021_val(_root_data, cfg, train=False)
            register_all_ovis_val(_root_data, cfg, train=False)
        except:
            pass

        dataset_name = cfg.DATASETS.TEST[0]
        if dataset_name.startswith('coco'):
            mapper = CocoClipDatasetMapper(cfg, is_train=False)
        elif dataset_name.startswith('ytvis') or dataset_name.startswith('ovis'):
            mapper = YTVISDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)


    @classmethod
    def build_model(cls, cfg, old=False):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """

        model = build_model(cfg)

        if not old:
            logger = logging.getLogger(__name__)
            logger.info("Model:\n{}".format(model))

            if (cfg.CONT.TASK > 0 and cfg.CONT.NUM_PROMPTS > 0 and cfg.CONT.BACKBONE_FREEZE):
                logger.info("Model Freezing for Visual Prompt Tuning")
                model.freeze_for_prompt_tuning()

        return model


    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        logger = logging.getLogger(__name__)
        unfreeze_params = []
        freeze_params = []

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    freeze_params.append(f"{module_name}.{module_param_name}")
                    continue

                unfreeze_params.append(f"{module_name}.{module_param_name}")
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})


        if comm.is_main_process():
            logger.info("\n============ Frozen Parameters ============")
            for param in freeze_params:
                logger.info(f"  {param}")
                
            logger.info("\n============ Unfrozen Parameters ============")
            for param in unfreeze_params:
                logger.info(f"  {param}")

            trainable_num = 0
            trainable_mem = 0
            total_num = 0
            total_mem = 0

            for p in model.parameters():
                numel = p.numel()
                mem = numel * p.element_size()
                total_num += numel
                total_mem += mem
                if p.requires_grad:
                    trainable_num += numel
                    trainable_mem += mem
            
            logger.info("\n" + "="*40)
            logger.info(f"{'Model Statistics':^40}")
            logger.info("="*40)
            logger.info(f"Trainable Params: {trainable_num / 1e6:.2f} M")
            logger.info(f"Trainable Memory: {trainable_mem / (1024**2):.2f} MB")
            logger.info("-" * 40)
            logger.info(f"Total Params:     {total_num / 1e6:.2f} M")
            logger.info(f"Total Memory:     {total_mem / (1024**2):.2f} MB")
            logger.info("="*40 + "\n")

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.

        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

        For example, with the original config like the following:

        .. code-block:: yaml

            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000

        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:

        .. code-block:: yaml

            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500

        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).

        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg


    def state_dict(self):
        ret = super().state_dict()
        ret['trainer'] = self._trainer.state_dict()
        return ret


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict['trainer'])


def setup(args):
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

    cfg.DATASETS.TRAIN_MEM = tuple(f"mem_{dataset}" for dataset in cfg.DATASETS.TRAIN)  
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS

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
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    cfg = setup(args)



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


    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":

    args = default_argument_parser().parse_args()


    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )













