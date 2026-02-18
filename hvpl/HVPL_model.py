from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from .modeling.vita_criterion import VitaSetCriterion
from .modeling.vita_matcher import VitaHungarianMatcher
from .modeling.transformer_decoder.vita import VITA


@META_ARCH_REGISTRY.register()
class HVPL(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        test_topk_per_image: int,
        vita_module: nn.Module,
        vita_criterion: nn.Module,
        num_frames: int,
        num_classes: int,
        is_multi_cls: bool,
        apply_cls_thres: float,
        freeze_detector: bool,
        test_run_chunk_size: int,
        test_interpolate_chunk_size: int,
        is_coco: bool,
        # prompt tuning
        num_prompts: int = 0,
        backbone_freeze: bool = False,
        cls_head_freeze: bool = False,
        mask_head_freeze: bool = False,
        pixel_decoder_freeze: bool = False,
        query_embed_freeze: bool = False,
        trans_decoder_freeze: bool = False,
        prompt_mask_mlp: bool = False,
        prompt_no_obj_mlp: bool = False,
        softmask: bool = False,
        softcls: bool = True,
        task,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.test_topk_per_image = test_topk_per_image

        # vita hyper-parameters
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.vita_module = vita_module
        self.vita_criterion = vita_criterion
        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres

        if freeze_detector:
            for name, p in self.named_parameters():
                if not "vita_module" in name:
                    p.requires_grad_(False)
        self.test_run_chunk_size = test_run_chunk_size
        self.test_interpolate_chunk_size = test_interpolate_chunk_size

        self.is_coco = is_coco

        self.model_old = False
        self.num_prompts = num_prompts
        self.backbone_freeze = backbone_freeze
        self.cls_head_freeze = cls_head_freeze
        self.mask_head_freeze = mask_head_freeze
        self.pixel_decoder_freeze = pixel_decoder_freeze
        self.query_embed_freeze = query_embed_freeze
        self.trans_decoder_freeze = trans_decoder_freeze
        self.prompt_mask_mlp = prompt_mask_mlp
        self.prompt_no_obj_mlp = prompt_no_obj_mlp
        self.softmask = softmask
        self.softcls = softcls
        self.task = task

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        vita_deep_supervision = cfg.MODEL.VITA.DEEP_SUPERVISION

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        sim_weight = cfg.MODEL.VITA.SIM_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            vita_last_layer_num=cfg.MODEL.VITA.LAST_LAYER_NUM,
            focal=cfg.MODEL.MASK_FORMER.FOCAL,
            focal_alpha=cfg.MODEL.MASK_FORMER.FOCAL_ALPHA, focal_gamma=cfg.MODEL.MASK_FORMER.FOCAL_GAMMA,
        )

        # Vita
        num_classes = sem_seg_head.num_classes
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        vita_module = VITA(cfg=cfg, in_channels=hidden_dim, aux_loss=vita_deep_supervision)

        # building criterion for vita inference
        vita_matcher = VitaHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        vita_weight_dict = {
            "loss_vita_ce": class_weight, "loss_vita_mask": mask_weight, "loss_vita_dice": dice_weight
        }
        if sim_weight > 0.0:
            vita_weight_dict["loss_vita_sim"] = sim_weight

        if vita_deep_supervision:
            vita_dec_layers = cfg.MODEL.VITA.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(vita_dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in vita_weight_dict.items()})
            vita_weight_dict.update(aux_weight_dict)
        vita_losses = ["vita_labels", "vita_masks"]
        if sim_weight > 0.0:
            vita_losses.append("fg_sim")

        vita_criterion = VitaSetCriterion(
            num_classes, 
            matcher=vita_matcher, 
            weight_dict=vita_weight_dict,
            eos_coef=cfg.MODEL.VITA.NO_OBJECT_WEIGHT,
            losses=vita_losses, 
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            sim_use_clip=cfg.MODEL.VITA.SIM_USE_CLIP,
            focal=cfg.MODEL.MASK_FORMER.FOCAL,
            focal_alpha=cfg.MODEL.MASK_FORMER.FOCAL_ALPHA, focal_gamma=cfg.MODEL.MASK_FORMER.FOCAL_GAMMA,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.VITA.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # vita
            "vita_module": vita_module,
            "vita_criterion": vita_criterion,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,

            "is_multi_cls": cfg.MODEL.VITA.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.VITA.APPLY_CLS_THRES,
            "freeze_detector": cfg.MODEL.VITA.FREEZE_DETECTOR,
            "test_run_chunk_size": cfg.MODEL.VITA.TEST_RUN_CHUNK_SIZE,
            "test_interpolate_chunk_size": cfg.MODEL.VITA.TEST_INTERPOLATE_CHUNK_SIZE,
            "is_coco": cfg.DATASETS.TEST[0].startswith("coco"),
            # prompt tuning
            "num_prompts": cfg.CONT.NUM_PROMPTS,
            "backbone_freeze": cfg.CONT.BACKBONE_FREEZE,
            "cls_head_freeze": cfg.CONT.CLS_HEAD_FREEZE,
            "mask_head_freeze": cfg.CONT.MASK_HEAD_FREEZE,
            "pixel_decoder_freeze": cfg.CONT.PIXEL_DECODER_FREEZE,
            "query_embed_freeze": cfg.CONT.QUERY_EMBED_FREEZE,
            "trans_decoder_freeze": cfg.CONT.TRANS_DECODER_FREEZE,
            "prompt_mask_mlp": cfg.CONT.PROMPT_MASK_MLP,
            "prompt_no_obj_mlp": cfg.CONT.PROMPT_NO_OBJ_MLP,
            "softmask": cfg.MODEL.MASK_FORMER.SOFTMASK,
            "softcls": cfg.CONT.SOFTCLS,
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            return self.train_model(batched_inputs)
        else:
            # NOTE consider only B=1 case.
            if self.task == 'vps':
                return self.inference_VPS(batched_inputs[0])
            else:
                return self.inference(batched_inputs[0])

    def train_model(self, batched_inputs):
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)


        BT = len(images)
        T = self.num_frames if self.training else BT 
        B = BT // T

        outputs, frame_queries, mask_features = self.sem_seg_head(features)

        mask_features = self.vita_module.vita_mask_features(mask_features)
        mask_features = mask_features.view(B, self.num_frames, *mask_features.shape[-3:])

        # mask classification target
        frame_targets, clip_targets = self.prepare_targets(batched_inputs, images)

        # bipartite matching-based loss
        losses, fg_indices = self.criterion(outputs, frame_targets)

        vita_outputs = self.vita_module(frame_queries)
        vita_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", vita_outputs["pred_mask_embed"], mask_features)
        for out in vita_outputs["aux_outputs"]:
            out["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", out["pred_mask_embed"], mask_features)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        vita_loss_dict = self.vita_criterion(vita_outputs, clip_targets, frame_targets, fg_indices)
        vita_weight_dict = self.vita_criterion.weight_dict

        for k in vita_loss_dict.keys():
            if k in vita_weight_dict:
                vita_loss_dict[k] *= vita_weight_dict[k]
        losses.update(vita_loss_dict)
        return losses

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        frame_gt_instances = []
        clip_gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_classes_per_video = targets_per_video["instances"][0].gt_classes.to(self.device)
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else: #polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2,3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long() # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()         # N, num_frames
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()    # N, num_frames, H, W
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id

            clip_gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video, "masks": gt_masks_per_video,
                    "video_len": targets_per_video["video_len"], "frame_idx": targets_per_video["frame_idx"],
                }
            )

            for f_i in range(self.num_frames):
                _cls = gt_classes_per_video.clone()
                _ids = gt_ids_per_video[:, f_i].clone()
                _mask = gt_masks_per_video[:, f_i].clone()

                valid = _ids != -1
                frame_gt_instances.append({
                    "labels": _cls[valid],
                    "ids": _ids[valid],
                    "masks": _mask[valid],
                })

        return frame_gt_instances, clip_gt_instances

    def inference(self, batched_inputs):
        frame_queries, mask_features = [], []
        num_frames = len(batched_inputs["image"])
        to_store = self.device if num_frames <= 36 else "cpu"

        for i in range(math.ceil(num_frames / self.test_run_chunk_size)):
            images = batched_inputs["image"][i*self.test_run_chunk_size : (i+1)*self.test_run_chunk_size]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)
            outputs, _frame_queries, _mask_features = self.sem_seg_head(features)

            _mask_features = self.vita_module.vita_mask_features(_mask_features)

            # BT is 1 as runs per frame
            frame_queries.append(_frame_queries[-1])    # T', fQ, C
            mask_features.append(_mask_features.to(to_store))  # T', C, H, W

        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation

        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs.get("width", image_size[1])

        del outputs, images, batched_inputs

        frame_queries = torch.cat(frame_queries)[None]  # 1, T, fQ, C
        mask_features = torch.cat(mask_features)        # T, C, H, W

        vita_outputs = self.vita_module(frame_queries)

        mask_cls = vita_outputs["pred_logits"][-1, 0]         # cQ, K+1
        mask_embed = vita_outputs["pred_mask_embed"][-1, 0]      # cQ, C

        del vita_outputs

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]  # if self.softcls else mask_cls.sigmoid()[:, :-1]  #####self.num_queries
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(mask_embed.shape[0], 1).flatten(0, 1)
        num_topk = self.test_topk_per_image
        scores_per_video, topk_indices = scores.flatten(0, 1).topk(num_topk, sorted=False)
        labels_per_video = labels[topk_indices]
        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')
        mask_embed = mask_embed[topk_indices]

        masks_per_video = []
        numerator = torch.zeros(len(mask_embed), dtype=torch.float, device=self.device)
        denominator = torch.zeros(len(mask_embed), dtype=torch.float, device=self.device)

        for i in range(math.ceil(len(mask_features) / self.test_interpolate_chunk_size)):
            m_f = mask_features[i*self.test_interpolate_chunk_size : (i+1)*self.test_interpolate_chunk_size].to(self.device)

            mask_pred = torch.einsum("qc,tchw->qthw", mask_embed, m_f)

            # upsample masks
            mask_pred = retry_if_cuda_oom(F.interpolate)(
                mask_pred,
                size=interim_size,
                mode="bilinear",
                align_corners=False,
            ) # cQ, T, H, W

            mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

            if not self.softmask:
                interim_mask_soft = mask_pred.sigmoid()
            else:
                interim_mask_soft = mask_pred

            interim_mask_hard = interim_mask_soft > 0.5
            numerator += (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)

            denominator += interim_mask_hard.flatten(1).sum(1)

            mask_pred = F.interpolate(
                mask_pred, size=(out_height, out_width), mode="bilinear", align_corners=False
            ) > 0.
            masks_per_video.append(mask_pred.to(to_store))
        masks_per_video = torch.cat(masks_per_video, dim=1)
        scores_per_video *= (numerator / (denominator + 1e-6))

        if self.is_coco:
            result = Instances((out_height, out_width))

            result.pred_masks = masks_per_video[:, 0].float()   # T=1 for COCO
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            
            result.scores = scores_per_video
            result.pred_classes = labels_per_video

            processed_results = [{"instances": result}]
        else:
            processed_results = {
                "image_size": (out_height, out_width),
                "pred_scores": scores_per_video.tolist(),
                "pred_labels": labels_per_video.tolist(),
                "pred_masks": masks_per_video.cpu(),
            }

        return processed_results

    def inference_VPS(self, batched_inputs):
        frame_queries, mask_features = [], []
        num_frames = len(batched_inputs["image"])
        to_store = self.device if num_frames <= 36 else "cpu"

        for i in range(math.ceil(num_frames / self.test_run_chunk_size)):
            images = batched_inputs["image"][i*self.test_run_chunk_size : (i+1)*self.test_run_chunk_size]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)
            outputs, _frame_queries, _mask_features = self.sem_seg_head(features)

            _mask_features = self.vita_module.vita_mask_features(_mask_features)

            # BT is 1 as runs per frame
            frame_queries.append(_frame_queries[-1])    # T', fQ, C
            mask_features.append(_mask_features.to(to_store))  # T', C, H, W

        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation

        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs.get("width", image_size[1])

        del outputs, images, batched_inputs

        frame_queries = torch.cat(frame_queries)[None]  # 1, T, fQ, C
        mask_features = torch.cat(mask_features)        # T, C, H, W

        vita_outputs = self.vita_module(frame_queries)

        mask_cls = vita_outputs["pred_logits"][-1, 0]         # cQ, K+1    ....
        mask_embed = vita_outputs["pred_mask_embed"][-1, 0]      # cQ, C

        del vita_outputs
        scores_1 = F.softmax(mask_cls, dim=-1)[:, :-1]

        pred_masks = []
        for i in range(math.ceil(len(mask_features) / self.test_interpolate_chunk_size)):
            m_f = mask_features[i*self.test_interpolate_chunk_size : (i+1)*self.test_interpolate_chunk_size].to(self.device)

            mask_pred_1 = torch.einsum("qc,tchw->qthw", mask_embed, m_f)

            cur_masks = retry_if_cuda_oom(F.interpolate)(
                    mask_pred_1,
                    size=interim_size,
                    mode="bilinear",
                    align_corners=False,
            ) # cQ, T, H, W

            cur_masks = cur_masks[:, :, : image_size[0], : image_size[1]].sigmoid() 

            cur_masks = F.interpolate(
                    cur_masks, size=(out_height, out_width), mode="bilinear", align_corners=False)


            pred_masks.append(cur_masks.to(to_store))
        pred_masks = torch.cat(pred_masks, dim=1)


        pred_id = [torch.arange(0, mask_embed.size(0))][0] ########################
        mask_pred = pred_masks
        scores, labels = scores_1.max(-1)
        # filter out the background prediction
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_ids = pred_id[keep]
        cur_masks = mask_pred[keep]


        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks


        # initial panoptic_seg and segments infos
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((cur_masks.size(1), h, w), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return {
                "image_size": (out_height, out_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = len(self.metadata.stuff_dataset_id_to_contiguous_id) <= pred_class < (len(self.metadata.thing_dataset_id_to_contiguous_id) + len(self.metadata.stuff_dataset_id_to_contiguous_id))
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_infos.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
                    out_ids.append(cur_ids[k])

            return {
                "image_size": (out_height, out_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }

    def freeze_for_prompt_tuning(self, ):
        if not self.model_old:

            for k, p in self.named_parameters():
                p.requires_grad = False

            # unfreeze last classifier layer
            for k, p in self.sem_seg_head.predictor.class_embed.cls[-1].named_parameters():
                p.requires_grad = True

            # unfreeze cls-layer for no-obj
            for k, p in self.sem_seg_head.predictor.class_embed.cls[0].named_parameters():
                p.requires_grad = True

            # unfreeze last classifier layer
            for k, p in self.vita_module.class_embed.cls[-1].named_parameters():
                p.requires_grad = True

            # unfreeze cls-layer for no-obj
            for k, p in self.vita_module.class_embed.cls[0].named_parameters():
                p.requires_grad = True

            if self.num_prompts > 0:
                # unfreeze prompt embeddings
                for k, p in self.sem_seg_head.predictor.prompt_feat[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_embed[-1].named_parameters():
                    p.requires_grad = True

                # unfreeze prompt embeddings
                for k, p in self.vita_module.prompt_feat[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.vita_module.prompt_embed[-1].named_parameters():
                    p.requires_grad = True

                for k, p in self.vita_module.prompt_fq_pos[-1].named_parameters(): 
                    p.requires_grad = True


            if not self.backbone_freeze:
                for k, p in self.backbone.named_parameters():
                    p.requires_grad = True

            if not self.trans_decoder_freeze and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_transformer_self_attention_layers[-1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_transformer_cross_attention_layers[
                    -1].named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.prompt_transformer_ffn_layers[-1].named_parameters():
                    p.requires_grad = True

            if not self.pixel_decoder_freeze:
                for k, p in self.sem_seg_head.pixel_decoder.named_parameters():
                    p.requires_grad = True

            if not self.cls_head_freeze:
                for k, p in self.sem_seg_head.predictor.class_embed.cls.named_parameters():
                    p.requires_grad = True

            if not self.mask_head_freeze:
                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = True

            if not self.query_embed_freeze:
                for k, p in self.sem_seg_head.predictor.query_feat.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.query_embed.named_parameters():
                    p.requires_grad = True

            if self.prompt_mask_mlp and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_mask_embed[-1].named_parameters():
                    p.requires_grad = True

                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = False

            
                for k, p in self.vita_module.prompt_mask_embed[-1].named_parameters():
                    p.requires_grad = True

                for k, p in self.vita_module.mask_embed.named_parameters():
                    p.requires_grad = False

            if self.prompt_no_obj_mlp and self.num_prompts > 0:
                for k, p in self.sem_seg_head.predictor.prompt_no_obj_embed[-1].named_parameters():
                    p.requires_grad = True

                for k, p in self.sem_seg_head.predictor.class_embed.cls[0].named_parameters():
                    p.requires_grad = False

            if self.num_prompts == 0:
                # unfreeze prompt embeddings
                for k, p in self.sem_seg_head.predictor.query_feat.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.query_embed.named_parameters():
                    p.requires_grad = True
                for k, p in self.sem_seg_head.predictor.mask_embed.named_parameters():
                    p.requires_grad = True

    def copy_prompt_embed_weights(self, ):
        if self.num_prompts > 0:
            base_feat_weights = self.sem_seg_head.predictor.query_feat.weight
            base_embed_weights = self.sem_seg_head.predictor.query_embed.weight

            base_feat_weights = base_feat_weights.mean(0, keepdims=True).repeat(self.num_prompts, 1)
            base_embed_weights = base_embed_weights.mean(0, keepdims=True).repeat(self.num_prompts, 1)


            base_feat_weights_vita_module = self.vita_module.query_feat.weight
            base_embed_weights_vita_module = self.vita_module.query_embed.weight

            base_feat_weights_vita_module = base_feat_weights_vita_module.mean(0, keepdims=True).repeat(self.num_prompts, 1)
            base_embed_weights_vita_module = base_embed_weights_vita_module.mean(0, keepdims=True).repeat(self.num_prompts, 1)

            base_pos_weights_vita_module = self.vita_module.fq_pos.weight
            base_pos_weights_vita_module = base_pos_weights_vita_module.mean(0, keepdims=True).repeat(self.num_prompts, 1)
            self.vita_module.prompt_fq_pos[-1].load_state_dict({"weight": base_pos_weights_vita_module})


            self.sem_seg_head.predictor.prompt_feat[-1].load_state_dict({"weight": base_feat_weights})

            self.vita_module.prompt_feat[-1].load_state_dict({"weight": base_feat_weights_vita_module})

            if isinstance(self.sem_seg_head.predictor.prompt_embed[-1], nn.ModuleList):
                embed_dict = {}
                for n in range(len(self.sem_seg_head.predictor.prompt_embed[-1])):
                    embed_dict[f"{n}.weight"] = base_embed_weights
            else:
                embed_dict = {"weight": base_embed_weights}

            self.sem_seg_head.predictor.prompt_embed[-1].load_state_dict(embed_dict)


            if isinstance(self.vita_module.prompt_embed[-1], nn.ModuleList):
                embed_dict_vita_module = {}
                for n in range(len(self.vita_module.prompt_embed[-1])):
                    embed_dict_vita_module[f"{n}.weight"] = base_embed_weights_vita_module
            else:
                embed_dict_vita_module = {"weight": base_embed_weights_vita_module}

            self.vita_module.prompt_embed[-1].load_state_dict(embed_dict_vita_module)

    def copy_mask_embed_weights(self, ):
        if self.num_prompts > 0 and self.prompt_mask_mlp:
            self.sem_seg_head.predictor.prompt_mask_embed[-1].load_state_dict(
                self.sem_seg_head.predictor.mask_embed.state_dict()
            )

        if self.num_prompts > 0 and self.prompt_mask_mlp:
            self.vita_module.prompt_mask_embed[-1].load_state_dict(
                self.vita_module.mask_embed.state_dict()
            )

    def copy_prompt_trans_decoder_weights(self, ):
        if self.num_prompts > 0 and not self.trans_decoder_freeze:
            self.sem_seg_head.predictor.prompt_transformer_self_attention_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_self_attention_layers.state_dict()
            )
            self.sem_seg_head.predictor.prompt_transformer_cross_attention_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_cross_attention_layers.state_dict()
            )
            self.sem_seg_head.predictor.prompt_transformer_ffn_layers[-1].load_state_dict(
                self.sem_seg_head.predictor.transformer_ffn_layers.state_dict()
            )

            self.vita_module.prompt_transformer_self_attention_layers[-1].load_state_dict(
                self.vita_module.transformer_self_attention_layers.state_dict()
            )
            self.vita_module.prompt_transformer_cross_attention_layers[-1].load_state_dict(
                self.vita_module.transformer_cross_attention_layers.state_dict()
            )
            self.vita_module.prompt_transformer_ffn_layers[-1].load_state_dict(
                self.vita_module.transformer_ffn_layers.state_dict()
            )

    def copy_no_obj_weights(self, ):
        if self.num_prompts > 0:
            no_obj_weights = self.sem_seg_head.predictor.class_embed.cls[0].state_dict()
            novel_cls_weights = self.sem_seg_head.predictor.class_embed.cls[-1].state_dict()

            no_obj_weights_vita_module = self.vita_module.class_embed.cls[0].state_dict()
            novel_cls_weights_vita_module = self.vita_module.class_embed.cls[-1].state_dict()

            for k, v in no_obj_weights.items():
                if v.shape == novel_cls_weights[k].shape:
                    novel_cls_weights[k] = v
                else:
                    if "weight" in k:
                        novel_cls_weights[k] = v.repeat(novel_cls_weights[k].shape[0], 1)
                    elif "bias" in k:
                        novel_cls_weights[k] = v.repeat(novel_cls_weights[k].shape[0])

            self.sem_seg_head.predictor.class_embed.cls[-1].load_state_dict(novel_cls_weights)

            for k, v in no_obj_weights_vita_module.items():
                if v.shape == novel_cls_weights_vita_module[k].shape:
                    novel_cls_weights_vita_module[k] = v
                else:
                    if "weight" in k:
                        novel_cls_weights_vita_module[k] = v.repeat(novel_cls_weights_vita_module[k].shape[0], 1)
                    elif "bias" in k:
                        novel_cls_weights_vita_module[k] = v.repeat(novel_cls_weights_vita_module[k].shape[0])

            self.vita_module.class_embed.cls[-1].load_state_dict(novel_cls_weights_vita_module)
