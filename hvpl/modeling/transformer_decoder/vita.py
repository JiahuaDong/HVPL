from math import ceil
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d
#############################################
from copy import deepcopy
from continual.modeling import IncrementalClassifier, CosineClassifier, IncrementalClassifier_Video
from typing import Optional, List
import numpy as np

##############################################
from .grootv import GrootVLayer ########################


def sigmoid_to_logit(x):
    x = x.clamp(0.001, 0.999)
    return torch.log(x / (1-x))

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class VITA(nn.Module):

    @configurable
    def __init__(
        self,
        in_channels,
        aux_loss,
        *,
        hidden_dim: int,
        num_frame_queries: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        enc_layers: int,
        dec_layers: int,
        enc_window_size: int,
        pre_norm: bool,
        enforce_input_project: bool,
        num_frames: int,
        num_classes: int,
        clip_last_layer_num: bool,
        conv_dim: int,
        mask_dim: int,
        sim_use_clip: list,
        use_sim: bool,
        ###########################################
        num_prompts: int,
        prompt_deep: bool = False,
        softmask: bool = False,
        inc_query: Optional[bool] = None,
        cosine: Optional[bool] = False,
        bias: Optional[bool] = False,
        classes: Optional[List[int]] = None,
        prompt_mask_mlp: Optional[bool] = False,
        prompt_no_obj_mlp: Optional[bool] = False,
        deep_cls: Optional[bool] = False,
        deltas: Optional[List[float]] = None,
        num_prompts_query: int,
        Mamba = True,


    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.clip_last_layer_num = clip_last_layer_num

        self.enc_layers = enc_layers
        self.window_size = enc_window_size
        self.sim_use_clip = sim_use_clip
        self.use_sim = use_sim
        self.aux_loss = aux_loss

        self.enc_layers = enc_layers

        self.Mamba = Mamba  ################################



        if enc_layers > 0:

            if self.Mamba:
                dpr = [
                    x.item() for x in torch.linspace(0, 0.2, 6)
                ]
                self.enc_mamba = nn.ModuleList([
                GrootVLayer(
                    channels=256,
                    mlp_ratio=4.0,
                    drop=0.0,
                    drop_path=dpr[i],
                    act_layer="GELU",
                    norm_layer="LN",
                    post_norm= False,
                    layer_scale=None,
                    with_cp= False,
                ) for i in range(enc_layers)
                ])
            else:
                self.enc_self_attn = nn.ModuleList()
                self.enc_ffn = nn.ModuleList()
                for _ in range(self.enc_layers):
                    self.enc_self_attn.append(
                        SelfAttentionLayer(
                            d_model=hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        ),
                    )
                    self.enc_ffn.append(
                        FFNLayer(
                            d_model=hidden_dim,
                            dim_feedforward=dim_feedforward,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        )
                    )


        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.vita_mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.vita_mask_features)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.fq_pos = nn.Embedding(num_frame_queries, hidden_dim)

        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj_dec = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.input_proj_dec = nn.Sequential()
        self.src_embed = nn.Identity()


        #self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        ##################################################################################
        ######### new_paraameters
        self.num_prompts = num_prompts   ###### 100-50  is  50
        self.prompt_deep = prompt_deep and self.num_prompts > 0  ##True
        self.prompt_mask_mlp = prompt_mask_mlp and self.num_prompts > 0 ###True
        self.prompt_no_obj_mlp = prompt_no_obj_mlp and self.num_prompts > 0  ### False


        self.deltas = deltas  # [0.6 , -0.6] seclect
        if self.deltas is None:
            self.deltas = [0.0 for _ in classes]
        elif type(self.deltas) == float:
            self.deltas = [self.deltas for _ in classes]
        elif len(self.deltas) > len(classes):
            self.deltas = self.deltas[:len(classes)]
        elif len(self.deltas) < len(classes):
            self.deltas = self.deltas + [self.deltas[-1] for _ in range(len(classes) - len(self.deltas))]

        assert len(self.deltas) == len(classes), "CONT."

        self.old_model = False
        self.classes = classes  # [C0, C1, ..., Cn]

        #self.num_layers = num_layers

        # prompt embeddings
        if self.num_prompts > 0:

            self.prompt_feat = nn.ModuleList(   ########
                [nn.Embedding(num_prompts, hidden_dim) for _ in classes[1:]]  #######
            )

            if self.prompt_deep:  ### True
                self.prompt_embed = nn.ModuleList(
                    [
                        nn.ModuleList(
                            [nn.Embedding(num_prompts, hidden_dim) for _ in range(self.num_layers)]
                        ) for _ in classes[1:]
                    ]
                )

            else:
                self.prompt_embed = nn.ModuleList(
                    [nn.Embedding(num_prompts, hidden_dim) for _ in classes[1:]]
                )


        if classes is not None:
                if cosine:  ### False
                    self.class_embed = CosineClassifier([1] + classes, channels=hidden_dim)
                else:  ##True
                    # [1] : no_obj, (we don't have bkg class)
                    self.class_embed = IncrementalClassifier_Video(
                        [1] + classes,
                        channels=hidden_dim,
                        bias=bias,
                        deep_cls=deep_cls,
                    )
        else:
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        if self.prompt_mask_mlp:  ## True
            self.prompt_mask_embed = nn.ModuleList(
                [deepcopy(self.mask_embed) for _ in classes[1:]]
            )

        if self.prompt_no_obj_mlp:  ## False
            self.prompt_no_obj_embed = nn.ModuleList(
                [MLP(hidden_dim, hidden_dim, 1, 3) for _ in classes[1:]]
            )

        #if self.num_prompts > 0:
            #self.prompt_input_proj_dec = nn.ModuleList(
               # [deepcopy(self.input_proj_dec) for _ in classes[1:]]
            #)

        if self.num_prompts > 0:
            self.prompt_fq_pos = nn.ModuleList(
                [nn.Embedding(num_prompts, hidden_dim) for _ in classes[1:]]
            )


        ####################################################
        self.num_prompts_query = num_prompts_query
        ##################################################################################

        if self.use_sim:
            self.sim_embed_frame = nn.Linear(hidden_dim, hidden_dim)
            #if self.num_prompts > 0:
            # self.prompt_sim_embed_frame = nn.ModuleList(
            # [deepcopy(self.sim_embed_frame) for _ in classes[1:]]
            # )

            if self.sim_use_clip:
                self.sim_embed_clip = nn.Linear(hidden_dim, hidden_dim)
                #if self.num_prompts > 0:
                #   self.prompt_sim_embed_clip = nn.ModuleList(
                #   [deepcopy(self.sim_embed_clip) for _ in classes[1:]]
                #    )

    @classmethod
    def from_config(cls, cfg, in_channels):
        ret = {}
        ret["in_channels"] = in_channels

        if hasattr(cfg, "CONT"):
            ret['inc_query'] = cfg.CONT.INC_QUERY
            ret["classes"] = [cfg.CONT.BASE_CLS] + cfg.CONT.TASK * [cfg.CONT.INC_CLS]
            ret["num_classes"] = cfg.CONT.BASE_CLS + cfg.CONT.TASK * cfg.CONT.INC_CLS
            ret["cosine"] = cfg.CONT.COSINE
            ret["bias"] = cfg.CONT.USE_BIAS
            if cfg.MODEL.MASK_FORMER.TEST.MASK_BG and (cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON or
                                                       cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON or
                                                       cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON):
                ret["num_classes"] += 1
                ret["classes"][0] += 1

            # Parameters for ECLIPSE
            ret['num_prompts'] = cfg.CONT.NUM_PROMPTS
            ret['prompt_deep'] = cfg.CONT.PROMPT_DEEP
            ret['prompt_mask_mlp'] = cfg.CONT.PROMPT_MASK_MLP
            ret['prompt_no_obj_mlp'] = cfg.CONT.PROMPT_NO_OBJ_MLP
            ret['deltas'] = cfg.CONT.LOGIT_MANI_DELTAS
            ret['deep_cls'] = cfg.CONT.DEEP_CLS
            #########################################
            ret['num_prompts_query'] = cfg.MODEL.VITA.NUM_OBJECT_QUERIES + cfg.CONT.TASK * cfg.CONT.NUM_PROMPTS

        else:
            ret['inc_query'] = None
            ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
            if not cfg.MODEL.MASK_FORMER.TEST.MASK_BG and (cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON or
                                                           cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON):
                ret["num_classes"] -= 1

        ret["hidden_dim"] = cfg.MODEL.VITA.HIDDEN_DIM
        ret["num_frame_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        ret["num_queries"] = cfg.MODEL.VITA.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.VITA.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.VITA.DIM_FEEDFORWARD

        assert cfg.MODEL.VITA.DEC_LAYERS >= 1
        ret["enc_layers"] = cfg.MODEL.VITA.ENC_LAYERS
        ret["dec_layers"] = cfg.MODEL.VITA.DEC_LAYERS
        ret["enc_window_size"] = cfg.MODEL.VITA.ENC_WINDOW_SIZE
        ret["pre_norm"] = cfg.MODEL.VITA.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.VITA.ENFORCE_INPUT_PROJ

        #ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES  ##############################
        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM
        ret["clip_last_layer_num"] = cfg.MODEL.VITA.LAST_LAYER_NUM

        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["sim_use_clip"] = cfg.MODEL.VITA.SIM_USE_CLIP
        ret["use_sim"] = cfg.MODEL.VITA.SIM_WEIGHT > 0.0

        ret['Mamba'] = cfg.CONT.Mamba

        return ret

    def forward(self, frame_query):
        """
        L: Number of Layers.
        B: Batch size.
        T: Temporal window size. Number of frames per video.
        C: Channel size.
        fQ: Number of frame-wise queries from IFC.
        cQ: Number of clip-wise queries to decode Q.
        """
        if not self.training:
            frame_query = frame_query[[-1]]

        L, BT, fQ, C = frame_query.shape
        B = BT // self.num_frames if self.training else 1
        T = self.num_frames if self.training else BT // B

        frame_query = frame_query.reshape(L*B, T, fQ, C)
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous()
        frame_query = self.input_proj_dec(frame_query) # T, fQ, LB, C



        if not self.training:
            #return self.forward_infer(x, src, pos, size_list, mask_features)

            if not self.Mamba:
                if self.window_size > 0:
                    pad = int(ceil(T / self.window_size)) * self.window_size - T
                    _T = pad + T
                    frame_query = F.pad(frame_query, (0, 0, 0, 0, 0, 0, 0, pad))  # _T, fQ, LB, C
                    enc_mask = frame_query.new_ones(L * B, _T).bool()  # LB, _T
                    enc_mask[:, :T] = False
                else:
                    enc_mask = None

                frame_query = self.encode_frame_query(frame_query, enc_mask)  ########################
                frame_query = frame_query[:T].flatten(0, 1)  # TfQ, LB, C

                
            ###############################################################
            else:
                enc_mask = None
                frame_query = self.encode_frame_query_mamba(frame_query, enc_mask) ########################
                frame_query = frame_query.flatten(0, 1)  # TfQ, LB, C    ##################### [:T]



            if self.use_sim:
                pred_fq_embed = self.sim_embed_frame(frame_query)  # TfQ, LB, C
                pred_fq_embed = pred_fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
            else:
                pred_fq_embed = None

            src = self.src_embed(frame_query)  # TfQ, LB, C

            dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L * B, 1).flatten(0, 1)  # TfQ, LB, C
            if self.num_prompts > 0:
                dec_pos = torch.cat(
                    [
                        dec_pos,
                        torch.cat([p.weight[None, :, None, :].repeat(T, 1, L * B, 1).flatten(0, 1) for p in self.prompt_fq_pos], dim=0)
                    ], dim=0
                )

            # QxNxC
            #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C
            #output = self.query_feat.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C

            if self.num_prompts > 0:
                mask_embeds = nn.ModuleList([self.mask_embed])
                mask_embeds = mask_embeds.extend(self.prompt_mask_embed)
                qdims = np.cumsum([0, self.num_queries] + [self.num_prompts] * len(self.prompt_embed))
            else:
                mask_embeds = self.mask_embed
                qdims = None

            # QxNxC
            self.V_prefix_feature = {}

            output = self.query_feat.weight.unsqueeze(1).repeat(1, L * B, 1)

            if self.num_prompts > 0:
                output = torch.cat(
                    [
                        output,
                        torch.cat([p.weight.unsqueeze(1).repeat(1, L * B, 1) for p in self.prompt_feat], dim=0)
                    ], dim=0
                )


            decoder_outputs = []
            for i in range(self.num_layers):
                # attention: cross-attention first
                query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L * B, 1)

                if self.num_prompts > 0:
                    if self.prompt_deep:
                        query_embed = torch.cat(
                            [
                                query_embed,
                                torch.cat([p[i].weight.unsqueeze(1).repeat(1, L * B, 1) for p in self.prompt_embed], dim=0)
                            ], dim=0
                        )
                    else:
                        query_embed = torch.cat(
                            [
                                query_embed,
                                torch.cat([p.weight.unsqueeze(1).repeat(1, L * B, 1) for p in self.prompt_embed], dim=0)
                            ], dim=0
                        )

                #######################################################################

                if self.num_prompts > 0:
                    self_attn_outputs_cross = torch.zeros_like(output)
                    for qn, qdim in enumerate(qdims[:-1]):
                        self_attn_outputs_cross[qdims[qn]:qdims[qn + 1]] = self.transformer_cross_attention_layers[i](
                        output[qdims[qn]:qdims[qn + 1]], src[qdims[qn]:qdims[qn + 1]],
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=dec_pos[qdims[qn]:qdims[qn + 1]], query_pos=query_embed[qdims[qn]:qdims[qn + 1]]
                        )
                    output = self_attn_outputs_cross

                else:
                #######################################################################
                    output = self.transformer_cross_attention_layers[i](
                        output, src,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=dec_pos, query_pos=query_embed
                    )

                ##################################### grad  #####
                #if i == 0:  ###  0
                    #self.V_prefix_feature[0] = {"key": output}  #####
                ####################################

                if self.num_prompts > 0:
                    self_attn_outputs = torch.zeros_like(output)
                    for qn, qdim in enumerate(qdims[:-1]):   ########### attention its dim..........
                        self_attn_outputs[qdims[qn]:qdims[qn + 1]] = self.transformer_self_attention_layers[i](
                            output[qdims[qn]:qdims[qn + 1]], tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=query_embed[qdims[qn]:qdims[qn + 1]]
                        )
                    output = self_attn_outputs

                else:
                    output = self.transformer_self_attention_layers[i](
                        output, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed,
                    )

                ##################################### grad_self_attention  #####
                if i == 0:
                    self.V_prefix_feature[i] = {"key": output}  #####
                ####################################

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                ##################################### grad  #####
                #if i == 0:
                    #self.V_prefix_feature[i] = {"key": output}  #####
                ####################################

                if (self.training and self.aux_loss) or (i == self.num_layers - 1):
                    dec_out = self.decoder_norm(output)  # cQ, LB, C
                    dec_out = dec_out.transpose(0, 1)  # LB, cQ, C
                    decoder_outputs.append(dec_out.view(L, B, self.num_prompts_query, C))  ###self.num_queries#######


            decoder_outputs = torch.stack(decoder_outputs, dim=0)  # D, L, B, cQ, C

            pred_cls = self.class_embed(decoder_outputs)

            #pred_mask_embed = mask_embeds(decoder_outputs) #############################

            # logit manipulation implementation
            if self.num_prompts > 0 and qdims is not None:  #  >
                pred_mask_embed = []
                for n in range(len(qdims) - 1):
                    pred_mask_embed.append(mask_embeds[n](decoder_outputs[:, :, :, qdims[n]:qdims[n + 1]]))

                    if self.prompt_no_obj_mlp and n > 0:
                        no_obj_logit = self.prompt_no_obj_embed[n - 1](decoder_outputs)
                        pred_cls[:, :, :, qdims[n]:qdims[n + 1], -1] = no_obj_logit[:, :, :, qdims[n]:qdims[n + 1], 0]

                    if self.deltas[n] > 0: # >0
                        # logit manipulation with delta: aggregation of other class knowledge
                        noobj_score = pred_cls[:, :, :, qdims[n]:qdims[n + 1],
                                      list(range(0, sum(self.classes[:n]))) + \
                                      list(range(sum(self.classes[:n + 1]), sum(self.classes)))
                                      ].sigmoid().sum(4).clamp(0., 1.)   ####  .sum(2)

                        pred_cls[:, :, :, qdims[n]:qdims[n + 1], -1] = sigmoid_to_logit(
                            noobj_score * self.deltas[n]
                        )

                    elif self.deltas[n] < 0:  #< 0:
                        # negative delta means calibration the class logits without aggregation of other class knowledge
                        # we empirically found that this strategy is effective when the number of incremental steps is small (e.g., 100-50).
                        pred_cls[:, :, :, qdims[n]:qdims[n + 1], -1] = sigmoid_to_logit(
                            pred_cls[:, :, :, qdims[n]:qdims[n + 1], -1].sigmoid() * -self.deltas[n]
                        )

                    # deactivate other class logits: regarding sigmoid(-10) => 0.0
                    pred_cls[:, :, :, qdims[n]:qdims[n + 1],
                    list(range(0, sum(self.classes[:n]))) + \
                    list(range(sum(self.classes[:n + 1]), sum(self.classes)))
                    ] = -10

                pred_mask_embed = torch.cat(pred_mask_embed, dim=3)    # #### dim=1

            else:
                pred_mask_embed = mask_embeds(decoder_outputs)

                if self.prompt_no_obj_mlp and qdims is not None:
                    for n in range(1, len(qdims) - 1):
                        no_obj_logit = self.prompt_no_obj_embed[n - 1](decoder_outputs)
                        pred_cls[:, :, :, qdims[n]:qdims[n + 1], -1] = no_obj_logit[:, :, :, qdims[n]:qdims[n + 1], 0]


            if self.use_sim and self.sim_use_clip:
                pred_cq_embed = self.sim_embed_clip(decoder_outputs)
            else:
                pred_cq_embed = [None] * self.num_layers

            out = {
                'pred_logits': pred_cls[-1],
                'pred_mask_embed': pred_mask_embed[-1],
                'pred_fq_embed': pred_fq_embed,
                'pred_cq_embed': pred_cq_embed[-1],
                'aux_outputs': self._set_aux_loss(
                    pred_cls, pred_mask_embed, pred_cq_embed, pred_fq_embed
                )
            }
            return out

        else:
            if self.num_prompts > 0 and not self.old_model:
                #return self.forward_new_train(x, src, pos, size_list, mask_features)

                if not self.Mamba:

                    if self.window_size > 0:
                        pad = int(ceil(T / self.window_size)) * self.window_size - T
                        _T = pad + T
                        frame_query = F.pad(frame_query, (0, 0, 0, 0, 0, 0, 0, pad))  # _T, fQ, LB, C
                        enc_mask = frame_query.new_ones(L * B, _T).bool()  # LB, _T
                        enc_mask[:, :T] = False
                    else:
                        enc_mask = None

                    frame_query = self.encode_frame_query(frame_query, enc_mask)
                    frame_query = frame_query[:T].flatten(0, 1)  # TfQ, LB, C

                #############################################################
                else:
                    enc_mask = None
                    frame_query = self.encode_frame_query_mamba(frame_query, enc_mask) ######################
                    frame_query = frame_query.flatten(0, 1)  # TfQ, LB, C   #########[:T]

                #############################################################



                if self.use_sim:
                    pred_fq_embed = self.sim_embed_frame(frame_query)  # TfQ, LB, C
                    pred_fq_embed = pred_fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
                else:
                    pred_fq_embed = None

                src = self.src_embed(frame_query)  # TfQ, LB, C
                dec_pos = self.prompt_fq_pos[-1].weight[None, :, None, :].repeat(T, 1, L * B, 1).flatten(0, 1)  # TfQ, LB, C

                # QxNxC
                #query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C

                output_p = self.prompt_feat[-1].weight.unsqueeze(1).repeat(1, L * B, 1)

                #output = self.query_feat.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C

                decoder_outputs = []
                for i in range(self.num_layers):
                    # attention: cross-attention first

                    if self.prompt_deep:  # True
                        prompt_embed = self.prompt_embed[-1][i].weight.unsqueeze(1).repeat(1, L * B, 1)
                    else:
                        prompt_embed = self.prompt_embed[-1].weight.unsqueeze(1).repeat(1, L * B, 1)


                    output_p = self.transformer_cross_attention_layers[i](
                        output_p, src,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=dec_pos, query_pos=prompt_embed
                    )

                    output_p = self.transformer_self_attention_layers[i](
                        output_p, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=prompt_embed
                    )

                    # FFN
                    output_p = self.transformer_ffn_layers[i](
                        output_p
                    )

                    if (self.training and self.aux_loss) or (i == self.num_layers - 1):
                        dec_out = self.decoder_norm(output_p)  # cQ, LB, C
                        dec_out = dec_out.transpose(0, 1)  # LB, cQ, C
                        decoder_outputs.append(dec_out.view(L, B, self.num_prompts, C))   #  #self.num_queries#######

                decoder_outputs = torch.stack(decoder_outputs, dim=0)  # D, L, B, cQ, C

                pred_cls = self.class_embed(decoder_outputs)
                pred_mask_embed = self.prompt_mask_embed[-1](decoder_outputs) ################
                if self.use_sim and self.sim_use_clip:
                    pred_cq_embed = self.sim_embed_clip(decoder_outputs)
                else:
                    pred_cq_embed = [None] * self.num_layers

                out = {
                    'pred_logits': pred_cls[-1],
                    'pred_mask_embed': pred_mask_embed[-1],
                    'pred_fq_embed': pred_fq_embed,
                    'pred_cq_embed': pred_cq_embed[-1],
                    'aux_outputs': self._set_aux_loss(
                        pred_cls, pred_mask_embed, pred_cq_embed, pred_fq_embed
                    )
                }
                return out


            else:
                #return self.forward_base_train(x, src, pos, size_list, mask_features)

                if not self.Mamba:

                    if self.window_size > 0:
                        pad = int(ceil(T / self.window_size)) * self.window_size - T
                        _T = pad + T
                        frame_query = F.pad(frame_query, (0, 0, 0, 0, 0, 0, 0, pad))  # _T, fQ, LB, C
                        enc_mask = frame_query.new_ones(L * B, _T).bool()  # LB, _T
                        enc_mask[:, :T] = False
                    else:
                        enc_mask = None

                    frame_query = self.encode_frame_query(frame_query, enc_mask)  ########################
                    frame_query = frame_query[:T].flatten(0, 1)  # TfQ, LB, C

                else:
                    enc_mask = None
                    frame_query = self.encode_frame_query_mamba(frame_query, enc_mask) #######################
                    frame_query = frame_query.flatten(0, 1)  # TfQ, LB, C  [:T]



                if self.use_sim:
                    pred_fq_embed = self.sim_embed_frame(frame_query)  # TfQ, LB, C
                    pred_fq_embed = pred_fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
                else:
                    pred_fq_embed = None

                src = self.src_embed(frame_query)  # TfQ, LB, C
                dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L * B, 1).flatten(0, 1)  # TfQ, LB, C

                # QxNxC
                query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C
                output = self.query_feat.weight.unsqueeze(1).repeat(1, L * B, 1)  # cQ, LB, C

                decoder_outputs = []
                for i in range(self.num_layers):
                    # attention: cross-attention first
                    output = self.transformer_cross_attention_layers[i](
                        output, src,
                        memory_mask=None,
                        memory_key_padding_mask=None,
                        pos=dec_pos, query_pos=query_embed
                    )

                    output = self.transformer_self_attention_layers[i](
                        output, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed
                    )

                    # FFN
                    output = self.transformer_ffn_layers[i](
                        output
                    )

                    if (self.training and self.aux_loss) or (i == self.num_layers - 1):
                        dec_out = self.decoder_norm(output)  # cQ, LB, C
                        dec_out = dec_out.transpose(0, 1)  # LB, cQ, C
                        decoder_outputs.append(dec_out.view(L, B, self.num_queries, C))

                decoder_outputs = torch.stack(decoder_outputs, dim=0)  # D, L, B, cQ, C

                pred_cls = self.class_embed(decoder_outputs)
                pred_mask_embed = self.mask_embed(decoder_outputs)
                if self.use_sim and self.sim_use_clip:
                    pred_cq_embed = self.sim_embed_clip(decoder_outputs)
                else:
                    pred_cq_embed = [None] * self.num_layers

                out = {
                    'pred_logits': pred_cls[-1],
                    'pred_mask_embed': pred_mask_embed[-1],
                    'pred_fq_embed': pred_fq_embed,
                    'pred_cq_embed': pred_cq_embed[-1],
                    'aux_outputs': self._set_aux_loss(
                        pred_cls, pred_mask_embed, pred_cq_embed, pred_fq_embed
                    )
                }
                return out




    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_cls, outputs_mask_embed, outputs_cq_embed, outputs_fq_embed
    ):
        return [{"pred_logits": a, "pred_mask_embed": b, "pred_cq_embed": c, "pred_fq_embed": outputs_fq_embed}
                for a, b, c in zip(outputs_cls[:-1], outputs_mask_embed[:-1], outputs_cq_embed[:-1])]

    def encode_frame_query_mamba(self, frame_query, attn_mask):
        """
        input shape (frame_query)   : T, fQ, LB, C
        output shape (frame_query)  : T, fQ, LB, C
        """
        #return_shape = frame_query.shape  # T, fQ, LB, C
        #frame_query = frame_query.flatten(0, 1)  # TfQ, LB, C

        for i in range(self.enc_layers):
            frame_query = self.enc_mamba[i](frame_query)

        #frame_query = frame_query.view(return_shape)
        return frame_query

    def encode_frame_query(self, frame_query, attn_mask):
        """
        input shape (frame_query)   : T, fQ, LB, C
        output shape (frame_query)  : T, fQ, LB, C
        """

        # Not using window-based attention if self.window_size == 0.
        if self.window_size == 0:
            return_shape = frame_query.shape        # T, fQ, LB, C
            frame_query = frame_query.flatten(0, 1) # TfQ, LB, C

            for i in range(self.enc_layers):
                frame_query = self.enc_self_attn[i](frame_query)
                frame_query = self.enc_ffn[i](frame_query)

            frame_query = frame_query.view(return_shape)
            return frame_query
        # Using window-based attention if self.window_size > 0.
        else:
            T, fQ, LB, C = frame_query.shape
            W = self.window_size
            Nw = T // W
            half_W = int(ceil(W / 2))

            window_mask = attn_mask.view(LB*Nw, W)[..., None].repeat(1, 1, fQ).flatten(1)

            _attn_mask  = torch.roll(attn_mask, half_W, 1)
            _attn_mask  = _attn_mask.view(LB, Nw, W)[..., None].repeat(1, 1, 1, W)    # LB, Nw, W, W
            _attn_mask[:,  0] = _attn_mask[:,  0] | _attn_mask[:,  0].transpose(-2, -1)
            _attn_mask[:, -1] = _attn_mask[:, -1] | _attn_mask[:, -1].transpose(-2, -1)
            _attn_mask[:, 0, :half_W, half_W:] = True
            _attn_mask[:, 0, half_W:, :half_W] = True
            _attn_mask  = _attn_mask.view(LB*Nw, 1, W, 1, W, 1).repeat(1, self.num_heads, 1, fQ, 1, fQ).view(LB*Nw*self.num_heads, W*fQ, W*fQ)
            shift_window_mask = _attn_mask.float() * -1000

            for layer_idx in range(self.enc_layers):
                if self.training or layer_idx % 2 == 0:
                    frame_query = self._window_attn(frame_query, window_mask, layer_idx)
                else:
                    frame_query = self._shift_window_attn(frame_query, shift_window_mask, layer_idx)
            return frame_query

    def _window_attn(self, frame_query, attn_mask, layer_idx):
        T, fQ, LB, C = frame_query.shape
        # LBN, WTfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W

        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_key_padding_mask=attn_mask)
        frame_query = self.enc_ffn[layer_idx](frame_query)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        return frame_query

    def _shift_window_attn(self, frame_query, attn_mask, layer_idx):
        T, fQ, LB, C = frame_query.shape
        # LBNH, WfQ, WfQ = attn_mask.shape

        W = self.window_size
        Nw = T // W
        half_W = int(ceil(W / 2))

        frame_query = torch.roll(frame_query, half_W, 0)
        frame_query = frame_query.view(Nw, W, fQ, LB, C)
        frame_query = frame_query.permute(1,2,3,0,4).reshape(W*fQ, LB*Nw, C)

        frame_query = self.enc_self_attn[layer_idx](frame_query, tgt_mask=attn_mask)
        frame_query = self.enc_ffn[layer_idx](frame_query)
        frame_query = frame_query.reshape(W, fQ, LB, Nw, C).permute(3,0,1,2,4).reshape(T, fQ, LB, C)

        frame_query = torch.roll(frame_query, -half_W, 0)

        return frame_query
