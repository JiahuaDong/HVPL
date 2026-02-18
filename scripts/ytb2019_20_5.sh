#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/.."

export DETECTRON2_DATASETS=/data2/HVPL-main/datasets
export CUDA_VISIBLE_DEVICES=1,2

NGPUS=2
CFG_FILE="configs/youtubevis_2019/vita_R50_bs8.yaml"
OUTPUT_BASE="results/Video_yvis2019_IS"
EXP_NAME="Video_Instance_20_5_mamba"

STEP_ARGS="CONT.BASE_CLS 20 CONT.INC_CLS 5 CONT.MODE overlap SEED 42"
METH_ARGS="MODEL.MASK_FORMER.TEST.MASK_BG False MODEL.MASK_FORMER.PER_PIXEL False MODEL.MASK_FORMER.FOCAL True"



BASE_QUERIES=100
ITER_BASE=80000
SOFT_MASK=False
SOFT_CLS=False
NUM_PROMPTS=0
DEEP_CLS=True

WEIGHT_ARGS="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES} \
             CONT.NUM_PROMPTS ${NUM_PROMPTS} \
             MODEL.MASK_FORMER.SOFTMASK ${SOFT_MASK} \
             CONT.SOFTCLS ${SOFT_CLS} \
             CONT.DEEP_CLS ${DEEP_CLS}"

COMM_ARGS="OUTPUT_DIR ${OUTPUT_BASE} ${METH_ARGS} ${STEP_ARGS} ${WEIGHT_ARGS}"

INC_ARGS_0="CONT.TASK 0 \
            TEST.EVAL_PERIOD 5000 \
            SOLVER.CHECKPOINT_PERIOD 5000 \
            CONT.WEIGHTS vita_r50_coco.pth \
            SOLVER.STEPS (55000,) \
            SOLVER.MAX_ITER ${ITER_BASE}"

#train the base model

python train_net_hvpl.py --num-gpus ${NGPUS} \
    --dist-url tcp://127.0.0.1:50164 \
    --config-file ${CFG_FILE} \
    ${COMM_ARGS} ${INC_ARGS_0} \
    NAME ${EXP_NAME}


OGC=True
THRESHOLD=0.7
ITER_INC=30000
BASE_QUERIES_INC=100
NUM_PROMPTS_INC=10

deltas=[0.4,0.5,0.5,0.5,0.5,0.5]

VPT_ARGS="CONT.BACKBONE_FREEZE True \
          CONT.TRANS_DECODER_FREEZE True \
          CONT.PIXEL_DECODER_FREEZE True \
          CONT.CLS_HEAD_FREEZE True \
          CONT.MASK_HEAD_FREEZE True \
          CONT.QUERY_EMBED_FREEZE True \
          CONT.PROMPT_DEEP True \
          CONT.PROMPT_MASK_MLP True \
          CONT.PROMPT_NO_OBJ_MLP False \
          CONT.DEEP_CLS True \
          CONT.LOGIT_MANI_DELTAS ${deltas}"

WEIGHT_ARGS_INC="MODEL.MASK_FORMER.NUM_OBJECT_QUERIES ${BASE_QUERIES_INC} \
                 MODEL.VITA.NUM_OBJECT_QUERIES ${BASE_QUERIES_INC} \
                 MODEL.MASK_FORMER.SOFTMASK ${SOFT_MASK} \
                 CONT.SOFTCLS ${SOFT_CLS} \
                 CONT.NUM_PROMPTS ${NUM_PROMPTS_INC}"

COMM_ARGS_INC="OUTPUT_DIR ${OUTPUT_BASE} ${METH_ARGS} ${STEP_ARGS} ${WEIGHT_ARGS_INC}"

#train first incremental step
PRETRAINED_PATH="results/Video_yvis2019_IS/coco2ytvis2019_train_20-5-ov/Video_Instance_20_5_mamba/step0/model_final.pth"
python train_net_hvpl.py --num-gpus ${NGPUS} \
    --config-file ${CFG_FILE} \
    ${COMM_ARGS_INC} ${VPT_ARGS} \
    CONT.TASK 1 \
    SOLVER.MAX_ITER ${ITER_INC} \
    CONT.OGC ${OGC} \
    CONT.THRESHOLD ${THRESHOLD} \
    CONT.WEIGHTS ${PRETRAINED_PATH} \
    NAME ${EXP_NAME}
#train the rest incremental steps
for t in 2 3 4; do
    echo ">>> Starting Task ${t}"
    python train_net_hvpl.py --num-gpus ${NGPUS} \
        --config-file ${CFG_FILE} \
        ${COMM_ARGS_INC} ${VPT_ARGS} \
        CONT.TASK ${t} \
        SOLVER.MAX_ITER ${ITER_INC} \
        CONT.OGC ${OGC} \
        CONT.THRESHOLD ${THRESHOLD} \
        NAME ${EXP_NAME}
done