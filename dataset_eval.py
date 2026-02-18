from ytvoseval_continuous import YTVOSeval
import numpy as np
import pylab
from hvpl.data.ytvos_continuous_per_task import YTVOS_incremental_per_task
import os
from tqdm import tqdm


class DummyCfg:
    def __init__(self):
        self.CONT = self.CONT_OBJ()

    class CONT_OBJ:
        def __init__(self):
            self.TASK = 0
            self.BASE_CLS = 0
            self.INC_CLS = 0


forgetting_type = 'class'
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
num_classes = 40
base_classes = 30
inc_classes = 10
annType = 'segm'
annFile = '/data1/phm/HVPL-main/datasets/split_dataset/YouTube-2021/test_split.json'

for method in ['ours']:
    resFile_dir = f'/data1/phm/HVPL-main/ytb2021_30_10_result/'

    num_tasks = (num_classes - base_classes) // inc_classes + 1

    if forgetting_type == 'task':
        metrix = np.zeros((12, num_tasks, num_tasks))
        max_metric = np.zeros((12, num_tasks))
        count_task = np.zeros((12, num_tasks))
    else:
        metrix = np.zeros((12, num_classes, num_tasks))
        max_metric = np.zeros((12, num_classes))
        count_task = np.zeros((12, num_classes))

    task_pbar = tqdm(range(num_tasks), desc=f"Overall Progress ({method})")

    for task_id in task_pbar:
        classes = base_classes + task_id * inc_classes
        resFile = os.path.join(resFile_dir, f'results_{task_id}.json')

        task_pbar.set_description(f"Task {task_id} [{classes} Cls]")

        if forgetting_type == 'task':
            count_task[:, task_id] = num_tasks - task_id - 1
            sub_loop = tqdm(range(task_id + 1), desc="  Evaluating Tasks", leave=False)
            for id in sub_loop:
                tmp_cfg = DummyCfg()
                if id == 0:
                    tmp_cfg.CONT.TASK = 0
                    tmp_cfg.CONT.BASE_CLS = base_classes
                else:
                    tmp_cfg.CONT.TASK = id
                    tmp_cfg.CONT.BASE_CLS = base_classes
                    tmp_cfg.CONT.INC_CLS = inc_classes

                visGt = YTVOS_incremental_per_task(annFile, cfg=tmp_cfg)
                visDt = visGt.loadRes(resFile, None)
                vidIds = sorted(visGt.vidid_filter)

                visEval = YTVOSeval(visGt, visDt, annType)
                visEval.params.vidIds = vidIds
                visEval.evaluate()
                visEval.accumulate()
                visEval.summarize()

                for idx in range(len(visEval.stats)):
                    metrix[idx, id, task_id] = visEval.stats[idx]
                    if id == task_id:
                        max_metric[idx, id] = visEval.stats[idx]
        else:
            cls_pbar = tqdm(range(classes), desc="  Classes", leave=False)
            for cls in cls_pbar:
                if task_id == 0 and cls < base_classes:
                    count_task[:, cls] = num_tasks - task_id - 1
                elif task_id > 0 and cls >= (classes - inc_classes) and cls < classes:
                    count_task[:, cls] = num_tasks - task_id - 1

                cls_pbar.set_postfix(cls=cls)

                tmp_cfg = DummyCfg()
                tmp_cfg.CONT.TASK = 0
                tmp_cfg.CONT.BASE_CLS = cls + 1

                visGt = YTVOS_incremental_per_task(annFile, cfg=tmp_cfg)
                target_cat_id = cls + 1

                visDt = visGt.loadRes(resFile, None)
                vidIds = visGt.getVidIds(catIds=[target_cat_id])

                visEval = YTVOSeval(visGt, visDt, annType)
                visEval.params.vidIds = vidIds
                visEval.params.catIds = [target_cat_id]

                visEval.evaluate()
                visEval.accumulate()
                visEval.summarize()

                for idx in range(len(visEval.stats)):
                    metrix[idx, cls, task_id] = visEval.stats[idx]
                    if task_id == 0 and cls < base_classes:
                        max_metric[idx, cls] = visEval.stats[idx]
                    elif task_id > 0 and cls >= (classes - inc_classes) and cls < classes:
                        max_metric[idx, cls] = visEval.stats[idx]

    file_path_txt = os.path.join(resFile_dir, f'{forgetting_type}_forgetting.csv')
    if os.path.exists(file_path_txt):
        os.remove(file_path_txt)

    for idx in range(metrix.shape[0]):
        if forgetting_type == 'task':
            forgetting_metric = metrix[idx, :, :]
            forgetting = max_metric[idx, :-1] - forgetting_metric[:-1, -1] + 1e-8
            forgetting[forgetting < 0] = 0
            forgetting = forgetting / (max_metric[idx, :-1] + 1e-8)
            forgetting = forgetting / count_task[idx, :-1]
            forgetting = np.sum(forgetting) / (num_tasks - 1)
        else:
            forgetting_metric = metrix[idx, :, :]
            forgetting = max_metric[idx, :-inc_classes] - forgetting_metric[:-inc_classes, -1] + 1e-8
            forgetting[forgetting < 0] = 0
            forgetting = forgetting / (max_metric[idx, :-inc_classes] + 1e-8)
            forgetting = forgetting / count_task[idx, :-inc_classes]
            forgetting = np.sum(forgetting) / (num_classes - inc_classes)

        with open(file_path_txt, 'a') as file:
            file.write(f"Metric {idx}: {forgetting}\n")

