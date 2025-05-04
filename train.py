import torch
import json
import copy
from K_Fold import K_fold
import sys

# input_dir="/kaggle/working/ws_model/"
# k_fold_result_dir="/kaggle/working/ws_model/data/datasets/waste_classification"
# working_dir='/kaggle/working/ws_model/runs/detect'
input_dir = ''
k_fold_result_dir = "data/datasets/waste_classification/"
working_dir = None
# 将ultralytics目录添加到模块搜索路径
sys.path.append(input_dir+'YOLO/ultralytics')

import ultralytics
def on_train_epoch_end(trainer):
    if trainer.epoch%20 == 0:
        copied_model = copy.deepcopy(trainer.model)
        copied_model.requires_grad = False
        copied_model.eval()
        print(f"第 {trainer.epoch} 个周期结束，进行验证...")
        with torch.no_grad():
            # args = dict(model=copied_model, data="./data/datasets/waste_classification/waste_classification.yaml",mode="val",name=trainer.args.name+'/val-epoch'+str(trainer.epoch))
            args = dict(model=copied_model, data="/kaggle/input/waste-classification/Graduation_project/data/datasets/waste_classification/waste_classification.yaml",
                        project='/kaggle/working/runs/detect', mode="val",
                        name=trainer.args.name + '/val-epoch' + str(trainer.epoch))
            validator = ultralytics.models.yolo.detect.DetectionValidator(args=args)
            validator()

        # 不知道有没有什么好办法，我直接用validator进行验证老是会报错return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass      RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # trainer.model.eval()
        # # 在这里添加每 10 个周期结束时要执行的操作
        # print(f"第 {trainer.epoch} 个周期结束，进行验证...")
        # # 手动调用验证操作
        # with torch.no_grad():
        #     args = dict(model=trainer.model, data="./data/datasets/cowboy_outfits/cowboy_outfits.yaml", mode="val")
        #     validator = ultralytics.models.yolo.detect.DetectionValidator(args=args)
        #     validator()
        #
        # trainer.model.train()



if __name__ == '__main__':

    # model = ultralytics.YOLO('/kaggle/input/waste-classification/Graduation_project/weights/yolov8n_pretrained.pt')
    # model.add_callback('on_train_epoch_end', on_train_epoch_end)
    # model.train(cfg='/kaggle/input/waste-classification/Graduation_project/cfg/train.yaml')



    ds_yamls=K_fold(input_dir,k_fold_result_dir)


    # model = ultralytics.YOLO(input_dir+'weights/yolov8n_pretrained.pt')

    model = ultralytics.YOLO("YOLO/ultralytics/cfg/models/v8/SEAtt_yolov8.yaml").load(input_dir+'weights/yolov8n_pretrained.pt')

    for k, dataset_yaml in enumerate(ds_yamls):
        model.train(
            cfg=input_dir+'cfg/train.yaml',
            project=working_dir,
            data=dataset_yaml,
            name=f"fold_{k + 1}"
        )  # include any additional train arguments