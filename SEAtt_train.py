import torch
import ultralytics
import torch.optim as optim
from torch_lr_finder import LRFinder

from ultralytics.data.split import autosplit


def configure_optimizer(model, lr=1e-4, lr_multiplier=10, weight_decay=0.05):
    """配置分层学习率优化器"""
    pretrained_params = []  # 骨干网络（预训练部分）
    attention_params = []  # SEAttention 模块
    head_params = []  # 检测头（head）

    for name, param in model.named_parameters():
        if "SEAttention" in name:  # 注意力模块
            attention_params.append(param)
        elif "backbone" in name:  # 骨干网络
            pretrained_params.append(param)
        else:  # 检测头
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": pretrained_params, "lr": lr},
            {"params": attention_params, "lr": lr * (lr_multiplier ** 0.5)},
            {"params": head_params, "lr": lr * lr_multiplier},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


def find_optimal_lr(model, train_loader, loss_fn, lr_multiplier=10):
    """学习率寻优（LR Finder）"""
    optimizer = configure_optimizer(model, lr=1e-7, lr_multiplier=lr_multiplier)
    lr_finder = LRFinder(model, optimizer, loss_fn, device="cuda" if torch.cuda.is_available() else "cpu")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")
    _, suggested_lr = lr_finder.plot()
    lr_finder.reset()
    return suggested_lr

class SchedulerCallback:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_train_epoch_end(self, trainer):
        self.scheduler.step()
def train_with_lr_finder(model, data_yaml, config_path, pretrained_weights, epochs=100):
    """分阶段训练 + 学习率寻优（改进版）"""
    # 阶段1：冻结骨干，固定学习率
    print("🚀 阶段1：冻结骨干，训练注意力+检测头")
    for name, param in model.model.named_parameters():
        if "backbone" in name and "SEAttention" not in name:
            param.requires_grad = False

    model.train(
        cfg=config_path,
        data=data_yaml,
        epochs=5,
        lr0=1e-3,  # 显式设置学习率
        name="SEAtt_train/phase1"
    )

    # 阶段2：解冻骨干，学习率寻优
    print("🔍 阶段2：解冻骨干，学习率寻优")
    for param in model.model.parameters():
        param.requires_grad = True

    train_loader = model.get_dataloader(data_yaml, batch_size=model.args.batch)
    loss_fn = model.criterion

    # 学习率寻优（更安全的范围）
    suggested_lr = find_optimal_lr(model.model, train_loader, loss_fn)
    print(f"✅ 建议学习率：{suggested_lr:.2e}")

    # 配置优化器和调度器
    optimizer = configure_optimizer(
        model.model,
        lr=suggested_lr,
        lr_multiplier=10,
        weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=7,
        T_mult=2,
        eta_min=max(1e-6, suggested_lr/10)
    )

    model.add_callback('on_train_epoch_end', lambda trainer: scheduler.step())

    # 阶段2训练
    print("🚀 阶段2：使用优化后的学习率训练")
    model.train(
        cfg=config_path,
        data=data_yaml,
        epochs=epochs,
        optimizer=optimizer,  # 禁用YOLO默认优化器
        name="SEAtt_train/phase2"
    )


if __name__ == '__main__':
    autosplit('data/datasets/waste_classification/images', weights=(0.99, 0.01, 0),
              annotated_only=True)

    # 配置参数
    config_path = 'cfg/train.yaml'  # 训练配置文件路径
    data_yaml = 'data/datasets/waste_classification/waste_classification.yaml'  # 数据配置文件路径
    pretrained_weights = 'weights/yolov8n_pretrained.pt'  # 预训练权重路径

    # 加载模型
    model = ultralytics.YOLO("cfg/models/v8/SEAtt_yolov8.yaml").load(pretrained_weights)

    # 开始训练
    train_with_lr_finder(
        model=model,
        data_yaml=data_yaml,
        config_path=config_path,
        pretrained_weights=pretrained_weights,
        epochs=100
    )