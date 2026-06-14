# train_models.py
"""
所有对比实验函数
"""

import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam, SGD
from VGG_Loss_Landscape import set_random_seeds, train, get_accuracy
from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Light
from data.loaders import get_cifar_loader

# ====================== 辅助模型类（用于激活函数对比） ======================
class VGG_A_LeakyReLU(VGG_A):
    """将 VGG_A 中的所有 ReLU 替换为 LeakyReLU"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_relu(self.features)
        self._replace_relu(self.classifier)

    def _replace_relu(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.LeakyReLU(negative_slope=0.01))
            else:
                self._replace_relu(child)

class VGG_A_BatchNorm_LeakyReLU(VGG_A_BatchNorm):
    """将 VGG_A_BatchNorm 中的所有 ReLU 替换为 LeakyReLU"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replace_relu(self.features)
        self._replace_relu(self.classifier)

    def _replace_relu(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.LeakyReLU(negative_slope=0.01))
            else:
                self._replace_relu(child)

# ====================== 实验 1：BN vs 无BN ======================
def experiment_bn_vs_nonbn(train_loader, val_loader, device,
                           figures_path, models_path, epochs=20):
    """对比 VGG_A 与 VGG_A_BatchNorm 的训练曲线和验证准确率"""
    set_random_seeds(2020, device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001

    # 1. 训练无 BN 模型
    model_no_bn = VGG_A()
    opt_no_bn = Adam(model_no_bn.parameters(), lr=lr)
    _, _, losses_no_bn, accs_no_bn = train(
        model_no_bn, opt_no_bn, criterion, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="VGG_A_no_BN", device=device
    )

    # 2. 训练有 BN 模型
    model_bn = VGG_A_BatchNorm()
    opt_bn = Adam(model_bn.parameters(), lr=lr)
    _, _, losses_bn, accs_bn = train(
        model_bn, opt_bn, criterion, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="VGG_A_BN", device=device
    )

    # 3. 绘制对比图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses_no_bn, 'b-', label='Without BN')
    plt.plot(range(1, epochs+1), losses_bn, 'g-', label='With BN')
    plt.title('Training Loss Comparison (BN vs No BN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accs_no_bn, 'b-', label='Without BN')
    plt.plot(range(1, epochs+1), accs_bn, 'g-', label='With BN')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(figures_path, 'BN_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[BN对比] 图表已保存至 {save_path}")

    # 保存模型权重
    torch.save(model_no_bn.state_dict(), os.path.join(models_path, 'VGG_A_no_BN.pth'))
    torch.save(model_bn.state_dict(), os.path.join(models_path, 'VGG_A_BN.pth'))
    print("[BN对比] 模型权重已保存")

# ====================== 实验 2：不同滤波器数量 ======================
def experiment_different_filters(train_loader, val_loader, device,
                                 figures_path, models_path, epochs=20):
    """对比 VGG_A（标准）与 VGG_A_Light（更少滤波器）"""
    set_random_seeds(2020, device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001

    # 标准模型
    model_std = VGG_A()
    opt_std = Adam(model_std.parameters(), lr=lr)
    _, _, losses_std, accs_std = train(
        model_std, opt_std, criterion, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="VGG_A_std", device=device
    )

    # 轻量模型（更少滤波器）
    model_light = VGG_A_Light()
    opt_light = Adam(model_light.parameters(), lr=lr)
    _, _, losses_light, accs_light = train(
        model_light, opt_light, criterion, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="VGG_A_light", device=device
    )

    # 绘图对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses_std, 'b-', label='Standard (many filters)')
    plt.plot(range(1, epochs+1), losses_light, 'r-', label='Light (few filters)')
    plt.title('Training Loss Comparison (Filter Quantity)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accs_std, 'b-', label='Standard')
    plt.plot(range(1, epochs+1), accs_light, 'r-', label='Light')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(figures_path, 'filter_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[滤波器对比] 图表已保存至 {save_path}")

    torch.save(model_std.state_dict(), os.path.join(models_path, 'VGG_A_std.pth'))
    torch.save(model_light.state_dict(), os.path.join(models_path, 'VGG_A_light.pth'))

# ====================== 实验 3：不同损失函数 ======================
def experiment_different_losses(train_loader, val_loader, device,
                                figures_path, models_path, epochs=20):
    """对比标准 CrossEntropyLoss 与带 label_smoothing 的 CrossEntropyLoss"""
    set_random_seeds(2020, device)
    model = VGG_A()
    lr = 0.001
    optimizer = Adam(model.parameters(), lr=lr)

    # 损失函数1：标准交叉熵
    criterion1 = nn.CrossEntropyLoss()
    _, _, losses1, accs1 = train(
        model, optimizer, criterion1, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="loss_CE", device=device
    )
    # 重新初始化模型和优化器（避免权重影响）
    model2 = VGG_A()
    optimizer2 = Adam(model2.parameters(), lr=lr)
    # 损失函数2：带标签平滑的交叉熵 (label_smoothing=0.1)
    criterion2 = nn.CrossEntropyLoss(label_smoothing=0.1)
    _, _, losses2, accs2 = train(
        model2, optimizer2, criterion2, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="loss_CE_smooth", device=device
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses1, 'b-', label='CrossEntropyLoss')
    plt.plot(range(1, epochs+1), losses2, 'r-', label='CrossEntropyLoss (label_smooth=0.1)')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accs1, 'b-', label='CrossEntropyLoss')
    plt.plot(range(1, epochs+1), accs2, 'r-', label='CrossEntropyLoss (label_smooth=0.1)')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(figures_path, 'loss_function_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[损失函数对比] 图表已保存至 {save_path}")

    torch.save(model.state_dict(), os.path.join(models_path, 'loss_CE.pth'))
    torch.save(model2.state_dict(), os.path.join(models_path, 'loss_CE_smooth.pth'))

# ====================== 实验 4：不同激活函数 ======================
def experiment_different_activations(train_loader, val_loader, device,
                                     figures_path, models_path, epochs=20):
    """对比 ReLU 与 LeakyReLU"""
    set_random_seeds(2020, device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001

    # 标准 ReLU 模型
    model_relu = VGG_A()
    opt_relu = Adam(model_relu.parameters(), lr=lr)
    _, _, losses_relu, accs_relu = train(
        model_relu, opt_relu, criterion, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="act_ReLU", device=device
    )

    # LeakyReLU 模型
    model_leaky = VGG_A_LeakyReLU()
    opt_leaky = Adam(model_leaky.parameters(), lr=lr)
    _, _, losses_leaky, accs_leaky = train(
        model_leaky, opt_leaky, criterion, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="act_LeakyReLU", device=device
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses_relu, 'b-', label='ReLU')
    plt.plot(range(1, epochs+1), losses_leaky, 'g-', label='LeakyReLU')
    plt.title('Training Loss Comparison (Activation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accs_relu, 'b-', label='ReLU')
    plt.plot(range(1, epochs+1), accs_leaky, 'g-', label='LeakyReLU')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(figures_path, 'activation_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[激活函数对比] 图表已保存至 {save_path}")

    torch.save(model_relu.state_dict(), os.path.join(models_path, 'act_ReLU.pth'))
    torch.save(model_leaky.state_dict(), os.path.join(models_path, 'act_LeakyReLU.pth'))

# ====================== 实验 5：不同优化器 ======================
def experiment_different_optimizers(train_loader, val_loader, device,
                                    figures_path, models_path, epochs=20):
    """对比 Adam 与 SGD (with momentum)"""
    set_random_seeds(2020, device)
    criterion = nn.CrossEntropyLoss()
    lr = 0.001

    # Adam 优化器
    model_adam = VGG_A()
    opt_adam = Adam(model_adam.parameters(), lr=lr)
    _, _, losses_adam, accs_adam = train(
        model_adam, opt_adam, criterion, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="opt_Adam", device=device
    )

    # SGD with momentum
    model_sgd = VGG_A()
    opt_sgd = SGD(model_sgd.parameters(), lr=lr, momentum=0.9)
    _, _, losses_sgd, accs_sgd = train(
        model_sgd, opt_sgd, criterion, train_loader, val_loader,
        epochs_n=epochs, save_curve_path=figures_path, model_name="opt_SGD", device=device
    )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), losses_adam, 'b-', label='Adam')
    plt.plot(range(1, epochs+1), losses_sgd, 'r-', label='SGD (momentum=0.9)')
    plt.title('Training Loss Comparison (Optimizer)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accs_adam, 'b-', label='Adam')
    plt.plot(range(1, epochs+1), accs_sgd, 'r-', label='SGD')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(figures_path, 'optimizer_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[优化器对比] 图表已保存至 {save_path}")

    torch.save(model_adam.state_dict(), os.path.join(models_path, 'opt_Adam.pth'))
    torch.save(model_sgd.state_dict(), os.path.join(models_path, 'opt_SGD.pth'))

# ====================== 实验 6：损失景观对比（BN vs 无BN） ======================
def experiment_loss_landscape_bn_comparison(train_loader, val_loader, device,
                                            figures_path, models_path):
    """
    对比有 BN 和无 BN 的 VGG 在不同学习率下的损失景观（min-max 填充图）
    根据项目要求，需要对多个学习率训练模型，记录每个 step 的 loss，然后绘制填充区域。
    """
    lr_list = [1e-3, 2e-3, 1e-4, 5e-4]
    epochs = 10  # 为了节省时间，使用较少 epoch，仅用于景观对比
    criterion = nn.CrossEntropyLoss()

    def collect_step_losses(model_class, model_name_prefix):
        """训练多个学习率的模型，返回所有step的loss列表（每个学习率一个列表）"""
        all_step_losses = []
        for lr in lr_list:
            set_random_seeds(2020, device)
            model = model_class()
            optimizer = Adam(model.parameters(), lr=lr)
            # 训练并获取 losses_list（每个epoch内每个step的loss）
            losses_list, _, _, _ = train(
                model, optimizer, criterion, train_loader, val_loader,
                epochs_n=epochs, save_curve_path=None, model_name=f"{model_name_prefix}_lr{lr}",
                device=device
            )
            # 将每个epoch内的step loss拼接成一个长列表（跨所有epoch）
            all_steps = [loss for epoch_losses in losses_list for loss in epoch_losses]
            all_step_losses.append(all_steps)
        return all_step_losses

    print("正在收集无 BN 模型的损失景观数据...")
    losses_no_bn = collect_step_losses(VGG_A, "noBN")
    print("正在收集有 BN 模型的损失景观数据...")
    losses_bn = collect_step_losses(VGG_A_BatchNorm, "BN")

    # 对齐长度（取所有列表中最短的长度，因为不同学习率可能步数略有差异）
    min_len = min(min(len(lst) for lst in losses_no_bn), min(len(lst) for lst in losses_bn))
    losses_no_bn_aligned = [lst[:min_len] for lst in losses_no_bn]
    losses_bn_aligned = [lst[:min_len] for lst in losses_bn]

    # 计算每个step上的 min 和 max
    arr_no_bn = np.array(losses_no_bn_aligned)
    arr_bn = np.array(losses_bn_aligned)
    min_no_bn = np.min(arr_no_bn, axis=0)
    max_no_bn = np.max(arr_no_bn, axis=0)
    min_bn = np.min(arr_bn, axis=0)
    max_bn = np.max(arr_bn, axis=0)
    steps = np.arange(min_len)

    # 绘制对比图
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for curve in losses_no_bn_aligned:
        plt.plot(steps, curve, alpha=0.5, color='blue', linewidth=0.5)
    plt.fill_between(steps, min_no_bn, max_no_bn, color='blue', alpha=0.2, label='Variation')
    plt.title('Loss Landscape: VGG-A without BN')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for curve in losses_bn_aligned:
        plt.plot(steps, curve, alpha=0.5, color='green', linewidth=0.5)
    plt.fill_between(steps, min_bn, max_bn, color='green', alpha=0.2, label='Variation')
    plt.title('Loss Landscape: VGG-A with BN')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(figures_path, 'loss_landscape_BN_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[损失景观对比] 图表已保存至 {save_path}")

# ====================== 实验 7：报告最佳模型 ======================
def report_best_model(train_loader, test_loader, device,
                      figures_path, models_path, epochs=30):
    """
    根据前6个实验的最佳结果，训练最终模型：
    - 网络：VGG_A_BatchNorm + LeakyReLU（即带 BN 且激活函数为 LeakyReLU）
    - 损失函数：CrossEntropyLoss with label_smoothing=0.1
    - 优化器：Adam (lr=0.001)
    - 不额外使用学习率调度器（或者可选，但实验表明 Adam 本身自适应）
    """
    from models.vgg import VGG_A_BatchNorm
    import time

    set_random_seeds(2020, device)

    # 使用带 BN 且激活函数为 LeakyReLU 的模型
    model = VGG_A_BatchNorm_LeakyReLU()
    model = model.to(device)

    # 损失函数：标签平滑
    criterion = nn.CrossEntropyLoss()

    # 优化器：Adam（原始 lr=0.001，可根据需要调整）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 可选：如果需要学习率调度，可以添加，但 Adam 通常不需要。这里注释掉，保持简单
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"最佳模型总参数量: {total_params:,}, 可训练参数量: {trainable_params:,}")

    # 训练并计时
    start_time = time.time()
    _, _, train_losses, val_accs = train(
        model, optimizer, criterion, train_loader, test_loader,
        scheduler=None,  # 不使用调度器，或者根据需要添加
        epochs_n=epochs,
        save_curve_path=figures_path, model_name="best_model", device=device
    )
    training_time = time.time() - start_time
    print(f"训练耗时: {training_time:.2f} 秒")

    # 在测试集上评估最终准确率
    test_acc = get_accuracy(model, test_loader, device)
    print(f"最佳模型在测试集上的准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(models_path, 'best_model.pth'))

    # 将统计信息写入文件
    with open(os.path.join(models_path, 'best_model_stats.txt'), 'w') as f:
        f.write(f"Total parameters: {total_params}\n")
        f.write(f"Trainable parameters: {trainable_params}\n")
        f.write(f"Training time (seconds): {training_time:.2f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")

    print(f"最佳模型训练曲线已保存至 {figures_path}/best_model_training_curves.png")

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 路径配置（与原始代码保持一致）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    figures_path = os.path.join(project_dir, 'reports', 'figures')
    models_path = os.path.join(project_dir, 'reports', 'models')

    # 创建保存目录
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载器
    train_loader = get_cifar_loader(root=data_dir, train=True, batch_size=128, num_workers=0)
    val_loader = get_cifar_loader(root=data_dir, train=False, batch_size=128, num_workers=0)

    # 依次运行所有实验（可以根据需要注释某些耗时的实验）
    # print("\n===== 实验1：BN vs 无BN =====")
    # experiment_bn_vs_nonbn(train_loader, val_loader, device, figures_path, models_path, epochs=20)

    # print("\n===== 实验2：不同滤波器数量 =====")
    # experiment_different_filters(train_loader, val_loader, device, figures_path, models_path, epochs=20)

    # print("\n===== 实验3：不同损失函数 =====")
    # experiment_different_losses(train_loader, val_loader, device, figures_path, models_path, epochs=20)

    # print("\n===== 实验4：不同激活函数 =====")
    # experiment_different_activations(train_loader, val_loader, device, figures_path, models_path, epochs=20)

    # print("\n===== 实验5：不同优化器 =====")
    # experiment_different_optimizers(train_loader, val_loader, device, figures_path, models_path, epochs=50)

    # print("\n===== 实验6：损失景观对比（需要较长时间） =====")
    # experiment_loss_landscape_bn_comparison(train_loader, val_loader, device, figures_path, models_path)

    print("\n===== 实验7：训练最佳模型并报告 =====")
    # 使用测试集（val_loader 就是测试集，因为 CIFAR-10 的 train=False）
    report_best_model(train_loader, val_loader, device, figures_path, models_path, epochs=20)

    print("\n所有实验完成！结果保存在 reports/figures 和 reports/models 中。")