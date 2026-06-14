import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader





# This function is used to calculate the accuracy of model classification
def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None, save_curve_path=None, model_name='model', device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    epoch_train_losses = []
    epoch_val_accs = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve
        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()
            ## --------------------


            loss.backward()
            if model.classifier[4].weight.grad is not None:
                grad.append(model.classifier[4].weight.grad.norm().item())
            optimizer.step()

        avg_loss = learning_curve[epoch] / batches_n
        epoch_train_losses.append(avg_loss)
        losses_list.append(loss_list)
        grads.append(grad)

        # display.clear_output(wait=True)
        # f, axes = plt.subplots(1, 2, figsize=(15, 3))

        # learning_curve[epoch] /= batches_n
        # axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_x, val_y = val_data
                val_x, val_y = val_x.to(device), val_y.to(device)
                outputs = model(val_x)
                _, predicted = torch.max(outputs.data, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()
        
        val_accuracy = correct / total
        val_accuracy_curve[epoch] = val_accuracy
        epoch_val_accs.append(val_accuracy)
        
        # axes[1].plot(val_accuracy_curve)
        # axes[1].set_title('Validation Accuracy')
        # plt.savefig(os.path.join(figures_path, f'epoch_{epoch}_curve.png'))
        # plt.close(f)
        ## --------------------
    if save_curve_path is not None:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs_n+1), epoch_train_losses, 'b-', linewidth=2)
        plt.title(f'{model_name} - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs_n+1), epoch_val_accs, 'r-', linewidth=2)
        plt.title(f'{model_name} - Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_curve_path, f'{model_name}_training_curves.png'), dpi=150)
        plt.close()
    
    return losses_list, grads, epoch_train_losses, epoch_val_accs    

def plot_loss_landscape(all_losses, min_curve, max_curve, save_path=None):
    """
    all_losses: list of lists, 每个元素是一个模型（或学习率）的 step-wise loss 列表
    min_curve: 每个 step 上的最小损失 (list)
    max_curve: 每个 step 上的最大损失 (list)
    save_path: 保存图片的路径，若为 None 则显示（但 Agg 后端无法显示，建议总是保存）
    """
    plt.figure(figsize=(10, 6))
    steps = np.arange(len(all_losses[0]))
    for curve in all_losses:
        plt.plot(steps, curve, alpha=0.7, linewidth=0.5)
    plt.fill_between(steps, min_curve, max_curve, color='gray', alpha=0.2, label='Variation')
    plt.title('Loss Landscape')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
