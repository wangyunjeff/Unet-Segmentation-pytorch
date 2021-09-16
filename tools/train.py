import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.model_training import CE_Loss, Dice_loss, LossHistory
from data.build import UnetDataset,unet_dataset_collate
from modeling.unet import Unet
from engine.trainer import do_train


if __name__ == '__main__':
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # -------------------------------#
    #   训练自己的数据集必须要修改的
    #   自己需要的分类个数+1，如2+1
    # -------------------------------#
    num_classes = 3
    # -------------------------------#
    #   所使用的的主干网络：
    #   VGG
    backbone = "VGG"
    # -------------------------------------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    #   预训练权重对于99%的情况都必须要用，不用的话权值太过随机，特征提取效果不明显
    #   网络训练的结果也不会好，数据的预训练权重对不同数据集是通用的，因为特征是通用的
    # ------------------------------------------------------------------------------------#
    model_path = "weights/ep022-loss0.560-val_loss0.573.pth"
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    input_shape = [512, 512]
    # ----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    # ----------------------------------------------------#
    Init_Epoch = 20
    Freeze_Epoch = 20
    Freeze_batch_size = 2
    Freeze_lr = 5e-4
    # ----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    # ----------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 1
    Unfreeze_lr = 5e-5
    # ------------------------------#
    #   数据集路径
    # ------------------------------#
    dataset_path = '../data/datasets'
    # --------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ---------------------------------------------------------------------#
    dice_loss = True
    # --------------------------------------------------------------------------------------------#
    #   主干网络预训练权重的使用，这里的权值部分仅仅代表主干，下方的model_path代表整个模型的权值
    #   如果想从主干开始训练，可以把这里的pretrained=True，下方model_path的部分注释掉
    # --------------------------------------------------------------------------------------------#
    pretrained = False
    # ------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------#
    Freeze_Train = True
    # ------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------#
    num_workers = 0
    model = Unet(num_classes=num_classes, in_channels=3, pretrained=pretrained)
    loss_history = LossHistory("logs/")


    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, dataset_path, False)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, dataset_path, False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)

        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.vgg.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            do_train(model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, num_classes)
            lr_scheduler.step()
    if True:
        print('1')
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = UnetDataset(train_lines, input_shape, num_classes, dataset_path, False)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, dataset_path, False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)

        epoch_step = len(train_lines) // batch_size
        epoch_step_val = len(val_lines) // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        if Freeze_Train:
            for param in model.vgg.parameters():
                param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            do_train(model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, num_classes)
            lr_scheduler.step()