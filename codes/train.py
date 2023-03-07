import torch
import torch.optim as optim
from model_unet import SRF_UNet
from dataset import octa500_Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import os
from utils import *
import numpy as np
from torch.utils.data import SubsetRandomSampler
from collections import defaultdict
import pandas as pd
from statistics import mean
from math import *
from torchinfo import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

Epoches = 3 * 10 ** 2 + 1
check_interval = 20
batch_size = 1
opt_step_size = 4
opt_gamma = 0.5
seg_lr = 1e-4

time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
result_dir = "./results/" + time_str
os.mkdir(result_dir)

def threshold(x, val, pieces_area=10):
    x = x.cpu()
    x = torch.where(x > val, torch.ones(x.shape), torch.zeros(x.shape))
    return x.to(device)

thred_val = 0.7
fov = "3M"

octa_dataset = octa500_Dataset(fov=fov)

# 定义十折交叉验证
k_fold = 10
num_samples = len(octa_dataset)
indices = list(range(num_samples))
np.random.shuffle(indices)
split = int(np.floor(num_samples / k_fold))

best_loss = inf
# 训练模型
for fold_i in range(k_fold):
    # 划分训练集和验证集
    val_indices = indices[fold_i * split:(fold_i + 1) * split]
    train_indices = indices[:fold_i * split] + indices[(fold_i + 1) * split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # model = SwinTransformerSys(in_chans=2, num_classes=1, num_classify_classes=octa_dataset.classify_num)
    model = SRF_UNet(img_ch=2)
    model.to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg,lr=1e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt_step_size, gamma=opt_gamma)

    # 定义数据加载器
    train_loader = DataLoader(octa_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(octa_dataset, batch_size=1, sampler=val_sampler)

    # 每个 fold 的子文件夹
    sample_dir = result_dir + "/" + str(fold_i)
    os.mkdir(sample_dir)

    metrics_train_dct, metrics_test_dct = defaultdict(list), defaultdict(list)

    for epoch in tqdm(range(Epoches)):
        loss_total_dct = defaultdict(list)
        loss_dct = {}

        for i, data in enumerate(train_loader, 0):
            img, label_seg, label_cls, _ = data
            img, label_seg = get_argumented_batch_samples(img, label_seg)
            img, label_seg, label_cls = img.to(device), label_seg.to(device), label_cls.to(device)
            # zero the parameter gradients
            # forward + backward + optimize
            optimizer.zero_grad()

            pred_vessel, pred_faz, pred_heatmap, pred_cls = model(img)
            label_vessel, label_faz, label_heatmap = label_seg[:,0].unsqueeze(1), label_seg[:,1].unsqueeze(1), label_seg[:,2].unsqueeze(1)

            loss_vessel = clDiceLoss()(pred_vessel, label_vessel)
            loss_faz = DiceLoss()(pred_faz, label_faz)
            loss_heatmap = torch.nn.MSELoss()(pred_heatmap, label_heatmap)
            loss_cls = torch.tensor([1]).to(device)
            loss_seg = 0.5 * loss_vessel + 0.5 * loss_faz + 0.05 * loss_heatmap
            # loss_total = loss_seg + 0.01 * loss_cls
            loss_seg.backward()
            optimizer.step()

            for k, v in zip(["loss_vessel", "loss_faz", "loss_heatmap", "loss_cls", "loss_seg"],
                            [loss_vessel, loss_faz, loss_heatmap, loss_cls, loss_seg]):
                loss_dct[k] = v.cpu().item()
            # 计算总体损失
            for loss_item in loss_dct.keys():
                loss_total_dct[loss_item].append(loss_dct[loss_item])

        for loss_item in loss_dct.keys():
            metrics_train_dct[loss_item].append("{:.5f}".format(mean(loss_total_dct[loss_item])))

        # 验证集测试:
        if epoch % check_interval == 0:
            metrics_dct = {}
            metrics_total_dct = defaultdict(list)

            os.mkdir("{}/{:0>4}/".format(sample_dir, epoch))

            metrics_test_dct["epoch"].append(epoch)
            metrics_test_dct["optimizer"].append("{:.6f}".format(optimizer.param_groups[0]['lr']))

            scheduler.step()

            for i, data in enumerate(val_loader, 0):
                metrics_dct = {}

                img, label_seg, label_cls, sample_id = data
                img, label_seg, label_cls = img.to(device), label_seg.to(device), label_cls.to(device)
                pred_vessel, pred_faz, pred_heatmap, pred_cls = model(img)
                label_vessel, label_faz, label_heatmap = label_seg[:, 0].unsqueeze(1), label_seg[:, 1].unsqueeze(1), label_seg[:, 2].unsqueeze(1)

                metrics_dct["loss_vessel"] = clDiceLoss()(pred_vessel, label_vessel).cpu().item()
                metrics_dct["loss_faz"] = DiceLoss()(pred_faz, label_faz).cpu().item()
                metrics_dct["loss_heatmap"] = torch.nn.MSELoss()(pred_faz, label_faz).cpu().item()
                # loss_heatmap = torch.nn.MSELoss()(pred_heatmap, label_heatmap)
                metrics_dct["loss_seg"] = 0.5 * metrics_dct["loss_vessel"] \
                                          + 0.5 * metrics_dct["loss_faz"] + 0.05 * metrics_dct["loss_heatmap"]
                # loss_cls = torch.nn.CrossEntropyLoss()(pred_cls, label_cls)
                loss_cls = torch.tensor([1])

                # 计算指标
                pred_seg = [pred_vessel, pred_faz, pred_heatmap]
                for seg_name, seg_func, layer_idx in zip(metrics_seg_names, metrics_seg_func, metrics_seg_layer_idx):
                    pred_seg_layer = threshold(pred_seg[layer_idx][:, 0], thred_val).type(torch.int)
                    label_seg_layer = label_seg[:, layer_idx].type(torch.int)
                    metrics_dct[seg_name] = seg_func(pred_seg_layer, label_seg_layer)

                img = img[0].cpu().detach().numpy()
                label_seg = label_seg[0].cpu().detach().numpy()
                pred_vessel = threshold(pred_vessel[0], thred_val).cpu().detach().numpy()
                pred_faz = threshold(pred_faz[0], thred_val).cpu().detach().numpy()
                pred_heatmap = pred_heatmap[0].cpu().detach().numpy()
                pred_seg = (pred_vessel[0], pred_faz[0], pred_heatmap[0])
                label_cls = label_cls.cpu().detach().numpy().item()
                # pred_cls = torch.max(pred_cls, dim=1)[1].cpu().detach().numpy().item()
                pred_cls = 0
                metrics_dct["accuracy_cls"] = 100.0 * int(label_cls == pred_cls)

                # img = img.numpy().transpose((1, 2, 0))

                save_result_sample_figure(img, (label_seg, label_cls), (pred_seg, pred_cls), sample_id.numpy().item(),
                                          "{}/{:0>4}/{:0>3}.png".format(sample_dir, epoch, i),
                                          octa_dataset.label2disease)

                for metrics_item in metrics_dct.keys():
                    metrics_total_dct[metrics_item].append(metrics_dct[metrics_item])

            for metrics_item in metrics_dct.keys():
                metrics_test_dct[metrics_item].append("{:.5f}".format(mean(metrics_total_dct[metrics_item])))

            pd.DataFrame(metrics_test_dct).to_excel(sample_dir + "/{:0>4}/metrics_val.xlsx".format(epoch))
            pd.DataFrame(metrics_train_dct).to_excel(sample_dir + "/{:0>4}/metrics_train.xlsx".format(epoch))
            # pd.DataFrame(metrics_test_dct).to_excel(sample_dir + "/metrics_val.xlsx")