import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
import torch
import numpy as np
import albumentations as alb

#保存样本以及结果
def save_result_sample_figure(input_img, label, pred, sample_id, save_file_name, label2disease):
    label_seg, label_classify = label
    pred_seg, pred_classify = pred

    img_ILM_OPL, img_OPL_BM = input_img
    pred_vessel_seg, pred_faz_seg, pred_heatmap = pred_seg
    label_vessel_seg, label_faz_seg, label_heatmap = label_seg
    img_labels = "img_ILM_OPL, pred_vessel_seg, pred_faz_seg, pred_heatmap, img_OPL_BM, label_vessel_seg, label_faz_seg, label_heatmap".split(",")
    img_lst = [img_ILM_OPL, pred_vessel_seg, pred_faz_seg, pred_heatmap, img_OPL_BM, label_vessel_seg, label_faz_seg, label_heatmap]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    plt.suptitle("{} - pred : {}, gt : {}".format(sample_id, label2disease[pred_classify], label2disease[label_classify]))
    # 循环遍历所有子图
    for i in range(len(img_lst)):
        plt.subplot(2, 4, i+1)
        plt.imshow(img_lst[i])
        plt.axis('off')
        plt.title(img_labels[i])
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(save_file_name)
    plt.close(fig)


# 计算指标
def cal_hausdorff(pred, label):
    array1 = pred.cpu().numpy()[0]
    array2 = label.cpu().numpy()[0]
    dist1 = directed_hausdorff(array1, array2)[0]
    dist2 = directed_hausdorff(array2, array1)[0]
    hausdorff_dist = max(dist1, dist2)
    return hausdorff_dist

def cal_confusion_matrix(pred, label):
    TP = ((pred == 1) & (label == 1)).sum().item()
    FP = ((pred == 0) & (label == 1)).sum().item()
    FN = ((pred == 1) & (label == 0)).sum().item()
    TN = ((pred == 0) & (label == 0)).sum().item()
    return TP, FP, FN, TN

def cal_precision(pred, label):
    TP, FP, FN, TN = cal_confusion_matrix(pred, label)
    return TP / (TP + FP + 1e-6)

def cal_recall(pred, label):
    TP, FP, FN, TN = cal_confusion_matrix(pred, label)
    return TP / (TP + FN + 1e-6)

def cal_specificity(pred, label):
    TP, FP, FN, TN = cal_confusion_matrix(pred, label)
    return TN / (TN + FP + 1e-6)

def cal_jaccard_index(pred, label):# 计算交集和并集
    intersection = (pred & label).sum().item()
    union = (pred | label).sum().item()
    # 计算Jaccard Index
    jaccard_index = intersection / (union + 1e-6)
    return jaccard_index

def cal_dice(pred, label):
    intersection = (pred & label).sum().item()
    union = pred.sum().item() + label.sum().item()

    # 计算Dice系数
    dice = 2 * intersection / (union + 1e-6)
    return dice

# 数据增强：
transform = alb.Compose([
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.Rotate(limit=(-10, 10), p=0.5)
])

def get_argumented_batch_samples(img, label):
    img_argumented, label_seg_argumented = [], []
    # 数据增强
    for img, label_seg in zip(img, label):
        img, label_seg = img.numpy().transpose((1, 2, 0)), label_seg.numpy().transpose((1, 2, 0))
        # print("img.shape :", img.shape, img.shape[:2])
        w, h, c = img.shape
        img = np.concatenate([np.ones((w, h, 1)), img], axis=-1)
        argumented = transform(image=np.array(img * 255, dtype=np.float32),
                               mask=np.array(label_seg * 255, dtype=np.float32))
        img, label_seg = argumented["image"].transpose((2, 0, 1)) / 255, argumented["mask"].transpose((2, 0, 1)) / 255
        img = img[1:, :]
        img, label_seg = torch.tensor(img, dtype=torch.float), torch.tensor(label_seg, dtype=torch.float)
        img_argumented.append(img)
        label_seg_argumented.append(label_seg)

    return torch.stack(img_argumented), torch.stack(label_seg_argumented)

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss

class clDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.):
        super(clDiceLoss, self).__init__()
        self.smooth = smooth

    def soft_cldice_loss(self, pred, target, target_skeleton=None):
        '''
        inputs shape  (batch, channel, height, width).
        calculate clDice loss
        Because pred and target at moment of loss calculation will be a torch tensors
        it is preferable to calculate target_skeleton on the step of batch forming,
        when it will be in numpy array format by means of opencv
        '''
        cl_pred = self.soft_skeletonize(pred)
        if target_skeleton is None:
            target_skeleton = self.soft_skeletonize(target)
        iflat = self.norm_intersection(cl_pred, target)
        tflat = self.norm_intersection(target_skeleton, pred)
        intersection = (iflat * tflat).sum()
        return 1. - (2. * intersection) / (iflat + tflat).sum()

    def dice_loss(self, pred, target):
        '''
        inputs shape  (batch, channel, height, width).
        calculate dice loss per batch and channel of sample.
        E.g. if batch shape is [64, 1, 128, 128] -> [64, 1]
        '''
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss

    def soft_skeletonize(self, x, thresh_width=10):
        '''
        Differenciable aproximation of morphological skelitonization operaton
        thresh_width - maximal expected width of vessel
        '''
        for i in range(thresh_width):
            min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
            contour = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
            x = torch.nn.functional.relu(x - contour)
        return x

    def norm_intersection(self, center_line, vessel):
        '''
        inputs shape  (batch, channel, height, width)
        intersection formalized by first ares
        x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
        '''
        smooth = 1.
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)
        intersection = (clf * vf).sum(-1)
        return (intersection + smooth) / (clf.sum(-1) + smooth)

    def forward(self, pred, target):
        return 0.8 * self.dice_loss(pred, target) + 0.2 * self.soft_cldice_loss(pred, target)

# metrics :
metrics_seg_names = list("dice_vessel, jaccard_vessel, HD_vessel, precision_vessel, recall_vessel, specificity_vessel, "
                         "dice_faz, jaccard_faz, HD_faz, precision_faz, recall_faz, specificity_faz".split(","))
metrics_seg_func = [cal_dice, cal_jaccard_index, cal_hausdorff, cal_precision, cal_recall, cal_specificity] * 2
metrics_seg_layer_idx = [0] * (len(metrics_seg_func) // 2) + [1] * (len(metrics_seg_func) // 2)