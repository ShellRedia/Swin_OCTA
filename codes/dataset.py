from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal

def points_to_gaussian_heatmap(centers, height, width, scale):
    gaussians = []
    for y, x in centers:
        s = np.eye(2) * scale
        g = multivariate_normal(mean=(x, y), cov=s)
        gaussians.append(g)

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.stack([xx.ravel(), yy.ravel()]).T

    # evaluate kernels at grid points
    zz = sum(g.pdf(xxyy) for g in gaussians)

    img = zz.reshape((height, width))

    return img / np.max(img)

class octa500_Dataset(Dataset):
    def __init__(self, fov="3M"):
        image_dir_OPL_BM = r"F:\datasets\octa500\OCTA-500\OCTA_{}\Projection Maps\OCTA(OPL_BM)".format(fov)
        image_dir_ILM_OPL = r"F:\datasets\octa500\OCTA-500\OCTA_{}\Projection Maps\OCTA(ILM_OPL)".format(fov)

        seg_label_dir = r"F:\datasets\octa500\OCTA-500\OCTA_{}\GroundTruth".format(fov)
        landmark_dir = r"F:\datasets\octa500\OCTA-500\OCTA_{}\Keypoints".format(fov)
        disease_label_excel_path = r"F:\datasets\octa500\OCTA-500\OCTA_{}\TextLabels.xlsx".format(fov)

        self.image_size = 304

        self.images_path_OPL_BM = sorted([image_dir_OPL_BM + "/" + x for x in os.listdir(image_dir_OPL_BM)])
        self.images_path_ILM_OPL = sorted([image_dir_ILM_OPL + "/" + x for x in os.listdir(image_dir_ILM_OPL)])

        self.seg_label_path = sorted([seg_label_dir + "/" + x for x in os.listdir(seg_label_dir)])

        self.heatmap_labels = []
        for landmark_file_path in sorted([landmark_dir + "/" + x for x in os.listdir(landmark_dir)]):
            landmark_file = open(landmark_file_path)
            pos = [[int(float(p) * self.image_size) for p in x.split()][::-1] for x in landmark_file.readlines()[1:]]
            self.heatmap_labels.append(points_to_gaussian_heatmap(pos, self.image_size, self.image_size, scale=21))

        self.id_strs = pd.read_excel(disease_label_excel_path)["ID"]
        disease_strs = pd.read_excel(disease_label_excel_path)["Disease"]
        disease_types = sorted(list(set(disease_strs)))
        disease_type_dct = dict(zip(disease_types, range(len(disease_types))))
        print("Octa500-{} diseases :".format(fov))
        for k, v in disease_type_dct.items():
            print("{} : {}".format(k, v))
        self.classification_labels = [disease_type_dct[x] for x in disease_strs]
        self.classify_num = len(disease_type_dct)
        self.label2disease = {v : k for k, v in disease_type_dct.items()}

    def __len__(self):
        return len(self.images_path_OPL_BM)

    def __getitem__(self, index):
        img_OPL_BM = Image.open(self.images_path_OPL_BM[index])
        img_ILM_OPL = Image.open(self.images_path_ILM_OPL[index])

        seg_label = Image.open(self.seg_label_path[index])
        classify_label = self.classification_labels[index]
        heatmap_label = self.heatmap_labels[index]

        img_OPL_BM = transforms.ToTensor()(img_OPL_BM)
        img_ILM_OPL = transforms.ToTensor()(img_ILM_OPL)

        seg_label = transforms.ToTensor()(seg_label)
        seg_vessel_label = torch.where(seg_label > 0.7, torch.ones(seg_label.shape), torch.zeros(seg_label.shape))
        seg_faz_label = torch.where(seg_label <= 0.7, seg_label, torch.zeros(seg_label.shape))
        seg_faz_label = torch.where(seg_faz_label > 0.2, torch.ones(seg_label.shape), torch.zeros(seg_label.shape))

        heatmap_label = torch.unsqueeze(torch.tensor(heatmap_label, dtype=torch.float32), dim=0)
        seg_label = torch.cat([seg_vessel_label, seg_faz_label, heatmap_label], dim=0)

        return torch.cat((img_ILM_OPL, img_OPL_BM), dim=0), seg_label, classify_label, self.id_strs[index]

# if __name__=="__main__":
#     img, label_seg, label_classify = octa500_Dataset()[0]
#     print(img.shape)