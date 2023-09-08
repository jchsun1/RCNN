import os
import numpy as np
import pandas as pd
import random
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def progress_bar(total: int, finished: int, length: int = 50):
    """
    进度条
    :param total: 任务总数
    :param finished: 已完成数量
    :param length: 进度条长度
    :return: None
    """
    percent = finished / total
    arrow = "-" * int(percent * length) + ">"
    spaces = "▓" * (length - len(arrow))
    end = "\n" if finished == total else ""
    print("\r进度: {0}% [{1}] {2}|{3}".format(int(percent * 100), arrow + spaces, finished, total), end=end)
    return


def cal_IoU(boxes: np.ndarray, gt_box) -> np.ndarray:
    """
    计算推荐区域与真值的IoU
    :param boxes: 推荐区域边界框, n*4维数组, 列对应左上和右下两个点坐标[x1, y1, w, h]
    :param gt_box: 真值, 对应左上和右下两个点坐标[x1, y1, w, h]
    :return: iou, 推荐区域boxes与真值的IoU结果
    """
    # 复制矩阵防止直接引用修改原始值
    bbox = boxes.copy()
    gt = gt_box.copy()

    # 将宽度转换成坐标
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    gt[2] = gt[0] + gt[2]
    gt[3] = gt[1] + gt[3]

    box_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    inter_w = np.minimum(bbox[:, 2], gt[2]) - np.maximum(bbox[:, 0], gt[0])
    inter_h = np.minimum(bbox[:, 3], gt[3]) - np.maximum(bbox[:, 1], gt[1])

    inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
    union = box_area + gt_area - inter
    iou = inter / union
    return iou


def cal_norm_params(root):
    """
    计算数据集归一化参数
    :param root: 待计算数据文件路径
    :return: 数据集的RGB分量均值和标准差
    """
    # 计算RGB分量均值
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((227, 227))])
    data = ImageFolder(root=root, transform=transform)
    m_r, m_g, m_b, s_r, s_g, s_b, = 0, 0, 0, 0, 0, 0
    print('正在计算数据集RGB分量均值和标准差...')
    for idx, info in enumerate(data):
        img = info[0]
        avg = torch.mean(img, dim=(1, 2))
        std = torch.std(img, dim=(1, 2))
        m_r += avg[0].item()
        m_g += avg[1].item()
        m_b += avg[2].item()
        s_r += std[0].item()
        s_g += std[1].item()
        s_b += std[2].item()
        progress_bar(total=len(data), finished=idx + 1)

    m_r = round(m_r / idx, 3)
    m_g = round(m_g / idx, 3)
    m_b = round(m_b / idx, 3)
    s_r = round(s_r / idx, 3)
    s_g = round(s_g / idx, 3)
    s_b = round(s_b / idx, 3)
    norm_params = [m_r, m_g, m_b, s_r, s_g, s_b]
    return norm_params


def Alexnet(pretrained=True, num_classes=2):
    """
    获取AlexNet模型结构
    :param pretrained: 是否加载预训练参数
    :param num_classes: 目标类别数
    :return: AlexNet
    """
    net = models.alexnet(pretrained=pretrained)
    net.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
    return net


def show_predict(dataset, network, device, transform=None, save: str = None):
    """
    显示微调分类模型结果
    :param dataset: 数据集
    :param network: 模型结构
    :param device: CPU/GPU
    :param transform: 数据预处理方法
    :param save: str-保存图像文件的名称, None-不保存
    :return:
    """
    network.eval()
    network.to(device)

    plt.figure(figsize=(30, 30))
    for i in range(12):
        im_path, label = random.choice(dataset.flist)
        name = im_path.split(os.sep)[-1]
        img = io.imread(im_path)
        if transform is not None:
            in_tensor = transform(img).unsqueeze(0).to(device)
        else:
            in_tensor = torch.tensor(img).unsqueeze(0).to(device)

        output = network(in_tensor)
        predict = torch.argmax(output)
        plt.subplot(2, 6, i + 1)
        plt.imshow(img)
        plt.title("{name}\ntruth:{label}\npredict:{predict}".format(name=name, label=label, predict=predict))
    plt.tight_layout()
    if save is not None:
        plt.savefig("./model/predict_" + save + ".jpg")
    plt.show()


def draw_box(img, boxes=None, save_name: str = None):
    """
    在图像上绘制边界框
    :param img: 输入图像
    :param boxes: bbox坐标, 列分别为[x, y, w, h]
    :param save_name: 保存bbox图像名称, None-不保存
    :return: None
    """
    plt.imshow(img)
    axis = plt.gca()
    if boxes is not None:
        for box in boxes:
            rect = patches.Rectangle((int(box[0]), int(box[1])), int(box[2]), int(box[3]), linewidth=1, edgecolor='r', facecolor='none')
            axis.add_patch(rect)
    if save_name is not None:
        os.makedirs("./predict", exist_ok=True)
        plt.savefig("./predict/" + save_name + ".jpg")
    plt.show()
    return None


class RegressDataSet(Dataset):
    def __init__(self, ss_csv_path, gt_csv_path, network, device, transform=None):
        """
        生成回归数据集
        :param ss_csv_path: 存储ss-bbox的文件路径
        :param gt_csv_path: 存储gt-bbox的文件路径
        :param network: 特征提取网络
        :param device: CPU/GPU
        :param transform: 数据预处理方法
        """
        self.ss_csv = pd.read_csv(ss_csv_path, header=None, index_col=None)
        self.gt_csv = pd.read_csv(gt_csv_path, header=0, index_col=None)
        self.gt_csv = self.rename()
        self.network = network
        self.device = device
        self.transform = transform

    def rename(self):
        """
        重命名gt_csv的name对象
        :return: gt_csv
        """
        for idx in range(self.gt_csv.shape[0]):
            fullname = self.gt_csv.iat[idx, 0]
            name = fullname.split("/")[-1]
            self.gt_csv.iat[idx, 0] = name
        return self.gt_csv

    def __getitem__(self, index):
        ss_img_path, *ss_loc = self.ss_csv.iloc[index, :5]
        target_name = ss_img_path.split(os.sep)[-1].rsplit("_", 1)[0] + ".jpg"
        gt_loc = self.gt_csv[self.gt_csv.name == target_name].iloc[0, 2: 6].tolist()
        label = torch.tensor(gt_loc, dtype=torch.float32) - torch.tensor(ss_loc, dtype=torch.float32)

        ss_img = io.imread(ss_img_path)
        ss_img = self.transform(ss_img).to(self.device).unsqueeze(0)
        ss_features = self.network.features(ss_img).squeeze(0)
        return ss_features, label

    def __len__(self):
        return len(self.ss_csv)


class DataSet(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.flist = self.get_flist()

    def get_flist(self):
        """
        获取数据路径及标签列表
        :return: flist-数据路径和对应标签列表
        """
        flist = []
        for roots, dirs, files in os.walk(self.root):
            for file in files:
                if not file.endswith(".jpg"):
                    continue
                im_path = os.path.join(roots, file)
                im_label = int(im_path.split(os.sep)[-2])
                flist.append([im_path, im_label])
        return flist

    def __getitem__(self, index):
        path, label = self.flist[index]
        img = io.imread(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.flist)

