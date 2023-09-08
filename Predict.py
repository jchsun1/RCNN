import os
import torch
from torchvision import transforms
from SelectiveSearch import SelectiveSearch
import Utils
import numpy as np
import pandas as pd
from skimage import io


def predict(im_path, classifier, regressor, transform, device):
    """
    回归模型预测
    :param im_path: 输入图像路径
    :param classifier: 分类模型
    :param regressor: 回归模型
    :param transform: 预处理方法
    :param device: CPU/GPU
    :return: None
    """
    classifier = classifier.to(device)
    regressor = regressor.to(device)
    # 计算proposal region
    img = io.imread(im_path)
    save_name = im_path.split(os.sep)[-1]
    proposals = SelectiveSearch.cal_proposals(img=img)

    boxes, offsets = [], []
    for box in proposals:
        with torch.no_grad():
            crop = img[box[1]: box[1] + box[3], box[0]: box[0] + box[2], :]
            crop_tensor = transform(crop).unsqueeze(0).to(device)
            # 分类模型检测有物体, 才进行后续回归模型计算坐标偏移值
            out = classifier(crop_tensor)
            if torch.argmax(out).item():
                features = classifier.features(crop_tensor)
                offset = regressor(features).squeeze(0).to(device)
                offsets.append(offset)
                boxes.append(torch.tensor(box, dtype=torch.float32, device=device))

    if boxes is not None:
        offsets, boxes = torch.vstack(offsets), torch.vstack(boxes)
        # 以坐标偏移的L1范数最小作为最终box选择标准
        index = offsets.abs().sum(dim=1).argmin().item()
        boxes = boxes[index] + offsets[index]
        Utils.draw_box(img, np.array(boxes.unsqueeze(0).cpu()), save_name=save_name)
    else:
        Utils.draw_box(img, save_name=save_name)
    return None


if __name__ == "__main__":
    device = torch.device('cuda:0')
    # 加载分类模型和回归模型
    classifier_path = './model/classify.pth'
    classifier = torch.load(classifier_path)
    regressor_path = './model/regress.pth'
    regressor = torch.load(regressor_path)
    classifier.eval()
    regressor.eval()

    # transforms数据预处理
    transform_params_path = "./model/classify_transform_params.csv"
    transform_params = pd.read_csv(transform_params_path, header=None, index_col=None).values
    transform_params = [x[0] for x in transform_params]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((227, 227)),
                                    transforms.Normalize(mean=transform_params[0: 3], std=transform_params[3: 6])])

    root = "./data/source/17flowers"
    for roots, dirs, files in os.walk(root):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            img_path = os.path.join(roots, file)
            predict(im_path=img_path, classifier=classifier, regressor=regressor, transform=transform, device=device)
