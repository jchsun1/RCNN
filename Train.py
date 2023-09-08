import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import Utils


def train(data_loader, network, num_epochs, optimizer, scheduler, criterion, device, train_rate=0.8, mode="classify"):
    """
    模型训练
    :param data_loader: 数据dataloader
    :param network: 网络结构
    :param num_epochs: 训练轮次
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param criterion: 损失函数
    :param device: CPU/GPU
    :param train_rate: 训练集比例
    :param mode: 模型类型, 预训练-pretrain, 分类-classify, 回归-regression
    :return: None
    """
    os.makedirs('./model', exist_ok=True)
    network = network.to(device)
    criterion = criterion.to(device)
    best_acc = 0.0
    best_loss = np.inf
    print("=" * 8 + "开始训练{mode}模型".format(mode=mode.lower()) + "=" * 8)
    batch_num = len(data_loader)
    train_batch_num = round(batch_num * train_rate)
    train_loss_all, val_loss_all, train_acc_all, val_acc_all = [], [], [], []

    for epoch in range(num_epochs):
        train_num = val_num = 0
        train_loss = val_loss = 0.0
        train_corrects = val_corrects = 0
        for step, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            # 模型训练
            if step < train_batch_num:
                network.train()
                y_hat = network(x)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算每个batch的loss结果与预测正确的数量
                label_hat = torch.argmax(y_hat, dim=1)
                # 预训练/分类模型计算loss和acc, 回归模型只计算loss
                if mode.lower() == 'pretrain' or mode.lower() == 'classify':
                    train_corrects += (label_hat == y).sum().item()
                train_loss += loss.item() * x.size(0)
                train_num += x.size(0)
            # 模型验证
            else:
                network.eval()
                with torch.no_grad():
                    y_hat = network(x)
                    loss = criterion(y_hat, y)
                    label_hat = torch.argmax(y_hat, dim=1)
                    if mode.lower() == 'pretrain' or mode.lower() == 'classify':
                        val_corrects += (label_hat == y).sum().item()
                    val_loss += loss.item() * x.size(0)
                    val_num += x.size(0)

        scheduler.step()
        # 记录loss和acc变化曲线
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        if mode.lower() == 'pretrain' or mode.lower() == 'classify':
            train_acc_all.append(100 * train_corrects / train_num)
            val_acc_all.append(100 * val_corrects / val_num)
            print("Mode:{}  Epoch:[{:0>3}|{}]  train_loss:{:.3f}  train_acc:{:.2f}%  val_loss:{:.3f}  val_acc:{:.2f}%".format(
                mode.lower(), epoch + 1, num_epochs,
                train_loss_all[-1], train_acc_all[-1],
                val_loss_all[-1], val_acc_all[-1]
            ))
        else:
            print("Mode:{}  Epoch:[{:0>3}|{}]  train_loss:{:.3f}  val_loss:{:.3f}".format(
                mode.lower(), epoch + 1, num_epochs,
                train_loss_all[-1], val_loss_all[-1]
            ))

        # 保存模型
        # 预训练/分类模型选取准确率最高的参数
        if mode.lower() == "pretrain" or mode.lower() == "classify":
            if val_acc_all[-1] > best_acc:
                best_acc = val_acc_all[-1]
                save_path = os.path.join("./model", mode + ".pth")
                # torch.save(network.state_dict(), save_path)
                torch.save(network, save_path)
        # 回归模型选取损失最低的参数
        else:
            if val_loss_all[-1] < best_loss:
                best_loss = val_loss_all[-1]
                save_path = os.path.join("./model", mode + ".pth")
                # torch.save(network.state_dict(), save_path)
                torch.save(network, save_path)

    # 绘制训练曲线
    if mode.lower() == "pretrain" or mode.lower() == "classify":
        fig_path = os.path.join("./model/", mode + "_curve.png")
        plt.subplot(121)
        plt.plot(range(num_epochs), train_loss_all, "r-", label="train")
        plt.plot(range(num_epochs), val_loss_all, "b-", label="val")
        plt.title("Loss")
        plt.legend()
        plt.subplot(122)
        plt.plot(range(num_epochs), train_acc_all, "r-", label="train")
        plt.plot(range(num_epochs), val_acc_all, "b-", label="val")
        plt.title("Acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    else:
        fig_path = os.path.join("./model/", mode + "_curve.png")
        plt.plot(range(num_epochs), train_loss_all, "r-", label="train")
        plt.plot(range(num_epochs), val_loss_all, "b-", label="val")
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    return None


def run(train_root=None, network=None, batch_size=64, criterion=None, device=None, train_rate=0.8,
        epochs=10, lr=0.001, mode="classify", show_fig=False):
    """
    模型训练
    :param train_root: 待训练数据路径
    :param network: 模型结构
    :param batch_size: batch size
    :param criterion: 损失函数
    :param device: CPU/GPU
    :param train_rate: 训练集比率
    :param epochs: 训练轮次
    :param lr: 学习率
    :param mode: 模型类型
    :param show_fig: 是否展示训练结果
    :return: None
    """

    # 判断transform参数文件是否存在
    transform_params_path = "./model/pretrain_transform_params.csv" if mode == "pretrain" else "./model/classify_transform_params.csv"
    exist = os.path.exists(transform_params_path)
    if not exist:
        print("正在计算{}模型归一化参数...".format(mode))
        transform_params = Utils.cal_norm_params(root=train_root)
        pf = pd.DataFrame(transform_params)
        pf.to_csv(transform_params_path, header=False, index=False)
    else:
        transform_params = pd.read_csv(transform_params_path, header=None, index_col=None).values
        transform_params = [x[0] for x in transform_params]

    # transforms数据预处理
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((227, 227)),
                                    transforms.Normalize(mean=transform_params[0: 3], std=transform_params[3: 6])])

    # 判断模型是否已经存在
    model_path = "./model/" + mode + ".pth"
    exist = os.path.exists(model_path)
    if not exist:
        print("目标路径下不存在{}模型".format(mode))

        # 预训练和分类模型直接加载数据文件
        if mode == "pretrain" or mode == "classify":
            optimizer = torch.optim.SGD(params=network.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
            train_set = Utils.DataSet(root=train_root, transform=transform)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            # 模型训练
            train(data_loader=train_loader, network=network, num_epochs=epochs, optimizer=optimizer, scheduler=scheduler,
                  criterion=criterion, device=device, train_rate=train_rate, mode=mode)
        # 回归模型需利用分类模型计算特征, 作为模型输入
        else:
            # 加载分类模型
            classifier = torch.load("./model/classify.pth")
            # 加载回归任务数据文件
            ss_csv_path = "./data/ss/ss_loc.csv"
            gt_csv_path = "./data/source/gt_loc.csv"
            print("正在利用微调分类模型计算特征作为回归模型的输入...")
            train_set = Utils.RegressDataSet(ss_csv_path=ss_csv_path, gt_csv_path=gt_csv_path, network=classifier,
                                             device=device, transform=transform)
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
            print("已完成回归模型数据集创建")

            # 定义线性回归模型并初始化权重
            regressor = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(256 * 6 * 6, 4))
            nn.init.xavier_normal_(regressor[-1].weight)
            optimizer = torch.optim.SGD(params=regressor.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            # 训练回归模型
            train(data_loader=train_loader, network=regressor, num_epochs=epochs, optimizer=optimizer, scheduler=scheduler,
                  criterion=criterion, device=device, train_rate=0.8, mode="regress")

        # 图像显示训练结果
        if show_fig:
            if mode != "regress":
                Utils.show_predict(dataset=train_set, network=network, device=device, transform=transform, save=mode)
    else:
        print("目标路径下已经存在{}模型".format(mode))
        if show_fig:
            network = torch.load(model_path)
            # 加载数据文件
            train_set = Utils.DataSet(root=train_root, transform=transform)
            if mode != "regress":
                Utils.show_predict(dataset=train_set, network=network, device=device, transform=transform, save=mode)
    return


if __name__ == "__main__":
    if not os.path.exists("./data/ss"):
        raise FileNotFoundError("数据不存在, 请先运行SelectiveSearch.py生成目标区域")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = [nn.CrossEntropyLoss(), nn.MSELoss()]
    model_root = "./model"
    os.makedirs(model_root, exist_ok=True)

    # 在17flowers数据集上进行预训练
    pretrain_root = "./data/source/17flowers/jpg"
    pretrain_net = Utils.Alexnet(pretrained=True, num_classes=17)
    run(train_root=pretrain_root, network=pretrain_net, batch_size=128, criterion=criterion[0], device=device,
        train_rate=0.8, epochs=15, lr=0.001, mode="pretrain", show_fig=True)

    # 在由2flowers生成的ss数据上进行背景/物体多分类训练
    classify_root = "./data/ss"
    classify_net = torch.load("./model/pretrain.pth")
    classify_net.classifier[-1] = nn.Linear(in_features=4096, out_features=3)
    run(train_root=classify_root, network=classify_net, batch_size=128, criterion=criterion[0], device=device,
        train_rate=0.8, epochs=15, lr=0.001, mode="classify", show_fig=True)

    # 在由2flowers生成的ss物体数据边界框回归训练
    run(batch_size=128, criterion=criterion[1], device=device, train_rate=0.8, epochs=50, lr=0.0001, mode="regress", show_fig=False)

