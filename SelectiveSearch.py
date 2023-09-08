# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2 as cv
import shutil
from Utils import cal_IoU
from skimage import io
import SelectiveSearchCode as Select
from multiprocessing import Process, Lock
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SelectiveSearch:
    def __init__(self, root, max_pos_regions: int = None, max_neg_regions: int = None, threshold=0.5):
        """
        采用ss方法生成候选区域文件
        :param root: 训练/验证数据集所在路径
        :param max_pos_regions: 每张图片最多产生的正样本候选区域个数, None表示不进行限制
        :param max_neg_regions: 每张图片最多产生的负样本候选区域个数, None表示不进行限制
        :param threshold: IoU进行正负样本区分时的阈值
        """
        self.source_root = os.path.join(root, 'source')
        self.ss_root = os.path.join(root, 'ss')
        self.csv_path = os.path.join(self.source_root, "gt_loc.csv")
        self.max_pos_regions = max_pos_regions
        self.max_neg_regions = max_neg_regions
        self.threshold = threshold
        self.info = None

    @staticmethod
    def cal_proposals(img, scale=200, sigma=0.7, min_size=20, use_cv=True) -> np.ndarray:
        """
        计算后续区域坐标
        :param img: 原始输入图像
        :param scale: 控制ss方法初始聚类大小
        :param sigma: ss方法高斯核参数
        :param min_size: ss方法最小像素数
        :param use_cv: (bool) true-采用cv生成候选区域, false-利用源码生成
        :return: candidates, 候选区域坐标矩阵n*4维, 每列分别对应[x, y, w, h]
        """
        rows, cols, channels = img.shape
        if use_cv:
            # 生成候选区域
            ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            proposals = ss.process()
            candidates = set()
            # 对区域进行限制
            for region in proposals:
                rect = tuple(region)
                if rect in candidates:
                    continue
                # if rect[2] * rect[3] < 500:
                #     continue
                #
                # x1, y1, w, h = rect
                # # 剔除异常区域
                # if x1 < 0 or y1 < 0 or x1 >= cols or y1 >= rows:
                #     continue
                # x2, y2 = x1 + w, y1 + h
                # if x2 < 0 or y2 < 0 or x2 >= cols or y2 >= rows:
                #     continue
                # if w == 0 or h == 0:
                #     continue
                # # 限制区域框形状和大小
                # if w / h > 2 or h / w > 2 or w / cols < 0.05 or h / rows < 0.05:
                #     continue
                candidates.add(rect)
        else:
            # ss方法返回4通道图像img_lbl, 其前三通道为rgb值, 最后一个通道表示该proposal-region在ss方法实现过程中所属的区域标签
            # ss方法返回字典regions, regions['rect']为(x, y, w, h), regions['size']为像素数,  regions['labels']为区域包含的对象的类别标签
            img_lbl, regions = Select.selective_search(im_orig=img, scale=scale, sigma=sigma, min_size=min_size)
            candidates = set()
            for region in regions:
                # excluding same rectangle with different segments
                if region['rect'] in candidates:
                    continue
                # # excluding small regions
                # if region['size'] < 220 or region['rect'][2] * region['rect'][3] < 500:
                #     continue
                #
                # x1, y1, w, h = region['rect']
                # # 剔除异常区域
                # if x1 < 0 or y1 < 0 or x1 >= cols or y1 >= rows:
                #     continue
                # x2, y2 = x1 + w, y1 + h
                # if x2 < 0 or y2 < 0 or x2 >= cols or y2 >= rows:
                #     continue
                # if w == 0 or h == 0:
                #     continue
                # # 限制区域框形状和大小
                # if w / h > 2 or h / w > 2 or w / cols < 0.05 or h / rows < 0.05:
                #     continue
                candidates.add(region['rect'])
        candidates = np.array(list(candidates))
        return candidates

    def save(self, num_workers=1, method="thread"):
        """
        生成目标区域并保存
        :param num_workers: 进程或线程数
        :param method: 多进程-process或者多线程-thread
        :return: None
        """
        self.info = pd.read_csv(self.csv_path, header=0, index_col=None)
        # label为0存储背景图, label不为0存储带目标图像
        categories = list(self.info['label'].unique())
        categories.append(0)
        for category in categories:
            folder = os.path.join(self.ss_root, str(category))
            os.makedirs(folder, exist_ok=True)
        index = self.info.index.to_list()
        span = len(index) // num_workers
        # 使用文件锁进行后续文件写入, 防止多进程或多线程由于并发写入出现的竞态条件, 即多个线程或进程同时访问和修改同一资源时，导致数据不一致或出现意外的结果
        # 获取文件锁，确保只有一个进程或线程可以执行写入操作。在完成写入操作后，释放文件锁，允许其他进程或线程进行写入。防止过程中出现错误或者空行等情况
        lock = Lock()
        # 多进程生成图像
        if "process" in method.lower():
            print("=" * 8 + "开始多进程生成候选区域图像" + "=" * 8)
            processes = []
            for i in range(num_workers):
                if i != num_workers - 1:
                    p = Process(target=self.save_proposals, kwargs={'lock': lock, 'index': index[i * span: (i + 1) * span]})
                else:
                    p = Process(target=self.save_proposals, kwargs={'lock': lock, 'index': index[i * span:]})
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        # 多线程生成图像
        elif "thread" in method.lower():
            print("=" * 8 + "开始多线程生成候选区域图像" + "=" * 8)
            threads = []
            for i in range(num_workers):
                if i != num_workers - 1:
                    thread = threading.Thread(target=self.save_proposals, kwargs={'lock': lock, 'index': index[i * span: (i + 1) * span]})
                else:
                    thread = threading.Thread(target=self.save_proposals, kwargs={'lock': lock, 'index': index[i * span: (i + 1) * span]})
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        else:
            print("=" * 8 + "开始生成候选区域图像" + "=" * 8)
            self.save_proposals(lock=lock, index=index)
        return None

    def save_proposals(self, lock, index, show_fig=False):
        """
        生成候选区域图片并保存相关信息
        :param lock: 文件锁, 防止写入文件错误
        :param index: 文件index
        :param show_fig: 是否展示后续区域划分结果
        :return: None
        """
        for row in index:
            name = self.info.iloc[row, 0]
            label = self.info.iloc[row, 1]
            # gt值为[x, y, w, h]
            gt_box = self.info.iloc[row, 2:].values
            im_path = os.path.join(self.source_root, name)
            img = io.imread(im_path)
            # 计算推荐区域坐标矩阵[x, y, w, h]
            proposals = self.cal_proposals(img=img)

            # 计算proposals与gt的IoU结果
            IoU = cal_IoU(proposals, gt_box)
            # 根据IoU阈值将proposals图像划分到正负样本集
            boxes_p = proposals[np.where(IoU >= self.threshold)]
            boxes_n = proposals[np.where((IoU < self.threshold) & (IoU > 0.1))]

            # 展示proposals结果
            if show_fig:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
                ax.imshow(img)
                for (x, y, w, h) in boxes_p:
                    rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                    ax.add_patch(rect)
                for (x, y, w, h) in boxes_n:
                    rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
                    ax.add_patch(rect)
                plt.show()

            # loc.csv用于存储带有目标图像的boxes_p边界框信息
            loc_path = os.path.join(self.ss_root, "ss_loc.csv")

            # 将正样本按照对应label存储到相应文件夹下, 并记录bbox的信息到loc.csv中用于后续bbox回归训练
            num_p = num_n = 0
            for loc in boxes_p:
                num_p += 1
                crop_img = img[loc[1]: loc[1] + loc[3], loc[0]: loc[0] + loc[2], :]
                crop_name = name.split("/")[-1].replace(".jpg", "_" + str(num_p) + ".jpg")
                crop_path = os.path.join(self.ss_root, str(label), crop_name)
                with lock:
                    # 保存的ss区域仍然为[x, y, w, h]
                    with open(loc_path, 'a', newline='') as fa:
                        fa.writelines([crop_path, ',', str(loc[0]), ',', str(loc[1]), ',', str(loc[2]), ',', str(loc[3]), '\n'])
                    fa.close()
                io.imsave(fname=crop_path, arr=crop_img, check_contrast=False)
                if self.max_pos_regions is None:
                    continue
                if num_p == self.max_pos_regions:
                    break

            # 将负样本按照存储到"./0/"文件夹下, 其bbox信息对于回归训练无用, 故不用记录
            for loc in boxes_n:
                num_n += 1
                crop_img = img[loc[1]: loc[1] + loc[3], loc[0]: loc[0] + loc[2], :]
                crop_name = name.split("/")[-1].replace(".jpg", "_" + str(num_n) + ".jpg")
                crop_path = os.path.join(self.ss_root, "0", crop_name)
                io.imsave(fname=crop_path, arr=crop_img, check_contrast=False)
                if self.max_neg_regions is None:
                    continue
                if num_n == self.max_neg_regions:
                    break
            print("{name}: {num_p}个正样本, {num_n}个负样本".format(name=name, num_p=num_p, num_n=num_n))


if __name__ == '__main__':
    data_root = "./data"
    ss_root = os.path.join(data_root, "ss")
    if os.path.exists(ss_root):
        print("正在删除{}目录下原有数据".format(ss_root))
        shutil.rmtree(ss_root)
    print("正在利用选择性搜索方法创建数据集: {}".format(ss_root))
    select = SelectiveSearch(root=data_root, max_pos_regions=None, max_neg_regions=40, threshold=0.5)
    select.save(num_workers=os.cpu_count(), method="thread")
