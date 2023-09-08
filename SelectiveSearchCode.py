# -*- coding: utf-8 -*-
import os
from skimage import util, io, feature, color, transform, segmentation
import numpy as np


# "Selective Search for Object Recognition" by J.R.R. Uijlings et al.

def _generate_segments(im_orig, scale, sigma, min_size):
    """
    根据Felzenswalb-Huttenlocher方法将图像分割为小区域图像
    :param im_orig: 输入3通道图像
    :param scale: 分割参数, 数值越小, 分割越精细
    :param sigma: 分割图像前对图像进行高斯平滑的参数
    :param min_size: 分割的最小单元, 一般设置10-100间
    :return: 带分割类别的4通道图
    """
    # 获取分割后每个小区域所属的类别
    im_mask = segmentation.felzenszwalb(util.img_as_float(im_orig), scale=scale, sigma=sigma, min_size=min_size)
    # 把类别合并到最后一个通道上, 维度为[w, h, 4]
    im_orig = np.append(im_orig, np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask
    return im_orig


def _sim_colour(r1, r2):
    """
    计算区域颜色直方图交集和
    :param r1: 区域 1
    :param r2:区域 2
    :return: 颜色直方图交集和
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
    计算区域纹理直方图交集和
    :param r1: 区域 1
    :param r2:区域 2
    :return: 颜色直方图交集和
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
    计算图像大小相似度
    :param r1: 区域 1
    :param r2: 区域 2
    :param imsize: 图像size
    :return: 图像大小相似度
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
    计算图像填充相似度
    :param r1: 区域 1
    :param r2: 区域 2
    :param imsize: 图像带下
    :return: 填充相似度结果
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    """
    整合区域相似度结果
    :param r1: 区域 1
    :param r2: 区域 2
    :param imsize: 整体相似度结果
    :return:
    """
    return _sim_colour(r1, r2) + _sim_texture(r1, r2) + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize)


def _calc_colour_hist(img):
    """
    在HSV空间计算图像颜色直方图, 输出维度为[BINS, COLOUR_CHANNELS(3)], 参考[uijlings_ijcv2013_draft.pdf]这里bins取值为25
    :param img: hsv空间图像
    :return: 颜色直方图
    """
    BINS = 25
    hist = np.array([])

    for colour_channel in (0, 1, 2):
        # extracting one colour channel
        c = img[:, colour_channel]

        # calculate histogram for each colour and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(img)
    return hist


def _calc_texture_gradient(img):
    """
    计算纹理梯度, 原始ss方法采用Gaussian导数方法, 此处采用lbp方法
    :param img: 输入图像
    :return: 和输入图像等大的lbp纹理特征
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = feature.local_binary_pattern(img[:, :, colour_channel], 8, 1.0)
    return ret


def _calc_texture_hist(img):
    """
    计算图像每个通道的纹理直方图
    :param img: 输入图像
    :return: 纹理直方图
    """
    BINS = 10
    hist = np.array([])

    for colour_channel in (0, 1, 2):
        # mask by the colour channel
        fd = img[:, colour_channel]
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = np.concatenate([hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1 Normalize
    hist = hist / len(img)
    return hist


def _extract_regions(img):
    """
    提取原始图像分割区域
    :param img: 带像素标签的4通道图像
    :return: 原始分割区域
    """
    R = {}
    # get hsv image
    hsv = color.rgb2hsv(img[:, :, :3])

    # step-1 像素位置计数
    # 遍历每个像素标签, 若其还没有被分配到一个区域, 就创建一个新的区域并将其添加到字典R中
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # bounding box
            # 根据该像素的坐标更新该区域的边界框, 即最小和最大的x和y坐标
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # step-2 计算纹理梯度
    tex_grad = _calc_texture_gradient(img)

    # step-3 计算颜色和纹理直方图
    for k, v in R.items():
        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])
    return R


def _extract_neighbours(regions):
    """
    提取给定区域之间的相邻关系
    :param regions: 输入所有区域
    :return: list-所有相交的区域对
    """
    def intersect(a, b):
        """
        判断两个区域是否相交
        :param a: 区域 a
        :param b: 区域 b
        :return: bool-区域是否相交
        """
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = regions.items()
    r = [elm for elm in R]
    R = r
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
    return neighbours


def _merge_regions(r1, r2):
    """
    区域合并并更新颜色直方图/纹理直方图/大小
    :param r1: 区域 1
    :param r2: 区域 2
    :return: 合并后的区域
    """
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selective_search(im_orig, scale=1.0, sigma=0.8, min_size=50):
    """
    选择性搜索生成候选区域
    :param im_orig: 输入3通道图像
    :param scale: 分割参数, 数值越小, 分割越精细
    :param sigma: 分割图像前对图像进行高斯平滑的参数
    :param min_size: 分割的最小单元, 一般设置10-100间
    :return: img-带有区域标签的图像(r, g, b, region), regions-字典{”rect“:(left, top, width, height), "labels":[...]}
    """
    assert im_orig.shape[2] == 3, "3ch image is expected"

    # 加载图像获取最小分割区域
    # 区域标签存储在每个像素的第四个通道 [r, g, b, region]
    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)

    # 获取相邻区域对
    neighbours = _extract_neighbours(R)

    # 计算初始相似度
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # 进行层次搜索, 直到没有新的相似度可以计算
    while S != {}:

        # 获取两最大相似度区域的下标(i, j)
        i, j = sorted(list(S.items()), key=lambda a: a[1])[-1][0]

        # 将最大相似度区域合并为一个新的区域rt
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # 标记相似度集合中与(i, j)相关的区域, 并将其移除
        key_to_delete = []
        for k, v in S.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # 移除相关区域
        for k in key_to_delete:
            del S[k]

        # 计算与新区域rt与相邻区域的相似度并添加到集合S中
        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
    return img, regions

