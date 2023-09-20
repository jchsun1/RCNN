# RCNN
Object detection using RCNN on the 17flowers dataset.

# 博客
算法原理及思路请参考本人博客：https://www.cnblogs.com/Haitangr/p/17690028.html

# 数据集
17flowers据集, 官网下载地址：http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

# 文件说明
data						        数据目录，data/source用于存放原始数据，data/ss用于存放选择性搜索生成数据\
doc							        文档目录，用于存放参考论文\
Predict.py					    预测代码，用于最终的目标检测\
requirements.txt			  环境依赖\
SelectiveSearch.py			项目的选择性搜索实现\
SelectiveSearchCode.py	选择性搜索源码\
Train.py					      各阶段模型训练代码\
Utils.py					      用于IoU计算、数据集加载、基本模型加载、图像绘制等功能的实现\

# 程序运行流程
(1) 运行SelectiveSearch.py, 根据2flowers数据和标注在data/ss目录下生成推荐区域数据集和对应的标注信息\
(2) 运行Train.py，程序会依次在17flowers上预训练，在ss数据集上进行微调和回归模型训练，并生成./model/路径保存模型\
(3) 运行Predict.py，对数据集进行预测，生成./predict/目录保存预测结果\

# 结果展示
(1) pretrain结果\
![pretrain_curve](https://github.com/jchsun1/RCNN/assets/127655914/e24db6c8-fd83-4396-8d0e-92f64378a533)\
![predict_pretrain](https://github.com/jchsun1/RCNN/assets/127655914/0b06db6e-b1d5-48e3-a6b1-c4c902212110)\
(2) classify结果\
![classify_curve](https://github.com/jchsun1/RCNN/assets/127655914/7924722e-08fb-4941-a3f4-ad401293485b)\
![predict_classify](https://github.com/jchsun1/RCNN/assets/127655914/6c734d3f-8cd7-4891-b09c-fc34c928ad8c)\
(3) regress结果\
![regress_curve](https://github.com/jchsun1/RCNN/assets/127655914/6df892f8-7196-4d23-922d-c6831e299ba3)\
(4) predict结果\
![image_0001 jpg](https://github.com/jchsun1/RCNN/assets/127655914/a84ff3df-306d-4220-82d6-6ecb2d2303cd)\

