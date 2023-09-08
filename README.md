# RCNN
Object detection using RCNN on the 17flowers dataset.
# 数据集
17flowers据集, 官网下载地址：http://www.robots.ox.ac.uk/~vgg/data/flowers/17/
# 目录结构
主目录
│  Predict.py\
│  requirements.txt\
│  SelectiveSearch.py\
│  SelectiveSearchCode.py\
│  Train.py\
│  Utils.py\
│       
├─data\
│  ├─source\
│  │  │  gt_loc.csv\
│  │  │  
│  │  ├─17flowers\
│              
└──doc\
       Rich feature hierarchies for accurate object detection and semantic segmentation.pdf\
# 文件说明
requirements.txt -- 项目的环境依赖\
SelectiveSearchCode.py -- 选择性搜索源码\
SelectiveSearch.py -- 用于项目中图像推荐区域生成\
Utils.py -- 用于IoU计算、数据集加载、基本模型加载、图像绘制等功能的实现\
Train.py -- 用于各阶段模型训练\
Predict.py -- 用于最终目标检测模型的预测\
data -- 原始数据目录, 其中17flowers目录用于预训练, 2flowers用于生成推荐区域进行分类模型微调和回归模型训练, gt_loc.csv为标注2flowers数据集边界框信息\
doc -- rcnn原始论文\
# 程序运行流程
(1) 运行SelectiveSearch.py, 根据2flowers数据和标注的边界框信息gt_loc.csv文件, 在data目录下生成推荐区域ss数据集和对应的边界框信息文件ss_loc.csv\
(2) 运行Train.py，程序会依次在17flowers上预训练，在ss数据集上进行微调和回归模型训练，并生成./model/路径保存模型\
(3) 运行Predict.py，对数据集进行预测，生成./predict/目录保存预测结果\
