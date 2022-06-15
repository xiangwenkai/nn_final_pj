
神经网络期末作业

# 一、Faster-Rcnn
## 所需环境
torch == 1.2.0


VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分

## 训练步骤
### a、训练VOC07数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07的数据集，解压后放在根目录 

2. 数据集的处理   
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。
backbone参数为空时，主干网络进行随机初始化；backbone参数为"resnet50"式，主干网络采用imagenet预训练的主干网络进行初始化；backbone参数为"maskrcnn"时，主干网络使用coco预训练的主干网络初始化；

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。我们首先需要去frcnn.py里面修改model_path以及classes_path，决定预测用到的模型。   
**model_path指向训练好的权值文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。


# 二、Transformer cifar100
创建参数量约为320000的transformer模型，在cifar100上进行训练（无预训练），直接运行main.py中的代码即可。
