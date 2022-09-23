# DeepSORT

# MOT(Multi-object tracking) using yolov5 with C++ support deepsort and bytetrack


flyfish

## 前言
代码采用C++实现，目标检测支持YOLOv5 6.x,跟踪支持deepsort and bytetrack。
检测模型可以直接从YOLOv5官网，导出onnx使用
特征提取可以自己训练，导出onnx使用，onnxruntime cpu 推理，方便使用.
特征支持自定义维度例如 128,256,512等

本文源码地址

```c
https://github.com/shaoshengsong/DeepSORT
```

## deepsort v1.12
新增bytetrack跟踪

bytetrack论文
```c
http://arxiv.org/abs/2110.06864
```

bytetrack代码
```c
https://github.com/ifzhang/ByteTrack
```

## deepsort v1.1
deepsort原论文地址 

```c
https://arxiv.org/pdf/1703.07402.pdf
```


```c
MOT using deepsort yolo5 with C++
```

操作系统：Ubuntu 18.04
### 版本更新说明

去除了TensorFlow依赖
为了不依赖硬件GPU，无需cuda，cudnn，更容易编译，使用PC版本。
为了更方便编译，采用CMakeList.txt。


### 依赖的库
opencv，可以下载opencv-4.6编译安装
Eigen3安装

```c
sudo apt-get install libeigen3-dev
```

onnxruntime，可以直接解压使用，无需编译
目标检测模型下载地址

```c
https://github.com/ultralytics/yolov5
```

网盘中有已经导出完成的模型

### 文件下载
百度网盘 
链接：`https://pan.baidu.com/s/1igjNK2ty-H5AU_Ut08pkoA` 
提取码：0000
内容包括

```c
cmake-3.21.4-linux-x86_64.tar.gz  
onnxruntime-linux-x64-1.12.1.tgz
coco_80_labels_list.txt           
opencv-4.6.0.zip
DeepSORT                          
yolov5s.onnx
feature.onnx                      
yolov5x.onnx
```


### 使用方法
#### 1 onnxruntime
设置自己的onnxruntime的解压目录

```
set(ONNXRUNTIME_DIR "/home/a/lib/onnxruntime-linux-x64-1.12.1")
```


#### 2 模型配置
以下三项根据自己的需要更改
文件`tracker/deepsort/include/dataType.h`
```c
const int k_feature_dim=512;//feature dim
const std::string  k_feature_model_path ="./feature.onnx";
const std::string  k_detect_model_path ="./yolov5s.onnx";
```

#### 3 主函数
选择打开视频文件或者视频流等

```c
cv::VideoCapture capture("./1.mp4");
```

### 扩展方式
1 整体分为两部分，新增检测模块放置detector文件夹，新增跟踪模块放置tracker文件夹

## deepsort v1.0
### MOT using deepsort yolo3 with C++
操作系统：Ubuntu 18.04
编译环境：Qt 5.12.2
深度学习的模型分两块，一个是目标检测，另一个是目标跟踪
#### 目标检测的模型
地址：`https://pjreddie.com/darknet/yolo/`


#### 目标跟踪模型
mars-small128 
OpenCV DNN加载YOLO模型，不依赖Darknet库，cuda，cudnn
依赖Tensorflow，目标跟踪的特征部分使用TensorFlow C++的api。

OpenCV的安装可以参考


地址:  `https://blog.csdn.net/flyfish1986/article/details/89157368`


Tensorflow的安装可以参考

地址：`https://blog.csdn.net/flyfish1986/article/details/89406211`




[多目标跟踪论文 Deep SORT 解读](https://flyfish.blog.csdn.net/article/details/89852370)  
[多目标跟踪论文 Deep SORT 实现](https://flyfish.blog.csdn.net/article/details/90034289)  
[多目标跟踪论文 Deep SORT 数据集说明](https://flyfish.blog.csdn.net/article/details/90070639) 
[多目标跟踪论文 Deep SORT 特征提取CNN Architecture](https://flyfish.blog.csdn.net/article/details/90642532)  
[多目标跟踪论文 Deep SORT 特征训练PyTorch实现](https://flyfish.blog.csdn.net/article/details/90702620)              
[多目标跟踪论文 Deep SORT 特征训练TensorFlow实现](https://flyfish.blog.csdn.net/article/details/90379444)  
[多目标跟踪论文 Deep SORT 评测指标](https://flyfish.blog.csdn.net/article/details/90200171)  
[匈牙利算法](https://flyfish.blog.csdn.net/article/details/104298521)  
[卡尔曼滤波 - 方程组转换为矩阵形式](https://flyfish.blog.csdn.net/article/details/118635703)  
[卡尔曼滤波 - 一个方程背后的样子](https://flyfish.blog.csdn.net/article/details/118636055)  
[卡尔曼滤波 - 匀变速直线运动](https://flyfish.blog.csdn.net/article/details/118613382)  
[卡尔曼滤波 - 冥冥之中自有定数的正态分布](https://flyfish.blog.csdn.net/article/details/116067569)  
[卡尔曼滤波 - 数据融合 data fusion](https://flyfish.blog.csdn.net/article/details/118613307)  
[卡尔曼滤波 - 当前均值与上一次均值的关系](https://flyfish.blog.csdn.net/article/details/117931292)  
[卡尔曼滤波 - 状态空间模型](https://flyfish.blog.csdn.net/article/details/118636364)  
[卡尔曼滤波 - 5个公式出现的顺序](https://flyfish.blog.csdn.net/article/details/118709808)  


