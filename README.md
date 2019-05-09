# Deep-SORT
MOT using deepsort yolo3 with C++
操作系统：Ubuntu 18.04
编译环境：Qt 5.12.2

深度学习的模型分两个，一个是目标检测，一个是目标跟踪

目标检测的模型
https://pjreddie.com/darknet/yolo/

目标跟踪中特征部分 
目标跟踪模型 mars-small128 

OpenCV的DNN加载YOLO模型，这样就不用依赖Darknet库
不依赖cuda，cudnn，这样方便环境搭建
现在目标跟踪的特征部分使用TensorFlow C++的api。如果再想轻量级一些，就要去除Tensorflow的依赖。

OpenCV的安装可以参考
https://blog.csdn.net/flyfish1986/article/details/89157368

Tensorflow的安装可以参考
https://blog.csdn.net/flyfish1986/article/details/89406211


里面使用了github作者的大量代码，站在巨人们的基础上。

如果您要使用我的代码搭建环境，您要做的是
用Qt打开工程后，更改deeplearning.pro文件的内容
主要是头文件，库文件的路径更改成您自己的文件所在路径
模型文件放置与生成文件相同的目录

本实例代码已编译通过，且正常运行。
