# 实验环境
- python环境：python 3.9
- 使用的库：pytorch、torchvision
- 硬件：GPU训练
# 数据集下载
- 使用cifar-10数据集，以60000张各种各样的彩色图片为训练库
- 可通过pytorch的torchvision.datasets模块直接下载
# 运行方式
## (1)环境配置
- 首先安装pytorch以及torchvison，并且要了解和cuda的协调性
- 然后安装cuda以保证能够充分利用GPU资源
## (2)运行
- 代码能够自动创建一个output文件夹用于存储生成的图片
- 确定参数设置无问题
- 直接在python环境里运行此代码即可
# 实验结果
- 初始参数 epoch=5，lr学习率=0.0002，优化器是AdamW（
- 由于epoch=5时样本数量太少，故在接下来将使用epoch=20为基本值
## 更改epoch
- epoch=5时的图像效果

|  假图像   | 真图像  |
|  ----  | ----  |
| ![](picture5\fake_samples_epoch_0.png) | ![](picture5\real_samples_epoch_0.png) |
| ![](picture5\fake_samples_epoch_1.png) | ![](picture5\real_samples_epoch_1.png) |
| ![](picture5\fake_samples_epoch_2.png) | ![](picture5\real_samples_epoch_2.png) |
| ![](picture5\fake_samples_epoch_3.png) | ![](picture5\real_samples_epoch_3.png) |
| ![](picture5\fake_samples_epoch_4.png) | ![](picture5\real_samples_epoch_4.png) |

- epoch=10时的图像效果

|  假图像   | 真图像  |
|  ----  | ----  |
| ![](picture10\fake_samples_epoch_0.png) | ![](picture10\real_samples_epoch_0.png) |
| ![](picture10\fake_samples_epoch_1.png) | ![](picture10\real_samples_epoch_1.png) |
| ![](picture10\fake_samples_epoch_2.png) | ![](picture10\real_samples_epoch_2.png) |
| ![](picture10\fake_samples_epoch_3.png) | ![](picture10\real_samples_epoch_3.png) |
| ![](picture10\fake_samples_epoch_4.png) | ![](picture10\real_samples_epoch_4.png) |
| ![](picture10\fake_samples_epoch_5.png) | ![](picture10\real_samples_epoch_5.png) |
| ![](picture10\fake_samples_epoch_6.png) | ![](picture10\real_samples_epoch_6.png) |
| ![](picture10\fake_samples_epoch_7.png) | ![](picture10\real_samples_epoch_7.png) |
| ![](picture10\fake_samples_epoch_8.png) | ![](picture10\real_samples_epoch_8.png) |
| ![](picture10\fake_samples_epoch_9.png) | ![](picture10\real_samples_epoch_9.png) |

- epoch=20时的图像效果

|  假图像   | 真图像  |
|  ----  | ----  |
| ![](picture20\fake_samples_epoch_0.png) | ![](picture20\real_samples_epoch_0.png) |
| ![](picture20\fake_samples_epoch_1.png) | ![](picture20\real_samples_epoch_1.png) |
| ![](picture20\fake_samples_epoch_2.png) | ![](picture20\real_samples_epoch_2.png) |
| ![](picture20\fake_samples_epoch_3.png) | ![](picture20\real_samples_epoch_3.png) |
| ![](picture20\fake_samples_epoch_4.png) | ![](picture20\real_samples_epoch_4.png) |
| ![](picture20\fake_samples_epoch_5.png) | ![](picture20\real_samples_epoch_5.png) |
| ![](picture20\fake_samples_epoch_6.png) | ![](picture20\real_samples_epoch_6.png) |
| ![](picture20\fake_samples_epoch_7.png) | ![](picture20\real_samples_epoch_7.png) |
| ![](picture20\fake_samples_epoch_8.png) | ![](picture20\real_samples_epoch_8.png) |
| ![](picture20\fake_samples_epoch_9.png) | ![](picture20\real_samples_epoch_9.png) |
| ![](picture20\fake_samples_epoch_10.png) | ![](picture20\real_samples_epoch_10.png) |
| ![](picture20\fake_samples_epoch_11.png) | ![](picture20\real_samples_epoch_11.png) |
| ![](picture20\fake_samples_epoch_12.png) | ![](picture20\real_samples_epoch_12.png) |
| ![](picture20\fake_samples_epoch_13.png) | ![](picture20\real_samples_epoch_13.png) |
| ![](picture20\fake_samples_epoch_14.png) | ![](picture20\real_samples_epoch_14.png) |
| ![](picture20\fake_samples_epoch_15.png) | ![](picture20\real_samples_epoch_15.png) |
| ![](picture20\fake_samples_epoch_16.png) | ![](picture20\real_samples_epoch_16.png) |
| ![](picture20\fake_samples_epoch_17.png) | ![](picture20\real_samples_epoch_17.png) |
| ![](picture20\fake_samples_epoch_18.png) | ![](picture20\real_samples_epoch_18.png) |
| ![](picture20\fake_samples_epoch_19.png) | ![](picture20\real_samples_epoch_19.png) |

- 发现随着epoch的增加，图片效果越来越好，像素越来越清晰，能够得以辨别
- 其实设定了epoch=50时的情况。但是训练时间太长了，虽然能够更加清楚，但是如果调整别的参数也要耗去一定的时间，所以接下来的测试调整都使用epoch=20
## 调整学习率
- 调整学习率，可以让生成器学习得更细致，这样可以改善图像质量
- 学习率更改为0.0001

|  假图像   | 真图像  |
|  ----  | ----  |
| ![](picturelrdown\fake_samples_epoch_0.png) | ![](picturelrdown\real_samples_epoch_0.png) |
| ![](picturelrdown\fake_samples_epoch_1.png) | ![](picturelrdown\real_samples_epoch_1.png) |
| ![](picturelrdown\fake_samples_epoch_2.png) | ![](picturelrdown\real_samples_epoch_2.png) |
| ![](picturelrdown\fake_samples_epoch_3.png) | ![](picturelrdown\real_samples_epoch_3.png) |
| ![](picturelrdown\fake_samples_epoch_4.png) | ![](picturelrdown\real_samples_epoch_4.png) |
| ![](picturelrdown\fake_samples_epoch_5.png) | ![](picturelrdown\real_samples_epoch_5.png) |
| ![](picturelrdown\fake_samples_epoch_6.png) | ![](picturelrdown\real_samples_epoch_6.png) |
| ![](picturelrdown\fake_samples_epoch_7.png) | ![](picturelrdown\real_samples_epoch_7.png) |
| ![](picturelrdown\fake_samples_epoch_8.png) | ![](picturelrdown\real_samples_epoch_8.png) |
| ![](picturelrdown\fake_samples_epoch_9.png) | ![](picturelrdown\real_samples_epoch_9.png) |
| ![](picturelrdown\fake_samples_epoch_10.png) | ![](picturelrdown\real_samples_epoch_10.png) |
| ![](picturelrdown\fake_samples_epoch_11.png) | ![](picturelrdown\real_samples_epoch_11.png) |
| ![](picturelrdown\fake_samples_epoch_12.png) | ![](picturelrdown\real_samples_epoch_12.png) |
| ![](picturelrdown\fake_samples_epoch_13.png) | ![](picturelrdown\real_samples_epoch_13.png) |
| ![](picturelrdown\fake_samples_epoch_14.png) | ![](picturelrdown\real_samples_epoch_14.png) |
| ![](picturelrdown\fake_samples_epoch_15.png) | ![](picturelrdown\real_samples_epoch_15.png) |
| ![](picturelrdown\fake_samples_epoch_16.png) | ![](picturelrdown\real_samples_epoch_16.png) |
| ![](picturelrdown\fake_samples_epoch_17.png) | ![](picturelrdown\real_samples_epoch_17.png) |
| ![](picturelrdown\fake_samples_epoch_18.png) | ![](picturelrdown\real_samples_epoch_18.png) |
| ![](picturelrdown\fake_samples_epoch_19.png) | ![](picturelrdown\real_samples_epoch_19.png) |
- 发现效果略有提升
## 更改优化器
- 将优化器回调至Adam优化器，查看效果

|  假图像   | 真图像  |
|  ----  | ----  |
| ![](pictureAdam\fake_samples_epoch_0.png) | ![](pictureAdam\real_samples_epoch_0.png) |
| ![](pictureAdam\fake_samples_epoch_1.png) | ![](pictureAdam\real_samples_epoch_1.png) |
| ![](pictureAdam\fake_samples_epoch_2.png) | ![](pictureAdam\real_samples_epoch_2.png) |
| ![](pictureAdam\fake_samples_epoch_3.png) | ![](pictureAdam\real_samples_epoch_3.png) |
| ![](pictureAdam\fake_samples_epoch_4.png) | ![](pictureAdam\real_samples_epoch_4.png) |
| ![](pictureAdam\fake_samples_epoch_5.png) | ![](pictureAdam\real_samples_epoch_5.png) |
| ![](pictureAdam\fake_samples_epoch_6.png) | ![](pictureAdam\real_samples_epoch_6.png) |
| ![](pictureAdam\fake_samples_epoch_7.png) | ![](pictureAdam\real_samples_epoch_7.png) |
| ![](pictureAdam\fake_samples_epoch_8.png) | ![](pictureAdam\real_samples_epoch_8.png) |
| ![](pictureAdam\fake_samples_epoch_9.png) | ![](pictureAdam\real_samples_epoch_9.png) |
| ![](pictureAdam\fake_samples_epoch_10.png) | ![](pictureAdam\real_samples_epoch_10.png) |
| ![](pictureAdam\fake_samples_epoch_11.png) | ![](pictureAdam\real_samples_epoch_11.png) |
| ![](pictureAdam\fake_samples_epoch_12.png) | ![](pictureAdam\real_samples_epoch_12.png) |
| ![](pictureAdam\fake_samples_epoch_13.png) | ![](pictureAdam\real_samples_epoch_13.png) |
| ![](pictureAdam\fake_samples_epoch_14.png) | ![](pictureAdam\real_samples_epoch_14.png) |
| ![](pictureAdam\fake_samples_epoch_15.png) | ![](pictureAdam\real_samples_epoch_15.png) |
| ![](pictureAdam\fake_samples_epoch_16.png) | ![](pictureAdam\real_samples_epoch_16.png) |
| ![](pictureAdam\fake_samples_epoch_17.png) | ![](pictureAdam\real_samples_epoch_17.png) |
| ![](pictureAdam\fake_samples_epoch_18.png) | ![](pictureAdam\real_samples_epoch_18.png) |
| ![](pictureAdam\fake_samples_epoch_19.png) | ![](pictureAdam\real_samples_epoch_19.png) |