# ʵ�黷��
- python������python 3.9
- ʹ�õĿ⣺pytorch��torchvision
- Ӳ����GPUѵ��
# ���ݼ�����
- ʹ��cifar-10���ݼ�����60000�Ÿ��ָ����Ĳ�ɫͼƬΪѵ����
- ��ͨ��pytorch��torchvision.datasetsģ��ֱ������
# ���з�ʽ
## (1)��������
- ���Ȱ�װpytorch�Լ�torchvison������Ҫ�˽��cuda��Э����
- Ȼ��װcuda�Ա�֤�ܹ��������GPU��Դ
## (2)����
- �����ܹ��Զ�����һ��output�ļ������ڴ洢���ɵ�ͼƬ
- ȷ����������������
- ֱ����python���������д˴��뼴��
# ʵ����
- ��ʼ���� epoch=5��lrѧϰ��=0.0002���Ż�����AdamW��
- ����epoch=5ʱ��������̫�٣����ڽ�������ʹ��epoch=20Ϊ����ֵ
## ����epoch
- epoch=5ʱ��ͼ��Ч��

|  ��ͼ��   | ��ͼ��  |
|  ----  | ----  |
| ![](picture5\fake_samples_epoch_0.png) | ![](picture5\real_samples_epoch_0.png) |
| ![](picture5\fake_samples_epoch_1.png) | ![](picture5\real_samples_epoch_1.png) |
| ![](picture5\fake_samples_epoch_2.png) | ![](picture5\real_samples_epoch_2.png) |
| ![](picture5\fake_samples_epoch_3.png) | ![](picture5\real_samples_epoch_3.png) |
| ![](picture5\fake_samples_epoch_4.png) | ![](picture5\real_samples_epoch_4.png) |

- epoch=10ʱ��ͼ��Ч��

|  ��ͼ��   | ��ͼ��  |
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

- epoch=20ʱ��ͼ��Ч��

|  ��ͼ��   | ��ͼ��  |
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

- ��������epoch�����ӣ�ͼƬЧ��Խ��Խ�ã�����Խ��Խ�������ܹ����Ա��
- ��ʵ�趨��epoch=50ʱ�����������ѵ��ʱ��̫���ˣ���Ȼ�ܹ�����������������������Ĳ���ҲҪ��ȥһ����ʱ�䣬���Խ������Ĳ��Ե�����ʹ��epoch=20
## ����ѧϰ��
- ����ѧϰ�ʣ�������������ѧϰ�ø�ϸ�£��������Ը���ͼ������
- ѧϰ�ʸ���Ϊ0.0001

|  ��ͼ��   | ��ͼ��  |
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
- ����Ч����������
## �����Ż���
- ���Ż����ص���Adam�Ż������鿴Ч��

|  ��ͼ��   | ��ͼ��  |
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