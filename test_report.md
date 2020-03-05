# CORNER 实验报告
## dataloader
对于不同数据集载入的部分，在DATA文件夹中，针对不同的数据集，路径规则不同的话，需要修改对应的载入路径
需要train_list.txt的，可以用指令：

当前文件夹的所有文件的文件名（按行）写入 txt 文件：
ls -R  *.jpg > file.txt

## 相对原repo的修改
由于原repo相对版本比较早，pytorch的版本应该在0.5之前，为了能在1.1之后的pytorch版本运行，修改了train脚本里的loss打印，同时layers/modules/文件夹下的multibox_loss进行了修改。

## 训练脚本
虽然原训练脚本有arg接口，不过为了方便，训练参数都直接写在了arg的默认参数里，也可以在调用脚本的时候写参数，
如果想要重新开始训练，确保weights文件夹下有vgg16_reducedfc.pth初始权重文件。然后在默认里将--resume默认参数赋值为''
在重新训练初始权重文件的情况下，使用2020/2/25我修改吼的loss文件，在第700iter，会报错：
invalid argument 2: non-empty vector or matrix expected
具体原因还在查


## OCR数据集
OCR数据集都存放在服务器/data/OCR/下，其中coco-text还写脚本对GT进行提取，synthtext已经完成了代码处理，其余数据集都有可读取的gt，可能会少train_list.txt。可以使用上文dataloader中的指令来生成。


## 实验过程记录
**2020/2/26**

2080ti为11G显存，batch设为6的情况下现存占用9.6g
由于之前在训练过程中（batch为2），loss计算出现了invalid argument 2: non-empty vector or matrix expected，但是使用synth预训练的权重transfer到mlt数据集上，65个epoch都没有出问题。transfer到65个epoch后，loss在1~2之间跳动。
今天把batch设到6后，再次使用基础网络重新训练，12个epoch后还未出错，由于权重梯度随机下降，没有复现出之前的bug，暂时没有复现BUG思路，ORZ

为了保证各个数据集的试验部覆盖，在调用脚本时，--save_folder代表了结果保存的一级目录，会在这一集目录下，按照--name参数中的名称再单独创建文件夹进行保存

**2020/3/2**

单独用mlt数据集训练，在130epoch后，loss会相对稳定的在1~2之间跳动

train脚本里的synth载入有问题，可以换成新的SSDAugmentation((ssd_dim, ssd_dim), means),这里需要将输入尺寸设为数组：(ssd_dim, ssd_dim)。原脚本里这里设为了SSDAugmentation(ssd_dim, means),这里会由于后面引用的是数组造成报错。
utils里的augmentations，两个差不多，可以都换到augmentations_poly，poly只是比synth多了几个方法，其他相同。具体可以看接口。
MLT和synthtext已经可以混合训练。实际数据集混合比想象中简单，pytorch已经提供了对应的dataloader的concat方法，具体代码可以惨开下文：
``` 
    synthdataset = SynthDetection(args.synth_root, SSDAugmentation(
             (ssd_dim, ssd_dim), means), AnnotationTransform())
    MLTdataset = MLTDetection(args.mlt_root, 'train', SSDAugmentation(
            (512, 512), means), AnnotationTransform())

    mergedataset = torch.utils.data.ConcatDataset([synthdataset, MLTdataset])
    data_loader = data.DataLoader(mergedataset, batch_size, num_workers=args.num_workers,
                                     shuffle=True, collate_fn=detection_collate, pin_memory=True)
```


## EAST 论文
![网络结构](https://pic1.zhimg.com/80/v2-72f5fd00d42dad74893c623362c10d64_1440w.jpg "网络结构")
以一个固定的的backbone出来的feature map后，类似于DenseBox和Unet网络中的特性，经过特征提取后，经过四个阶段的卷积层可以得到四张feature map， 分别为
f_{4},f_{3},f_{2},f_{1}； 它们相对于输入图片分别缩小1/4,1/8，1/16,1/32，之后使用上采样、concat(串联)、卷积操作依次得到h_{1},h_{2},h_{3},h_{4}，在得到h_{4}这个融合的feature map后，使用大小为3X3通道数为32的卷积核卷积得到最终的feature map。文中对文本框的定义有两种，一种是旋转矩形(RBOX)
得到最终的feature map后，使用一个大小1x1为通道数为1的卷积核得到一张score map用Fs表示。在feature map上使用一个大小为1x1通道数为4的卷积核得到text boxes，使用一个大小1x1为通道数为1的卷积核得到text rotation angle，这里text boxes和text rotation angle合起来称为geometry map用Fg表示。

关于Fs,Fg的说明：
Fs为原图的1/4，通道数为1，每个像素表示对应于原图中像素为文字的概率值，所以值在[0,1]范围内。
Fg为原图的1/4，通道数为5，即4+1(text boxes + text rotation angle)。
text boxes通道数为4，其中text boxes每个像素如果对应原图中该像素为文字，四个通道分别表示该像素点到文本框的四条边的距离，范围定义为输入图像大小，如果输入图像为512，那范围就是[0,512]。
text rotation angle通道数为1，其中text rotation angle每个像素如果对应原图中该像素为文字，该像素所在框的倾斜角度，角度范围定义为[-45,45]度。
(这里的定义和center net很像，anchurfree)

损失定义：
![损失函数](https://math.jianshu.com/math?formula=L%20%3D%20L_%7Bs%7D%20%2B%20%5Clambda_%7Bg%7DL_%7Bg%7D "损失函数")
L=Ls+k*Lg
Ls和Lg分别代表score map和geometry map的损失，k代表损失的权重，文章设为1 
1. score map的损失计算
文章使用交叉熵，持续使用dice loss
Ls = 1-((2*Ys*Ps)/(Ys+Ps))
Ys表示位置敏感图像分割(position-sensitive segmentation)的label，
Ps代表预测的分割值

2. geometry map的损失计算
采用IoU loss，计算方法如下
Lg = Laabb+k*Ltheta
k = 10

其中Laabb表示相对目标框边长上的差距，Ltheta表示角度上的差距
Laabb = -log(IoU)
Ltheta = 1 - cos(角度差)


## Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation 论文(corner)
![网络结构](https://images2018.cnblogs.com/blog/1058268/201803/1058268-20180302201238278-1510352903.png "网络结构")

- backbone：基础网络，用来特征提取（不同分支特征共享）
- corner detection：用来生成候选检测框，是一个独立的检测模块，类似于RPN的功能
- Position Sensitive Segmentation：整张图逐像素的打分，和一般分割不同的是输出4个score map，分别对应左上、左下、右上、右下不同位置的得分
- coring + NMS：综合打分，利用（2）的框和（3）的score map再综合打分，去掉非文字框，最后再接一个NMS

![网络结构](https://images2018.cnblogs.com/blog/1058268/201803/1058268-20180302202753732-970984833.png "网络结构")
- Backbone取自DSSD = VGG16(pool5) + conv6(fc6) + conv7(fc7) + 4conv + 6 deconv (with 6 residual block)
- Corner Point Detection是类似于SSD，从多个deconv的feature map上单独做detection得到候选框，然后多层的检测结果串起来nms后为最后的结果

Corner Detection
- 思路说明
    - Step1: 用DSSD框架（任何一个目标检测的框架都可以）找到一个框的四个角点，然后整张图的所有角点都放到一个集合中
    - Step2: 把集合中的所有角点进行组合得到所有候选的框
- 网络结构
    - Fi表示backbone结构中的后面几个deconv得到的feature map（每层都单独做了detection）
    - w，h是feature map大小，k是defalt box的个数，q表示角点类型，这里q = 4，即每个位置（左上、左下、右上、右下） 都能单独得到2个score map和4个offset map
- 角点信息
    - 实际上是一个正方形，正方形中心为gt框（指的是文字框）的顶点，正方形的边长 = gt框的最短边
    - corner detection对每一种角点（四种）单独输出corner box，可以看做是一个四类的目标检测问题
- 角点如何组合成文字框？
    - 由于角点不但有顶点位置信息，也有边长信息，所以满足条件的两个corner point组合起来可以确定一个文字框。
    - 具体组合思路如下： 一个rotated rectangle可以由两个顶点+垂直于两个顶点组成的边且已知长度的边来确定。（1. 由于角点的顶点类型确定，所以短边方向也是确定的，例如左上-左下连边确定短边在右边。2. 垂直的边的长度可以取两个角点的正方形边长的平均值）
    - 可以组合的两个corner point满足条件如下：
        - 角点分数阈值>0.5
        - 角点的正方形边长大小相似（边长比<1.5）
        - 框的顶点类型和位置先验信息（例如，“左上”、“左下”的角点的x应该比“右上”、“右下”小）