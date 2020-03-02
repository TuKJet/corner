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
2020/2/26
2080ti为11G显存，batch设为6的情况下现存占用9.6g
由于之前在训练过程中（batch为2），loss计算出现了invalid argument 2: non-empty vector or matrix expected，但是使用synth预训练的权重transfer到mlt数据集上，65个epoch都没有出问题。transfer到65个epoch后，loss在1~2之间跳动。
今天把batch设到6后，再次使用基础网络重新训练，12个epoch后还未出错，由于权重梯度随机下降，没有复现出之前的bug，暂时没有复现BUG思路，ORZ

为了保证各个数据集的试验部覆盖，在调用脚本时，--save_folder代表了结果保存的一级目录，会在这一集目录下，按照--name参数中的名称再单独创建文件夹进行保存

2020/3/2
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
以一个固定的的backbone出来的feature map后，类似于DenseBox和Unet网络中的特性，