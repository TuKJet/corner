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