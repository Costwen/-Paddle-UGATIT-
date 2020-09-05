# -Paddle-UGATIT-
使用百度paddlepaddle框架 关于论文UGATIT的复现 

## 文件说明

### main.py

args类定义了训练过程和网络相关的一些超参数,主程序则实例化了一个UGATIT类,调用其训练过程

### UGATIT.py 

创建了一个UGATIT类

UGATIT.build_model 定义了两个生成器和四个判别器以及生成器和判别器的优化器

UGATIT.train中定义了网络的损失函数,训练方法,并通过不断的迭代训练网络

### networks.py 

UGATIT的主网络定义,其中包含了生成器 ResnetGenerator, 判别器 Discriminator

### reader.py

提供了将图片集转化为reader的接口,其中包含了图片的随机裁剪和随机翻转从而进行了数据增强

### loss.py 

将paddle之中的mse_loss,bce_loss封装好为更易调用的函数

### utils.py

将读取数据转化为tensor的函数 以及将tensor转化为图片的函数

### gobalvar.py

为了给rho 层编号而实现的全局函数,可以获得一个全局变量

### uzip.py

解压函数

### eval.py

评估函数,将验证集A中的图片全部转为B

### Parameters 

存放网络参数的文件夹

其中包含了训练了大约30w轮的权重文件

### Images

存放训练过程之中输出的关于验证集的效果图片

### 使用说明

通过在终端执行

```py
python main.py
```

即可开始网络的训练

其中main.py 中的 args.start 设置训练的开始轮数, args.pretrain 设置是否需要加载预训练模型 

执行

```py
python eval.py
```

可以进行验证,其中可以设置网络需要加载的权重文件

