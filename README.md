ultralytics/
├── cfg/            # 配置文件 \
├── hub/            # 模型下载和管理\
├── models/         # 模型定义\
├── tasks/          # 任务相关代码\
├── utils/          # 工具函数\
├── train.py        # 训练脚本\
├── predict.py      # 预测脚本\
└── ...

style of yolo`s dataset

txt

content: class_id x_center y_center width height

style of dataset`s folder

├── images/            
├── labels/            
├── train.csv    
├── val.csv\
└── ...

reference：\
比赛流程借鉴：https://blog.csdn.net/weixin_43955293/article/details/121278420
比赛性能提升：https://muyun99.github.io/pages/9f35c1/#kaggle-cowboy-outfits-detection-%E7%AB%9E%E8%B5%9B%E6%96%B9%E6%A1%88%E5%AD%A6%E4%B9%A0\
比赛流程借鉴：https://blog.csdn.net/m0_67647321/article/details/135538605
Ultralytics官网：https://docs.ultralytics.com/zh
runs结果分析：https://blog.csdn.net/qq_63075864/article/details/147011368?spm=1001.2014.3001.5502
运行train方法时的日志输出：https://blog.csdn.net/qq_63075864/article/details/147014466?spm=1001.2014.3001.5502
train参数详解：https://blog.csdn.net/qq_63075864/article/details/147014740?spm=1001.2014.3001.5502
val参数详解：https://blog.csdn.net/qq_63075864/article/details/147029113?spm=1001.2014.3001.5502
predict参数详解：https://blog.csdn.net/qq_63075864/article/details/147069395?spm=1001.2014.3001.5502
SMOTE优化不平衡采样：https://blog.csdn.net/weixin_46287760/article/details/136085527
result-analysis:
![](note/img/epoch_output.png)

the content of folder 'runs'

train\
├── weights/ \
│   ├── best_model.pth   # 训练时表现最好的权重文件 \
│   └── last_model.pth   # 最后更新的权重文件 \
├── args.yaml            #model.train(args)里边的所有参数 \
├── confusion_matrix         #混淆矩阵，存放 \
├── tasks/          # \
├── utils/          # \
├── train.py        # \
├── predict.py      # \
└── ...
    

train方法中的resume参数不能乱用，网上说有什么bug，我的理解是resume=true后，会去runs文件夹下找那次训练的last.pt，然后继续训练，有个叫event.out.tfevents...的文件，不会那个就是断点吧，但是我有那个文件的情况下，运行也报莫名其妙的错了。

样本极度不均衡怎么解决？
1.smote（不过我还是没弄明白要怎么用，说什么要用欧氏距离弄出临近点，但是根据框的四个变量，要怎么弄）
2.纯粹把类少的框复制一定比例，增加样本
3.把存在少数类的图片进行数据增广，产生新的图片，然后把这些图片也作为样本（这些图片只存在少数类的框，不要多数类的框，不然都一起变多了，就没意义了）