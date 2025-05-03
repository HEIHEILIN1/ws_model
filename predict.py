import pandas as pd
import ultralytics
import yaml

if __name__ == '__main__':
    model = ultralytics.YOLO('runs/detect/train/weights/best.pt')  # yaml是刚开始的模型，pt是预训练好了的模型
    #model.predict(source='data/datasets/cowboy_outfits/images/0a0a5cb609c10f09.jpg' ,cfg='./cfg/predict.yaml')#这样好像并不能读到配置文件
    with open('./cfg/predict.yaml', 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # result=model.predict(source=new_source,**config)#总感觉用csv做source会有问题，应该不是我自己的问题，难道是框架的吗
    # results=model.predict(source='data/datasets/cowboy_outfits/images_test',**config)
    results=model.predict(source='data/datasets/cowboy_outfits/images/0a0a5cb609c10f09.jpg',**config)
    for result in results:
        print(result)
        print(result.to_json())
