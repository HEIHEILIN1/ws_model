import os

# from ultralytics.data.converter import convert_coco
from ultralytics.yolo.data.dataloaders.v5loader import autosplit
from utils.coco_to_yolo import coco_to_yolo
import pandas as pd
import json
import shutil

if __name__ == '__main__':
    datasets_path='data/datasets/waste_classification'
    json_file_path = 'data/datasets/waste_classification/annotations.json'
    yolo_anno_path = 'data/datasets/waste_classification/labels'
    #简略查看一下json内容
    with open(os.path.join(datasets_path, 'annotations.json')) as f:
        json_annotations = json.load(f)
    for key,value in json_annotations.items():
        if isinstance(value,list):
            print(f"annotations key:{key} value[0]:{value[0]}")
        else:  print(f"annotations key:{key} value:{value}")

    #建立id与类别的映射
    map_id_name={}
    for item in json_annotations['categories']:
        map_id_name[item['id']] = item['name']
    print(map_id_name)

    #建立image_id与file_name的映射
    map_imageId_fileName={}
    for item in json_annotations['images']:
        map_imageId_fileName[item['id']] = item['file_name']

    #拿到全部类别以及数量
    map_class_num={}
    sample_sum=len(json_annotations['images'])
    annotations_sum=len(json_annotations['annotations'])
    for item in json_annotations['annotations']:
        if item['category_id'] in map_class_num:
            map_class_num[item['category_id']] += 1
        else:
            map_class_num[item['category_id']]=1
    print(map_class_num)
    print(annotations_sum,sample_sum)

    #coco变yolo
    coco_to_yolo(json_file_path, yolo_anno_path)
    #划分数据集
    autosplit(os.path.join(datasets_path,'images'), weights=(0.9,0.1, 0), annotated_only=True)
    #把test的images拷贝出来，便于后面预测
    # test_file_path='data/datasets/waste_classification/autosplit_test.txt'
    # for file_name in df['file_name']:
    #     source_file='data/datasets/cowboy_outfits/images/'+file_name
    #     target_folder='data/datasets/cowboy_outfits/images_test'
    #     shutil.copy2(source_file, target_folder)






