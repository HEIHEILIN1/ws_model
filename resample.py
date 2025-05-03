import copy
import json
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.composition import BboxParams
import cv2
import pandas as pd

from utils.convert_bbox import convert_bbox_yolo_to_xyxy, convert_xyxy_to_center_wh

if __name__=='__main__':
    json_file_path = 'data/datasets/cowboy_outfits/train.json'
    train_csv_file_path = 'data/datasets/cowboy_outfits/train.csv'
    data = json.load(open(json_file_path, 'r'))

    # print(data['annotations'][0])#{'id': 12550146, 'image_id': 15526467552013451612, 'freebase_id': '/m/017ftj', 'category_id': 1034, 'iscrowd': False, 'bbox': [102.49, 181.12, 137.08, 97.92], 'area': 13423.09}
    # print(data['images'][0])#{'id': 9860841628484337660, 'file_name': '88d8bf3754317ffc.jpg', 'neg_category_ids': [434], 'pos_category_ids': [69, 161, 216, 277, 433], 'width': 1024, 'height': 681, 'source': 'OpenImages'}
    # print(data['categories'])

    sample_count_map = {}
    for item in data['annotations']:
        if item['category_id'] in sample_count_map:
            sample_count_map[item['category_id']] += 1
        else:
            sample_count_map[item['category_id']] = 1
    print(sample_count_map)

    # 设置自己想让每个类的样本变成多少的比例
    sampling_rate = {
        87: 10,
        1034: 1,
        131: 2,
        318: 2,
        588: 1
    }



    transform_train = A.Compose([
        # albumentations.Resize(256, 256),
        # albumentations.RandomResizedCrop(height=224,width=224, scale=(0.08, 1.0), ratio=(1.0, 1.0)),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(#
            translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
            scale=(0.9375, 1.0625),
            rotate=(-45, 45),
            p=0.5
        ),
        A.Normalize(),
        # 注释掉转 Tensor 的操作，保持 HWC 格式
        # ToTensorV2()
    ], bbox_params=BboxParams(format='coco'))




    image_folder_path = 'data/datasets/cowboy_outfits/images'
    df_train = pd.read_csv(train_csv_file_path)

    # new_annotations=data['annotations']#当 data 是字典（dict）、列表（list）、集合（set）等可变对象时，Python 会使用引用赋值。也就是说，new_annotations 和 data['annotations'] 会指向内存中的同一个对象。所以这里不能简单等号
    new_annotations = copy.deepcopy(data['annotations'])
    new_images = copy.deepcopy(data['images'])
    for item in data['annotations']:
        for i in range(sampling_rate[item['category_id']]-1):#每个框重复采样次数

            image_card=next((card for card in data['images'] if card["id"] == item['image_id']), None)

            image_name=image_card['file_name']
            #默认读进来是bgr格式，要转换一下
            image = cv2.imread(os.path.join(image_folder_path,image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_card['width'],image_card['height']))#因为默认读入是224*224的，会跟标注框大小范围不匹配，所以这里恢复成原来尺寸

            # print(item)
            # print(image_card)

            augmented=transform_train(image=image,bboxes=[item['bbox'] + [item['category_id']]])
            #给新图片赋一个id，这里赋12位数的，不太可能重复
            image_id = random.randint(100000000000, 999999999999)
            annotation_id=random.randint(10000000, 99999999)
            if augmented['bboxes']:
                new_bbox=augmented['bboxes'][0][:-1]
            else:
                break #挺奇怪的，按道理来说应该用了数据增广会产生新的标注框，但好像运行时有时候没产生

            x1, y1, x2, y2 = convert_bbox_yolo_to_xyxy(new_bbox, image_card['width'], image_card['height'])
            new_bbox=convert_xyxy_to_center_wh([x1, y1, x2, y2])#保证数据增广时标注框不出界
            new_annotation={
                'id': annotation_id,
                'image_id': image_id,
                'freebase_id': '/m/017ftj',
                'category_id': item['category_id'],
                'iscrowd': False,
                'bbox':  new_bbox,#记得把category去掉
                'area': augmented['bboxes'][0][2]*augmented['bboxes'][0][3]
            }
            new_image_name=f'new_{image_name.rsplit('.', 1)[0]}_{i}.jpg'
            new_image={
                'id': image_id,
                'file_name': new_image_name,
                'neg_category_ids': [434],#有关负类别？不太懂
                'pos_category_ids': [69, 161, 216, 277, 433],#这里我没改变这两个，因为没涉及到姿态估计，没用
                'width': image_card['width'],'height':image_card['height'],
                'source': 'OpenImages'
            }
            # 将图像从 RGB 转换回 BGR（因为 cv2.imwrite 期望 BGR 格式）
            augmented_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            augmented_image = (augmented_image * 255) #反归一化
            # print(augmented_image.shape)





            # 画的框好像不太准，不过我看了下好像就是给的样本不准
            # cv2.rectangle(augmented_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            # cv2.putText(augmented_image, str(item['category_id']), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 保存带有标注框的图片
            cv2.imwrite(os.path.join('data/datasets/cowboy_outfits/images',new_image_name), augmented_image)
            # print(new_annotation)
            # print(new_image)
            new_images.append(new_image)
            new_annotations.append(new_annotation)
            #还要在train.csv中追加id与file_name的映射
            df_train.loc[len(df_train)] = {'id':str(image_id),'file_name':new_image_name}#转成str，避免大整数出现精度损失



    data['annotations']=new_annotations
    data['images']=new_images
    sample_count_map = {}
    for item in data['annotations']:
        if item['category_id'] in sample_count_map:
            sample_count_map[item['category_id']] += 1
        else:
            sample_count_map[item['category_id']] = 1
    print(sample_count_map)

    df_train.to_csv('data/datasets/cowboy_outfits/resample_train.csv', index=False)
    with open(os.path.join(os.path.dirname(json_file_path),'resample_train.json'), 'w') as f:
        json.dump(data,f,indent=4)

