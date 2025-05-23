import os
import json
from tqdm import tqdm

# convert the bounding box from COCO to YOLO format.

def cc2yolo_bbox(img_width, img_height, bbox):
    dw = 1. / img_width
    dh = 1. / img_height
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def coco_to_yolo(json_file_path,yolo_anno_path):
    # Firstly, We will transfer the coco format to yolo format

    data = json.load(open(json_file_path, 'r'))
    if not os.path.exists(yolo_anno_path):
        os.makedirs(yolo_anno_path)

    # 需要注意下因为我们的annotation label是不连续的,会导致后面报错,所以这里生成一个map映射
    cate_id_map = {}
    num = 0
    for cate in data['categories']:
        cate_id_map[cate['id']] = num
        num += 1

    # transfer the annotation, and generated a train dataframe file
    f = open(os.path.dirname(os.path.dirname(yolo_anno_path))+'/train.csv', 'w')#在labels文件夹的上一级目录建立train.csv
    f.write('id,file_name\n')
    for i in tqdm(range(len(data['images']))):
        filename = data['images'][i]['file_name']
        img_width = data['images'][i]['width']
        img_height = data['images'][i]['height']
        img_id = data['images'][i]['id']
        yolo_txt_name = filename.split('.')[0] + '.txt'  # remove .jpg

        f.write('{},{}\n'.format(img_id, filename))
        yolo_txt_file = open(os.path.join(yolo_anno_path, yolo_txt_name), 'w')

        for anno in data['annotations']:
            if anno['image_id'] == img_id:#一个图片能有多个标注
                yolo_bbox = cc2yolo_bbox(img_width, img_height, anno['bbox'])  # "bbox": [x,y,width,height]
                yolo_txt_file.write(
                    '{} {} {} {} {}\n'.format(cate_id_map[anno['category_id']], yolo_bbox[0], yolo_bbox[1],
                                              yolo_bbox[2], yolo_bbox[3]))
        yolo_txt_file.close()
    f.close()



