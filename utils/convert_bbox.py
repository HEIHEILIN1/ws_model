def convert_bbox_yolo_to_xyxy(bbox, image_width, image_height):
    """
    将 (x_center, y_center, width, height) 格式的边界框转换为 (x1, y1, x2, y2) 格式，
    并确保坐标不超出图像范围。

    :param bbox: (x_center, y_center, width, height) 格式的边界框，坐标为未归一化值
    :param image_width: 图像的宽度
    :param image_height: 图像的高度
    :return: (x1, y1, x2, y2) 格式的边界框，坐标为像素值
    """
    x_center, y_center, width, height = bbox

    # 反归一化
    # x_center *= image_width
    # y_center *= image_height
    # width *= image_width
    # height *= image_height

    # 计算左上角和右下角坐标
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # 确保坐标不超出图像范围
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))

    return [x1, y1, x2, y2]

def convert_xyxy_to_center_wh(bbox):
    """
    将 (x1, y1, x2, y2) 格式的边界框转换为 (center_x, center_y, w, h) 格式
    :param bbox: (x1, y1, x2, y2) 格式的边界框
    :return: (center_x, center_y, w, h) 格式的边界框
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [center_x, center_y, w, h]
