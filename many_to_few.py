from pathlib import Path

garbage_category_mapping = {
        0: 2,
        1: 3,
        2: 0,
        3: 1,
        4: 3,
        5: 2,
        6: 3,
        7: 3,
        8: 3,
        9: 2,
        10: 0,
        11: 3,
        12: 2,
        13: 3,
        14: 3,
        15: 3,
        16: 1,
        17: 3,
        18: 2,
        19: 2,
        20: 2,
        21: 2,
        22: 2,
        23: 3,
        24: 2,
        25: 2,
        26: 3,
        27: 2,
        28: 1,
        29: 1,
        30: 1,
        31: 3,
        32: 0,
        33: 0,
        34: 3,
        35: 3,
        36: 3,
        37: 3,
        38: 3,
        39: 2,
        40: 3,
        41: 3,
        42: 3,
        43: 1
    }
def modify_file(file_path):
    try:
        # 打开输入文件进行读取
        with open(file_path, 'r') as infile:
            lines = infile.readlines()

        new_lines = []
        for line in lines:
            # 分割每行的数据
            values = line.strip().split()
            # 获取第一个值
            first_value = int(values[0])

            # 根据条件修改第一个值
            first_value=garbage_category_mapping[first_value]

            # 更新第一个值
            values[0] = str(first_value)
            # 重新组合成一行
            new_line = ' '.join(values) + '\n'
            new_lines.append(new_line)

        # 打开输出文件进行写入
        with open(file_path, 'w') as outfile:
            outfile.writelines(new_lines)

        print(f"文件已成功修改并保存到 {file_path}")
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"发生未知错误: {e}")




if __name__=='__main__':
    dataset_path = Path("data/datasets/waste_classification/")  # replace with 'path/to/dataset' for your custom data
    labels = sorted(dataset_path.rglob("*labels/*.txt"))  # all data in 'labels'

    for label in labels:
        modify_file(label)
