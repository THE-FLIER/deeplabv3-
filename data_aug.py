import os

# 指定你的文件夹路径
folder_path = "datasets/data/VOC_661/JPEGImages"

# 指定你想要写入的文件名
output_file = "datasets/data/train_aug.txt"

# 获取文件夹中所有的.jpg文件
jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 提取文件名前缀并排序
prefixes = sorted([f.split('.')[0] for f in jpg_files])

# 将前缀写入到新的文件中
with open(output_file, 'w') as f:
    for prefix in prefixes:
        f.write(prefix + '\n')
