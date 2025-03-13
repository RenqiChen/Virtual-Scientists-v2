import os
import shutil

def move_files(source_folder, destination_folder):
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)

        # 移动文件
        if os.path.isfile(source_path):
            shutil.move(source_path, destination_path)
            print(f'Moved: {source_path} -> {destination_path}')

# 示例用法
source_folder = '/home/bingxing2/ailab/scxlab0066/SocialScience/database/0307_3000/paper'
destination_folder = '/home/bingxing2/ailab/scxlab0066/SocialScience/database/0308_3000/paper'
move_files(source_folder, destination_folder)