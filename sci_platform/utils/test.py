from tqdm import tqdm
import os
folder_path = "/home/bingxing2/ailab/scxlab0066/SocialScience/database/paper"
def read_txt_files_as_dict_continue(folder_path):
    dict_list = []  # 用于存储所有文件的字典
    # 遍历文件夹中所有的 .txt 文件
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):  # 确保文件是 .txt 类型
            file_path = os.path.join(folder_path, filename)

            # 打开并读取每个 .txt 文件的内容
            with open(file_path, 'r') as file:
                file_content = file.read()
                try:
                    # 将内容解析为字典（假设内容是JSON格式）
                    file_dict_old = eval(file_content)
                    file_dict={}
                    file_dict['title']=file_dict_old['title']
                    if isinstance(file_dict_old['abstract'], tuple):
                        file_dict['abstract']=file_dict_old['abstract'][0]
                    else:
                        file_dict['abstract']=file_dict_old['abstract']
                    file_dict['year']=file_dict_old['year']
                    file_dict['citation']=file_dict_old['citation']
                    file_dict['id'] = file_dict_old['id']
                    file_dict['authors'] = file_dict_old['authors']
                    file_dict['cite_papers'] = file_dict_old['cite_papers']
                    file_dict['reviews'] = file_dict_old['reviews']
                    file_dict['discipline'] = file_dict_old['discipline']
                except json.JSONDecodeError:
                    print(f"文件 {filename} 的内容不是有效的JSON格式，跳过该文件。")
                    continue

                # 将字典添加到列表
                dict_list.append(file_dict)
    return dict_list
dict = read_txt_files_as_dict_continue(folder_path)
dict.sort(key=lambda x: x['id'])
print(dict[0]['id'])
print(dict[1]['id'])
print(dict[2]['id'])
print(dict[3]['id'])
print(dict[4]['id'])
print(dict[5]['id'])
print(dict[6]['id'])
print(dict[7]['id'])
print(dict[8]['id'])
print(dict[9]['id'])
print(dict[10]['id'])
print(len(dict))