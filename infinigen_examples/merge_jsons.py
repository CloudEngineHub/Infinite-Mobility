import os
import json
import glob
import argparse

def merge_json_files_in_folder(folder_path, output_file):
    # 初始化一个空列表来存储所有字典
    merged_data = []

    # 使用glob模块捕获所有的data_infos_*.json文件
    pattern = os.path.join(folder_path, "data_infos_*.json")
    for file_path in glob.glob(pattern):
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except Exception as e:
                print("The file path", file_path, "can not be loaded")
                print(e)
                continue
            # "id": "0",
            # "obj_name": "Staircase",
            # "file_obj_path": "outputs/LShapedStaircaseFactory/LShapedStaircaseFactory/0/objs/whole.obj",
            # "part_name": "arm_part",
            # "file_name": "1.obj",
            # "file_obj_path": "outputs/AASofaFactory/SofaFactory/0/objs/1.obj"
            # 检查文件内容是否为列表且列表中只有一个字典
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                path = data[0]["part"][0]["file_obj_path"]
                path_list = path.split("/")
                if "id" not in data[0].keys():
                    data[0]["id"] = path_list[-3]
                if "obj_name" not in data[0].keys():
                    data[0]["obj_name"] = path_list[-4][:-7]
                if "file_obj_path" not in data[0].keys():
                    path_list[-1] = "whole.obj"
                    data[0]["file_obj_path"] = os.path.join(*path_list)
                # 将字典添加到合并列表中
                merged_data.append(data[0])
            else:
                print(f"Skipping file {os.path.basename(file_path)}, as it does not contain a single dictionary in a list.")

    # 将合并后的数据写入新的JSON文件
    with open(os.path.join(folder_path, output_file), 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, indent=4, ensure_ascii=False)

def process_folders(folders_file, former_path="outputs"):
    # 读取包含文件夹路径的文本文件
    with open(folders_file, 'r', encoding='utf-8') as f:
        folders = [line.strip() for line in f.readlines()]

    # 对每个文件夹执行合并操作
    for folder in folders:
        path = os.path.join(former_path, folder)
        if os.path.isdir(path):
            print(f"Processing folder: {path}")
            merge_json_files_in_folder(path, 'data_infos.json')
        else:
            print(f"Folder not found: {path}")

def main():
    parser = argparse.ArgumentParser(description="Merge JSON files in folders.")
    parser.add_argument("folders_file", help="The file containing the list of folders to process.")
    parser.add_argument("former_path", default='outputs', help="The former_path of the classes.")
    parser.add_argument("--output", default='data_infos.json', help="The name of the output JSON file.")
    
    args = parser.parse_args()

    process_folders(args.folders_file, args.former_path)

if __name__ == "__main__":
    main()