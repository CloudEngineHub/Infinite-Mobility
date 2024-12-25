import infinigen
import gin
import importlib
import os

from infinigen.assets.utils.object import (
    join_objects_save_whole,
    save_file_path_obj,
    save_obj_parts_add,
    add_joint,
    get_joint_name,
    robot_tree
)
import shutil
import random
import bpy
import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

OBJECTS_PATH = infinigen.repo_root() / "infinigen/assets/objects"
assert OBJECTS_PATH.exists(), OBJECTS_PATH
from tqdm import tqdm

with open('./text.txt','r',encoding='utf8')as fp:
    objs = fp.readlines()

for obj in tqdm(objs):
    items = obj.split(' ')
    fac_name = items[0]
    fac_seed = int(items[1])
    print(fac_name)
    fac = None
    for subdir in sorted(list(OBJECTS_PATH.iterdir())):
        clsname = subdir.name.split(".")[0].strip()
        with gin.unlock_config():
            module = importlib.import_module(f"infinigen.assets.objects.{clsname}")
        if hasattr(module, fac_name):
            fac = getattr(module, fac_name)
            break
    if fac is None:
        raise ModuleNotFoundError(f"{fac_name} not Found.")
    path = f'./outputs/objects/{fac_name}_{random.randint(0,100000)}'
    fac = fac(fac_seed)
    if not os.path.exists("./outputs/objects"):
        os.makedirs("./outputs/objects")
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        placeholder = None
        if hasattr(fac, 'create_placeholder'):
            placeholder = fac.create_placeholder(i = fac_seed, path=path)
        robot_tree.clear()
        asset = fac.create_asset(i=fac_seed, save_urdf=True, path=path, placeholder=placeholder)
    except Exception as e:
        print(e)
        continue
    #asset = fac.create_asset(i=fac_seed, save_urdf=True, path=path)
    if not os.path.exists(f'{path}/{fac_seed}/scene.urdf'):
        if os.path.exists(f'{path}/{fac_seed}'):
            shutil.rmtree(f'{path}/{fac_seed}')
        fac.finalize_assets(asset)
        robot_tree.clear()
        if not asset.name in bpy.context.collection.objects.keys():
            bpy.context.collection.objects.link(asset)
        save_obj_parts_add([asset], path=path, name='obj', idx=fac_seed, first=True, use_bpy=True)
        for k in robot_tree.keys():
            if k != 0 and k != '0':
                robot_tree[0] = robot_tree[k]
                del robot_tree[k]
        join_objects_save_whole(asset, path=path, name='obj', use_bpy=True, idx=fac_seed)
    with open("urdfs.txt","a") as file:
            file.write(f"{path}/{fac_seed}/scene.urdf {items[2]} {items[3]} {items[4]} {items[5]} {items[6]} {items[7]}\n")

    if 'door' in fac_name.lower():
        path = f'./outputs/objects/casing_{random.randint(0,100000)}'
        casing_fac = fac.casing_factory
        try:
            casing = casing_fac.create_asset(i=fac_seed, save_urdf=True, path=path)
        except Exception as e:
            print(e)
            continue
        save_obj_parts_add([asset], path=path, name='obj', idx=fac_seed, first=True, use_bpy=True)
        for k in robot_tree.keys():
            if k != 0 and k != '0':
                robot_tree[0] = robot_tree[k]
                del robot_tree[k]
        join_objects_save_whole(asset, path=path, name='obj', use_bpy=True, idx=fac_seed)
        with open("urdfs.txt","a") as file:
            file.write(f"{path}/{fac_seed}/scene.urdf {items[2]} {items[3]} {items[4]} {items[5]} {items[6]} {items[7]}\n")

    robot_tree.clear()
    #input("Press Enter to continue...")
    
        
        

        