import infinigen
import gin
import importlib
import os

from infinigen.assets.utils.object import (
    join_objects_save_whole,
    save_file_path_obj,
    save_obj_parts_add,
    add_joint,
    get_joint_name
)
import shutil
import random

OBJECTS_PATH = infinigen.repo_root() / "infinigen/assets/objects"
assert OBJECTS_PATH.exists(), OBJECTS_PATH
from tqdm import tqdm

with open('./test.txt','r',encoding='utf8')as fp:
    objs = fp.readlines().split('\n')

for obj in tqdm(objs):
    items = obj.split(' ')
    fac_name = items[0]
    fac = obj['generator']
    fac_name = fac.split('(')[0]
    fac_seed = items[1]
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
    path = f'/outputs/scene/objects/{obj['generator']}_{random.randint(0,100000)}'
    if not os.path.exists(path):
        os.makedirs(path)
    asset = fac.create_asset(i=fac_seed, save_urdf=True, path=path)
    if not os.path.exists(f'{path}/{fac_seed}/scene.urdf'):
        shutil.rmtree(f'{path}/{fac_name}/{fac_seed}')
        fac.finalize_asset(asset)
        save_obj_parts_add(asset, path=path, name='obj', idx=fac_seed, first=True, use_bpy=True)
        join_objects_save_whole(asset, path=path, name='obj', use_bpy=True, idx=fac_seed)
    with open("urdfs.txt","a") as file:
            file.write(f"{path}/{fac_seed}/scene.urdf {items[2]} {items[3]} {items[4]} {items[5]} {items[6]} {items[7]}\n")
    
        
        

        