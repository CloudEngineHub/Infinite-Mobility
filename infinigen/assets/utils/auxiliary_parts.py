import bpy
import os
import random

from infinigen.core.util.blender import deep_clone_obj

ALL_AUXILIARY_CATAGORY = [
    "handles",
    "drawers",
]

AUXILIARY_PATH = "/home/pjlab/datasets/parts"

LOADED_AUXILIARY = {}

def all_auxiliary_catagory():
    return ALL_AUXILIARY_CATAGORY

def query_auxiliary_samples(catagory):
    if catagory not in ALL_AUXILIARY_CATAGORY:
        return None
    #if catagory == "handles":
    catagory_path = f"{AUXILIARY_PATH}/{catagory}"
    ids = os.listdir(catagory_path)
    if 'des.txt' in ids:
        ids.remove('des.txt')
    return ids
    #return None

def parse_description(path):
    res = {}
    with open(path, "r") as f:
        content = f.read().split(" ")
        for i in range(0, len(content), 2):
            res[content[i]] = content[i + 1]
    return res


def sample_auxiliary(catagory, id):
    if catagory not in ALL_AUXILIARY_CATAGORY:
        return None
    if str(id) not in query_auxiliary_samples(catagory):
        return None
    #if catagory == "handles":
    if f"{catagory}_{id}" in LOADED_AUXILIARY:
        return deep_clone_obj(LOADED_AUXILIARY[f"{catagory}_{id}"])
    catagory_path = f"{AUXILIARY_PATH}/{catagory}/{id}/whole/whole/whole.obj"
    print(catagory_path)
    bpy.ops.wm.obj_import(filepath=catagory_path)
    des = {}
    if os.path.exists(f"{AUXILIARY_PATH}/{catagory}/{id}/whole/whole/whole.txt"):
        des = parse_description(f"{AUXILIARY_PATH}/{catagory}/{id}/whole/whole/whole.txt")
    obj = bpy.context.object
    LOADED_AUXILIARY[f"{catagory}_{id}"] = deep_clone_obj(obj)
    return obj, des
    #return None

def random_auxiliary(catagory):
    if catagory not in ALL_AUXILIARY_CATAGORY:
        return None
    #if catagory == "handles":
    ids = query_auxiliary_samples(catagory)
    id = ids[random.randint(0, len(ids) - 1)]
    return sample_auxiliary(catagory, id)
    #return None



