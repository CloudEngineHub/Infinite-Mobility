import bpy
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.assets.utils.object import (
  export_curr_scene,
  join_objects,
)
import os
from pathlib import Path
import shutil
from infinigen.core.util import blender as butil
import numpy as np

def rotation(obj, x, y, z):
    obj.rotation_euler = (x, y, z)
    butil.apply_transform(obj, True)

def normalize(obj):
    co = read_co(obj)
    center = (co[:, 0].min() + co[:, 0].max()) / 2, (co[:, 1].min() + co[:, 1].max()) / 2, (co[:, 2].min() + co[:, 2].max()) / 2
    co[:, 0] -= center[0]
    co[:, 1] -= center[1]
    co[:, 2] -= center[2]
    scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
    co[:, 0] *= (1 / scale[0])
    co[:, 1] *= (1 / scale[1])
    co[:, 2] *= (1 / scale[2])
    write_co(obj, co)

# dir_path = "/home/pjlab/datasets/parts/handles"
# ids = os.listdir(dir_path)
# for id in ids:
#     path = f"{dir_path}/{id}/whole/whole/whole.obj"
#     bpy.ops.wm.obj_import(filepath=path)
#     obj = bpy.context.object
#     normalize(obj)
#     shutil.rmtree(f"{dir_path}/{id}/whole")
#     obj.name = "whole"
#     export_curr_scene([obj], Path(f"{dir_path}/{id}/whole"), "obj", individual_export=True)
# exit(0)



dir_path = "/home/pjlab/datasets/parts/drawers/3"
parts = os.listdir(dir_path)
paths = [f"{dir_path}/{part}" for part in parts if part.endswith('obj')]
objs = []
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for path in paths:
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.object
    objs.append(obj)
obj = join_objects(objs)
obj.name = "whole"
normalize(obj)
rotation(obj, np.pi / 2, 0, np.pi / 2)
export_curr_scene([obj], Path(f"{dir_path}/whole"), "obj", individual_export=True)
#bpy.ops.wm.obj_export(filepath="outputs/output.obj")