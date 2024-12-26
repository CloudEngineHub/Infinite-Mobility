import bpy
from infinigen.assets.utils.object import (
  export_curr_scene,
  join_objects,
)
import os
from pathlib import Path

dir_path = "/home/pjlab/datasets/parts/handles/1"
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
export_curr_scene([obj], Path(f"{dir_path}/whole"), "obj", individual_export=True)
#bpy.ops.wm.obj_export(filepath="outputs/output.obj")