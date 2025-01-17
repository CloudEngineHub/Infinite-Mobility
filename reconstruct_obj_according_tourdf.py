import bpy
import urdfpy
import numpy as np

from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.assets.utils.object import join_objects

urdf_path = "/home/pjlab/datasets/partnet_mobility/3140/mobility.urdf"
tree = urdfpy.URDF.load(urdf_path)
root = tree.base_link
links = tree.links
all_objs = []
positions = []

def get_transform_matrix():
    res = np.eye(4)
    for p in positions:
        res = p @ res
    return res

def process_obj(obj):
    co = read_co(obj)
    co = np.concatenate([co, np.ones((len(co), 1))], axis=1)
    co = co @ get_transform_matrix().T
    co = co[:, :3]
    write_co(obj, co)
    return obj

def iter_tree(r):
    p_name = r.name
    if len(r.visuals) >= 1:
        for v in r.visuals:
            positions.append(v.origin)
            g = v.geometry
            m = g.mesh
            f = m.filename
            path = urdf_path.replace('mobility.urdf', f)
            bpy.ops.wm.obj_import(filepath=path)
            obj = process_obj(bpy.context.object)
            all_objs.append(obj)
            positions.pop()
    for j in tree.joints:
        if j.parent == p_name:
            c_name = j.child
            print(p_name, '->', c_name)
            positions.append(j.origin)
            for l in links:
                if l.name == c_name:
                    iter_tree(l)
            positions.pop()

iter_tree(root)
res = join_objects(all_objs)
res.name = "whole"
bpy.ops.object.select_all(action='DESELECT')
res.select_set(True)
bpy.ops.wm.obj_export(
    filepath='outputs/res.obj',
    path_mode="COPY",
    export_materials=True,
    export_pbr_extensions=True,
    export_eval_mode="DAG_EVAL_RENDER",
    export_selected_objects=True,
)
