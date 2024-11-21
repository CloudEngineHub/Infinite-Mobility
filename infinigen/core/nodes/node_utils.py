# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Alexander Raistrick: primary author
# - Lahav Lipson: resample nodegroup


import bpy
import bmesh
from pathlib import Path
import os
import json
import numpy as np
from numpy.random import normal, uniform
from infinigen.core.util import blender as butil

import urdfpy

from infinigen.assets.utils.object import (
    join_objects_save_whole,
    save_file_path_obj,
    save_obj_parts_add,
)
from infinigen.core import surface
from infinigen.core.nodes.node_wrangler import (
    Nodes,
    NodeWrangler,
    geometry_node_group_empty_new,
)

# from infinigen.core.util.bevelling import (
#     add_bevel,
#     complete_bevel,
#     complete_no_bevel,
#     get_bevel_edges,
# )
# from infinigen.core.util.blender import set_geomod_inputs
from infinigen.core.util.color import random_color_mapping
from infinigen.core.util.geometry import get_geometry_data


def to_material(name, singleton):
    """Wrapper for initializing and registering materials."""

    if singleton:
        name += " (no gc)"

    def registration_fn(fn):
        def init_fn(*args, **kwargs):
            if singleton and name in bpy.data.materials:
                return bpy.data.materials[name]
            else:
                return surface.shaderfunc_to_material(fn, *args, name=name, *kwargs)

        return init_fn

    return registration_fn


def to_nodegroup(name=None, singleton=False, type="GeometryNodeTree"):
    """Wrapper for initializing and registering new nodegroups."""

    def registration_fn(fn):
        nonlocal name
        if name is None:
            name = fn.__name__
        if singleton:
            name = name + " (no gc)"

        def init_fn(*args, **kwargs):
            if singleton and name in bpy.data.node_groups:
                return bpy.data.node_groups[name]
            else:
                ng = bpy.data.node_groups.new(name, type)
                nw = NodeWrangler(ng)
                fn(nw, *args, **kwargs)
                return ng

        return init_fn

    return registration_fn


def to_modifier(name=None, singleton=False, type="GeometryNodeTree"):
    """Wrapper for initializing and registering new nodegroups."""

    def registration_fn(fn):
        nonlocal name
        if name is None:
            name = fn.__name__
        if singleton:
            name = name + " (no gc)"

        def init_fn(*args, **kwargs):
            if singleton and name in bpy.data.node_groups:
                return bpy.data.node_groups[name]
            else:
                if "obj" in kwargs.keys():
                    obj = kwargs.get("obj")
                    mod = obj.modifiers.new(name, "NODES")
                    mod.show_viewport = True
                    # Add node group here
                    if mod.node_group is None:
                        group = geometry_node_group_empty_new()
                        mod.node_group = group
                else:
                    mod = bpy.data.node_groups.new(name, type)
                # if "inputs" in kwargs.keys():
                #     set_geomod_inputs(mod, kwargs.get("inputs"))
                nw = NodeWrangler(mod)
                fn(nw, *args, **kwargs)
                ng = mod.node_group
                ng.name = name
                return mod

        return init_fn

    return registration_fn


def assign_curve(c, points, handles=None):
    for i, p in enumerate(points):
        if i < 2:
            c.points[i].location = p
        else:
            c.points.new(*p)

        if handles is not None:
            c.points[i].handle_type = handles[i]


def facing_mask(nw, dir, thresh=0.5):
    normal = nw.new_node(Nodes.InputNormal)
    up_mask = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: normal, 1: dir},
        attrs={"operation": "DOT_PRODUCT"},
    )
    up_mask = nw.new_node(
        Nodes.Math, input_args=[up_mask, thresh], attrs={"operation": "GREATER_THAN"}
    )

    return up_mask


def noise(nw, scale, **kwargs):
    return nw.new_node(
        Nodes.NoiseTexture,
        input_kwargs={
            "Scale": scale,
            "W": uniform(1e3),
            # Making this as big as 1e6 seems to cause bugs
            "Detail": kwargs.get("detail", uniform(0, 10)),
            "Roughness": kwargs.get("roughness", uniform(0, 1)),
            "Distortion": kwargs.get("distortion", normal(0.7, 0.4)),
        },
        attrs={"noise_dimensions": "4D"},
    )


def resample_node_group(nw: NodeWrangler, scene_seed: int):
    for node in nw.nodes:
        # Randomize 'W' in noise nodes
        if node.bl_idname in {Nodes.NoiseTexture, Nodes.WhiteNoiseTexture}:
            node.noise_dimensions = "4D"
            node.inputs["W"].default_value = np.random.uniform(1000)

        if node.bl_idname == Nodes.ColorRamp:
            for element in node.color_ramp.elements:
                element.color = random_color_mapping(element.color, scene_seed)

        if node.bl_idname == Nodes.RGB:
            node.outputs["Color"].default_value = random_color_mapping(
                node.outputs["Color"].default_value, scene_seed
            )

        # Randomized fixed color input
        for input_socket in node.inputs:
            if input_socket.type == "RGBA":
                # print(f"Mapping", input_socket)
                input_socket.default_value = random_color_mapping(
                    input_socket.default_value, scene_seed
                )

            if input_socket.name == "Seed":
                input_socket.default_value = np.random.randint(1000)


def build_color_ramp(nw, x, positions, colors, mode="HSV"):
    cr = nw.new_node(Nodes.ColorRamp, input_kwargs={"Fac": x})
    cr.color_ramp.color_mode = mode
    elements = cr.color_ramp.elements
    size = len(positions)
    assert len(colors) == size
    if size > 2:
        for _ in range(size - 2):
            elements.new(0)
    for i, (p, c) in enumerate(zip(positions, colors)):
        elements[i].position = p
        elements[i].color = c
    return cr


def save_geometry(
    nw: NodeWrangler,
    geometry,
    path=None,
    name=None,
    idx="unknown",
    first=False,
    bevel=False,
    bevel_edges=None,
    use_bpy=True,
    parent_obj_id=None,
    joint_info=None,
    material=None
):
    output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": geometry},
        attrs={"is_active_output": True},
    )

    # Save a reference to the original scene
    original_scene = bpy.context.scene

    # Create a new mesh and object to store the geometry
    new_mesh = bpy.data.meshes.new(name=f"{name}_mesh")
    new_object = bpy.data.objects.new(name=name, object_data=new_mesh)

    # Link the new object to the current scene
    bpy.context.collection.objects.link(new_object)

    # Assign vertices and faces to the new mesh
    vertices, faces = get_geometry_data(geometry)
    res = None
    if vertices and faces:
        new_mesh.from_pydata(
            vertices,  # Assuming your geometry outputs vertices
            [],  # Edges (if any)
            faces,  # Assuming your geometry outputs faces
        )
        new_mesh.update()  # Update the mesh

        if name == 'whole':
            join_objects_save_whole([new_object], path, idx, name, join=False, use_bpy=use_bpy)
            res = True
        else:
            res = save_obj_parts_add([new_object], path, idx, name, first=first, use_bpy=True, parent_obj_id=parent_obj_id, joint_info=joint_info,material=material)
    #bpy.ops.export_scene.obj(filepath='./file.obj', use_selection=True)
    # 取消选择新对象
    #new_obj.select_set(False)
    return res

def get_seperate_objects(obj):
    # print(bpy.context.collection.objects.keys())
    # butil.select_none()
    # obj.select_set(True)
    # bpy.ops.object.duplicate(linked=True)
    # print(bpy.context.collection.objects.keys())
    # obj.select_set(False)
    # butil.select_none()
    # #new_obj = obj.copy()
    # new_obj = bpy.context.collection.objects.get(obj.name + ".001")
    #bpy.context.collection.objects.link(new_obj)
    #new_obj = deep_clone_obj(obj)
    #print(bpy.context.collection.objects.keys())
    bpy.ops.object.mode_set(mode='EDIT')
    obj.select_set(True)
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')
    butil.select_none()

count = 0
def save_part(obj, type, path, idx, first):
        global count
        name = str(count)
        obj.name = name
        initial = bpy.context.collection.objects.keys()
        temp_obj = obj.copy()
        bpy.context.collection.objects.link(temp_obj)
        get_seperate_objects(obj)
        new_obj = []
        for ob in bpy.context.collection.objects:
            if not ob.name in initial and name in ob.name:
                butil.select_none()
                save_obj_parts_add([ob], path, idx, type, first=first, use_bpy=True)
                new_obj.append(ob)
                first = False
        count += 1
        return new_obj

def save_geometry_new(obj, name, part_idx, idx, path, first, use_bpy=False, separate=False, parent_obj_id=None, joint_info=None, material=None):
    butil.select_none()
    if not obj.name in bpy.context.collection.objects.keys():
        bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    # print(bm.verts.layers.float.keys(), bm.faces.layers.float.keys(), bm.edges.layers.float.keys())
    if name not in bm.verts.layers.float.keys() and name not in bm.verts.layers.int.keys() and name != "whole":
        # 切换回对象模式
        bpy.ops.object.mode_set(mode='OBJECT')
        return
    elif name is not None and name != "whole":
        if name in bm.verts.layers.float.keys():
            attr_layer = bm.verts.layers.float[name]
        else:
            attr_layer = bm.verts.layers.int[name]
        found_verts = [v for v in bm.verts if v[attr_layer] == part_idx]
    else:
        found_verts = [v for v in bm.verts]
    if len(found_verts) == 0:
        # 切换回对象模式
        bpy.ops.object.mode_set(mode='OBJECT')
        return False
    new_mesh = bpy.data.meshes.new(name="SubMesh")
    new_obj = bpy.data.objects.new("SubMeshObject", new_mesh)
    bpy.context.collection.objects.link(new_obj)

    # 创建新的 bmesh 用于子网格，并将符合条件的顶点、边、面复制到新网格中
    new_bm = bmesh.new()
    vert_map = {}

    # 复制符合条件的顶点
    for v in found_verts:
        new_vert = new_bm.verts.new(v.co)
        vert_map[v.index] = new_vert
        
    # 复制符合条件的边和面
    for edge in bm.edges:
        if edge.verts[0].index in vert_map and edge.verts[1].index in vert_map:
            new_bm.edges.new((vert_map[edge.verts[0].index], vert_map[edge.verts[1].index]))

    for face in bm.faces:
        if all(v.index in vert_map for v in face.verts):
            new_face = new_bm.faces.new([vert_map[v.index] for v in face.verts])
            new_face.material_index = face.material_index

    # 写入新的网格数据并释放 bmesh
    new_bm.to_mesh(new_mesh)
    new_bm.free()

    # 切换回对象模式并导出新创建的子网格对象为 OBJ 文件
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = new_obj
    #new_obj.select_set(True)  # 选择新对象

    if separate:
        save_part(new_obj, name, path, idx, first)
        return True
    if name == 'whole':
        join_objects_save_whole([new_obj], path, idx, name, join=False, use_bpy=use_bpy)
        res = True
    else:
        res = save_obj_parts_add([new_obj], path, idx, name, first=first, use_bpy=True, parent_obj_id=parent_obj_id, joint_info=joint_info, material=material)
    #bpy.ops.export_scene.obj(filepath='./file.obj', use_selection=True)
    # 取消选择新对象
    #new_obj.select_set(False)
    return res


def save_geometry_export_obj(
    nw: NodeWrangler, geometry, path=None, name=None, idx="unknown", i=999, bevel=False
):
    # Create a new geometry node setup for the seat
    # print(geometry)
    output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": geometry},
        attrs={"is_active_output": True},
    )
    # Save the seat as a new .blend file. Update with your path
    file_path = save_file_path_obj(path, name, idx, i)

    bpy.ops.wm.save_as_mainfile(filepath=file_path)
