# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import wood
from infinigen.assets.utils.decorate import read_normal
from infinigen.assets.utils.object import join_objects, new_bbox, new_cube
from infinigen.core import tags as t
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.nodes.node_utils import save_geometry_new
from infinigen.assets.utils.object import (
    join_objects_save_whole,
    save_file_path_obj,
    save_obj_parts_add,
)


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
    print(bpy.context.collection.objects.keys())

first = True
count_val = 0
count_ho = 0

class PalletFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(PalletFactory, self).__init__(factory_seed, coarse)
        self.depth = uniform(1.2, 1.4)
        self.width = uniform(1.2, 1.4)
        self.thickness = uniform(0.01, 0.015)
        self.tile_width = uniform(0.06, 0.1)
        self.tile_slackness = uniform(1.5, 2)
        self.height = uniform(0.2, 0.25)
        self.surface = wood

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        bbox = new_bbox(0, self.width, 0, self.depth, 0, self.height)
        write_attr_data(
            bbox,
            f"{PREFIX}{t.Subpart.SupportSurface.value}",
            read_normal(bbox)[:, -1] > 0.5,
            "INT",
            "FACE",
        )
        return bbox
    
    def save_part(self, obj, name, type):
        global first
        initial = bpy.context.collection.objects.keys()
        temp_obj = obj.copy()
        bpy.context.collection.objects.link(temp_obj)
        print(initial)
        get_seperate_objects(obj)
        new_obj = []
        for ob in bpy.context.collection.objects:
            if not ob.name in initial and name in ob.name:
                butil.select_none()
                save_obj_parts_add([ob], self.params.get("path", None), self.params.get("i", None), type, first=first, use_bpy=True)
                new_obj.append(ob)
                first = False
        return new_obj

    def create_asset(self, save=False, **params) -> bpy.types.Object:
        global first
        self.params = params
        vertical = self.make_vertical(save)
        if save:
            objs = self.save_part(vertical, 'vertical', "Bar")
        #vertical.location[-1] = self.thickness
        vertical_ = deep_clone_obj(vertical)
        vertical_.location[-1] = self.height - self.thickness
        if save:
            for ob in objs:
                ob.location[-1] = self.height - self.thickness
                save_obj_parts_add([ob], self.params.get("path", None), self.params.get("i", None), "Bar", first=first, use_bpy=True)
        horizontal = self.make_horizontal(save)
        if save:
            objs = self.save_part(horizontal, 'horizontal', "Bar")
        horizontal_ = deep_clone_obj(horizontal)
        horizontal_.location[-1] = self.height - 2 * self.thickness
        if save:
            for ob in objs:
                ob.location[-1] = self.height - 2 * self.thickness
                save_obj_parts_add([ob], self.params.get("path", None), self.params.get("i", None), "Bar", first=first, use_bpy=True)
        support = self.make_support()
        if save:
            _ = self.save_part(support, 'support', "Support") 
        obj = join_objects([horizontal, horizontal_, vertical, vertical_, support])
        if not save:
            self.create_asset(save=True, **params)
            save_geometry_new(obj, 'whole', 0, self.params.get("i", None), self.params.get("path", None), True, use_bpy=True)
            first = True
        return obj

    def make_vertical(self, save=False):
        obj = new_cube()
        name = f"vertical_{self.params.get('i', None)}"
        obj.name = name
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.tile_width / 2, self.depth / 2, self.thickness / 2
        butil.apply_transform(obj)
        obj.location[-1] = self.thickness
        #butil.apply_transform(obj, True)
        count = (
            int(
                np.floor(
                    (self.width - self.tile_width)
                    / self.tile_width
                    / self.tile_slackness
                )
                / 2
            )
            * 2
        )
        self.count_v = count
        # objs.append(obj.copy())
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=((self.width - self.tile_width) / count, 0, 0),
            count=count + 1,
        )
            
        return obj

    def make_horizontal(self, save=False):
        obj = new_cube()
        name = f"horizontal_{self.params.get('i', None)}"
        obj.name = name
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.width / 2, self.tile_width / 2, self.thickness / 2
        butil.apply_transform(obj)
        count = (
            int(
                np.floor(
                    (self.depth - self.tile_width)
                    / self.tile_width
                    / self.tile_slackness
                )
                / 2
            )
            * 2
        )
        self.count_h = count
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=(0, (self.depth - self.tile_width) / count, 0),
            count=count + 1,
        )

        # if save:
        #     global first
        #     get_seperate_objects(obj)
        #     for i in range(1, count + 2):
        #         print(f"{name}." + str(i).zfill(3))
        #         butil.select_none()
        #         save_obj_parts_add([bpy.data.objects.get(f"{name}." + str(i).zfill(3))], self.params.get("path", None), self.params.get("i", None), "Bar", first=first)
        #         first = False

        return obj

    def make_support(self):
        obj = new_cube()
        obj.name = 'support'
        obj.location = 1, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = (
            self.tile_width / 2,
            self.tile_width / 2,
            self.height / 2 - 2 * self.thickness,
        )
        butil.apply_transform(obj)
        obj.location[-1] = 2 * self.thickness
        #save_obj_parts_add([obj], self.params.get("path", None), self.params.get("i", None), "Support", first=first)
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=((self.width - self.tile_width) / 2, 0, 0),
            count=3,
        )
        butil.modify_mesh(
            obj,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            constant_offset_displace=(0, (self.depth - self.tile_width) / 2, 0),
            count=3,
        )
        return obj

    def finalize_assets(self, assets):
        self.surface.apply(assets)