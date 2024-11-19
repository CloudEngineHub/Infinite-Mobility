# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import bmesh
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials import metal
from infinigen.assets.objects.elements.warehouses.pallet import PalletFactory
from infinigen.assets.utils.decorate import (
    read_co,
    remove_faces,
    solidify,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    join_objects,
    new_base_cylinder,
    new_bbox,
    new_cube,
    new_line,
    new_plane,
)
from infinigen.core import surface
from infinigen.core import tags as t
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import write_attr_data
from infinigen.core.tagging import PREFIX
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed

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
    if obj.name not in bpy.context.collection.objects.keys():
        bpy.context.collection.objects.link(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    obj.select_set(True)
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')
    butil.select_none()

first = True


class RackFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(RackFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.depth = uniform(1, 1.2)
            self.width = uniform(4.0, 5.0)
            self.height = uniform(1.6, 1.8)
            self.steps = np.random.randint(3, 6)
            self.thickness = uniform(0.06, 0.08)
            self.hole_radius = self.thickness / 2 * uniform(0.5, 0.6)
            self.support_angle = uniform(np.pi / 6, np.pi / 4)
            self.is_support_round = uniform() < 0.5
            self.frame_height = self.thickness * uniform(3, 4)
            self.frame_count = np.random.randint(20, 30)

            self.stand_surface = self.support_surface = self.frame_surface = metal
            self.pallet_factory = PalletFactory(self.factory_seed)
            self.margin_range = 0.3, 0.5

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        bbox = new_bbox(
            -self.depth - self.thickness / 2,
            self.thickness / 2,
            -self.thickness / 2,
            self.width + self.thickness / 2,
            0,
            self.height * self.steps,
        )
        objs = [bbox]
        for i in range(self.steps):
            obj = new_plane()
            obj.scale = self.depth / 2, self.width / 2 - self.thickness, 1
            obj.location = -self.depth / 2, self.width / 2, self.height * i
            butil.apply_transform(obj, True)
            write_attr_data(
                obj,
                f"{PREFIX}{t.Subpart.SupportSurface.value}",
                np.ones(1).astype(bool),
                "INT",
                "FACE",
            )
            objs.append(obj)
        obj = join_objects(objs)
        return obj

    def create_asset(self, save=False, **params) -> bpy.types.Object:
        global first
        self.params = params
        stands = self.make_stands(save)
        supports = self.make_supports(save)
        frames = self.make_frames(save)
        obj = join_objects(stands + supports + frames)
        co = read_co(obj)
        co[:, -1] = np.clip(co[:, -1], 0, self.height * self.steps)
        write_co(obj, co)
        # obj = join_objects([obj] + pallets)
        obj.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(obj)
        if not save:
            self.create_asset(save=True, **params)
            save_geometry_new(obj, 'whole', 0, self.params.get("i", None), self.params.get("path", None), True)
            first = True
        return obj

    def make_stands(self, save=False):
        obj = new_cube()
        obj.scale = [self.thickness / 2] * 3
        butil.apply_transform(obj, True)
        cylinder = new_base_cylinder()
        cylinder.scale = self.hole_radius, self.hole_radius, self.thickness * 2
        cylinder.rotation_euler[1] = np.pi / 2
        butil.apply_transform(cylinder)
        butil.modify_mesh(obj, "BOOLEAN", object=cylinder, operation="DIFFERENCE")
        cylinder.rotation_euler[-1] = np.pi / 2
        butil.apply_transform(cylinder)
        butil.modify_mesh(obj, "BOOLEAN", object=cylinder, operation="DIFFERENCE")
        butil.delete(cylinder)
        remove_faces(
            obj,
            lambda x, y, z: (np.abs(x) < self.thickness * 0.49)
            & (np.abs(y) < self.thickness * 0.49)
            & (np.abs(z) < self.thickness * 0.49),
        )
        remove_faces(obj, lambda x, y, z: np.abs(x) + np.abs(y) < self.thickness * 0.1)
        obj.location[-1] = self.thickness / 2
        butil.apply_transform(obj, True)
        butil.modify_mesh(
            obj,
            "ARRAY",
            count=int(np.ceil(self.height / self.thickness * self.steps)),
            relative_offset_displace=(0, 0, 1),
            use_merge_vertices=True,
        )
        write_attribute(obj, 1, "stand", "FACE")
        stands = [obj]
        for locs in [(0, 1), (1, 1), (1, 0)]:
            o = deep_clone_obj(obj)
            o.location = locs[0] * self.width, locs[1] * self.depth, 0
            butil.apply_transform(o, True)
            stands.append(o)
        if save:
            global first
            for stand in stands:
                save_obj_parts_add([stand], self.params.get("path", None), self.params.get("i", None), "stand", first=first, use_bpy=True) 
                first = False
        return stands


    def make_supports(self,save=False):
        n = int(
            np.floor(self.height * self.steps / self.depth / np.tan(self.support_angle))
        )
        obj = new_line(n, self.height * self.steps)
        print(self.steps, n, self.height)
        obj.rotation_euler[1] = -np.pi / 2
        butil.apply_transform(obj, True)
        global first
        #save_obj_parts_add([obj], self.params.get("path", None), self.params.get("i", None), "support", first=first) 
        co = read_co(obj)
        co[1::2, 1] = self.depth
        write_co(obj, co)
        #self.dismantle_support(obj)

        if self.is_support_round:
            surface.add_geomod(
                obj, geo_radius, apply=True, input_args=[self.thickness / 2, 16]
            )
        else:
            solidify(obj, 1, self.thickness)
        
        if save:
            for i in range(n):
                obj_ = new_line(1, self.height)
                obj_.rotation_euler[1] = -np.pi / 2
                butil.apply_transform(obj_, True)
                co = read_co(obj_)
                if i % 2:
                    co[1, 2] = -self.depth
                    co[0, 2] = 0
                else:
                    co[0, 2] = -self.depth
                    co[1, 2] = 0
                co[0, 1] = i * self.depth * np.tan(self.support_angle)
                co[1, 1] = (i + 1) * self.depth * np.tan(self.support_angle)
                write_co(obj_, co)
                if self.is_support_round:
                    surface.add_geomod(
                        obj_, geo_radius, apply=True, input_args=[self.thickness / 2, 16]
                    )
                else:
                    solidify(obj_, 1, self.thickness)
                save_obj_parts_add([obj_], self.params.get("path", None), self.params.get("i", None), "support", first=first)
                obj_ = new_line(1, self.height)
                obj_.rotation_euler[1] = -np.pi / 2
                butil.apply_transform(obj_, True)
                co = read_co(obj_)
                if i % 2:
                    co[1, 2] = -self.depth
                    co[0, 2] = 0
                else:
                    co[0, 2] = -self.depth
                    co[1, 2] = 0
                co[0, 1] = i * self.depth * np.tan(self.support_angle)
                co[1, 1] = (i + 1) * self.depth * np.tan(self.support_angle)
                co[0, 0] = co[1, 0] = self.width
                write_co(obj_, co)
                if self.is_support_round:
                    surface.add_geomod(
                    obj_, geo_radius, apply=True, input_args=[self.thickness / 2, 16]
                )
                else:
                    solidify(obj_, 1, self.thickness)
                save_obj_parts_add([obj_], self.params.get("path", None), self.params.get("i", None), "support", first=first)


        write_attribute(obj, 1, "support", "FACE")
        #self.dismantle_support(obj)
        #save_obj_parts_add([obj], self.params.get("path", None), self.params.get("i", None), "support", first=first) 
        o = deep_clone_obj(obj)
        o.location[0] = self.width
        butil.apply_transform(o, True)
        #save_obj_parts_add([o], self.params.get("path", None), self.params.get("i", None), "support", first=first) 
        return [obj, o]
    
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

    def make_frames(self, save= False):
        x_bar = new_cube()
        x_bar.scale = self.width / 2, self.thickness / 2, self.frame_height / 2
        x_bar.location = self.width / 2, 0, self.height - self.frame_height / 2
        butil.apply_transform(x_bar, True)
        x_bar_ = deep_clone_obj(x_bar)
        x_bar_.location[1] = self.depth
        butil.apply_transform(x_bar_, True)
        y_bar = new_cube()
        y_bar.scale = self.thickness / 2, self.depth / 2, self.thickness / 2
        margin = self.width / self.frame_count
        y_bar.location = margin, self.depth / 2, self.height - self.thickness / 2
        y_bar.name = 'bar'
        x_bar.name = 'bar__'
        x_bar_.name = 'bar_'
        butil.apply_transform(y_bar, True)
        butil.modify_mesh(
            y_bar,
            "ARRAY",
            use_relative_offset=False,
            use_constant_offset=True,
            count=self.frame_count - 1,
            constant_offset_displace=(margin, 0, 0),
        )
        frames = [x_bar, x_bar_, y_bar]
        if save:
            objs = self.save_part(y_bar, "bar", "frame")
            x_bar.location[1] = 0
        #bpy.context.collection.objects.link(x_bar)
            butil.apply_transform(x_bar, True)
            self.save_part(x_bar, "bar", "frame")
        # x_bar_.location[1] = self.depth
        # bpy.context.collection.objects.link(x_bar_)
        # butil.apply_transform(x_bar_, True)
            self.save_part(x_bar_, "bar", "frame")
        for i in range(1, self.steps - 1):
            if save:
                for ob in objs:
                    ob_ = deep_clone_obj(ob)
                    ob_.location[-1] += self.height * i
                    butil.apply_transform(ob_, True)
                    self.save_part(ob_, "bar", "frame")
            for obj in [x_bar, x_bar_, y_bar]:
                o = deep_clone_obj(obj)
                o.location[-1] += self.height * i
                butil.apply_transform(o, True)
                frames.append(o)
                if (obj.name == 'bar__' or obj.name == 'bar_') and save:
                    self.save_part(o, "bar", "frame")
        # global first
        # for frame in frames:
        #     save_obj_parts_add([frame], self.params.get("path", None), self.params.get("i", None), "frame", first=first) 
            #scene.collection.objects.link(frame)

        #.collection.objects.link(part)

        # for o in frames:
        #     write_attribute(o, 1, "frame", "FACE")
        return frames

    def finalize_assets(self, assets):
        self.stand_surface.apply(assets, "stand", metal_color="bw")
        self.support_surface.apply(assets, "support", metal_color="bw")
        self.frame_surface.apply(assets, "frame", metal_color="bw")
