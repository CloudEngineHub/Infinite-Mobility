# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

# Authors:
# - Lingjie Mei: primary author
# - Karhan Kayan: fix rotation

import bmesh
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.materials.text import Text
from infinigen.assets.utils.decorate import (
    mirror,
    read_area,
    read_co,
    read_normal,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    data2mesh,
    join_objects,
    join_objects_save_whole,
    mesh2obj,
    new_bbox,
    new_cube,
    new_plane,
    save_obj_parts_join_objects,
    save_objects_obj,
    save_obj_parts_add,
    get_joint_name
)
from infinigen.assets.utils.uv import (
    compute_uv_direction,
    face_corner2faces,
    unwrap_faces,
)
from infinigen.core import surface
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import write_attr_data
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform
import random
from infinigen.assets.utils.auxiliary_parts import random_auxiliary


class TVFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TVFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.aspect_ratio = np.random.choice([9 / 16, 3 / 4])
            self.width = uniform(0.6, 2.1)
            self.screen_bevel_width = uniform(0, 0.01)
            self.side_margin = log_uniform(0.005, 0.01)
            self.bottom_margin = uniform(0.005, 0.03)
            self.depth = uniform(0.02, 0.04)
            self.has_depth_extrude = uniform() < 0.4
            if self.has_depth_extrude:
                self.depth_extrude = self.depth * uniform(2, 5)
            else:
                self.depth_extrude = self.depth * 1.5
            self.leg_type = np.random.choice(["two-legged", "single-legged"])  # 'none',
            self.leg_length = uniform(0.1, 0.2)
            self.leg_length_y = uniform(0.1, 0.15)
            self.leg_radius = uniform(0.008, 0.015)
            self.leg_width = uniform(0.5, 0.8)
            self.leg_bevel_width = uniform(0.01, 0.02)

            materials = self.get_material_params()
            self.surface = materials["surface"]
            self.scratch = materials["scratch"]
            self.edge_wear = materials["edge_wear"]
            self.screen_surface = materials["screen_surface"]
            self.support_surface = materials["support"]
            self.button_surface = materials["button_surface"]
            self.use_aux_leg = np.random.choice([True, False], p=[0.7, 0.3])
            self.use_aux_leg = True
            if self.use_aux_leg:
                self.aux_leg = random_auxiliary('tv_supports')

    def get_material_params(self):
        material_assignments = AssetList["TVFactory"]()
        surface = material_assignments["surface"].assign_material()
        button_surface = material_assignments["button_surface"].assign_material()
        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = np.random.uniform() < scratch_prob
        is_edge_wear = np.random.uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        args = (self.factory_seed, False)
        kwargs = {"emission": 0.01 if uniform() < 0.1 else uniform(2, 3)}
        screen_surface = material_assignments["screen_surface"].assign_material()
        if screen_surface == Text:
            screen_surface = screen_surface(*args, **kwargs)
        support = material_assignments["support"].assign_material()
        return {
            "surface": surface,
            "scratch": scratch,
            "edge_wear": edge_wear,
            "screen_surface": screen_surface,
            "support": support,
            "button_surface": button_surface,
        }

    @property
    def height(self):
        return self.aspect_ratio * self.width

    @property
    def total_width(self):
        return self.width + 2 * self.side_margin

    @property
    def total_height(self):
        return self.height + self.side_margin + self.bottom_margin

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        match self.leg_type:
            case "two-legged":
                max_x = (
                    self.leg_length_y / 2 - (1 - self.leg_width) * self.depth_extrude
                )
            case _:
                max_x = self.leg_length_y / 2 - self.depth_extrude / 2
        return new_bbox(
            -self.depth_extrude - self.depth,
            max_x,
            -self.total_width / 2,
            self.total_width / 2,
            -self.leg_length - self.leg_radius / 2,
            self.total_height,
        )

    def create_asset(self, **params) -> bpy.types.Object:
        match self.leg_type:
            case "two-legged":
                legs = self.add_two_legs()
            case _:
                legs = self.add_single_leg()
        for leg_obj in legs:
            write_attribute(leg_obj, 1, "leg", "FACE", "INT")
        connector_radius = 0
        legs = join_objects(legs)
        y_min = read_co(legs)[:, 1].min()
        obj = self.make_base(params.get("path", None), params.get("i", "unknown"))
        self.make_screen(obj)
        if self.use_aux_leg:
            leg = self.aux_leg[0]
            leg.rotation_euler[0] = np.pi / 2
            butil.apply_transform(leg, True)
            co = read_co(legs)
            #print(co[:, 2].max())
            scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
            location = (co[:, 0].max() + co[:, 0].min()) / 2, (co[:, 1].max() + co[:, 1].min()) / 2, (co[:, 2].max() + co[:, 2].min()) / 2
            leg.scale = scale#(scale[0], scale[1], scale[2])
            butil.apply_transform(leg, True)
            leg.location = location
            butil.apply_transform(leg, True)
            legs = leg
            top_1_height = (co[:, 2].max() - co[:, 2].min()) * 0.9 + co[:, 2].min()
            co_ = read_co(legs)
            #print(co_[:, 2].max(), co_[:, 2].min())
            pts = []
            for c in co_:
                #print(c, top_1_height)
                if c[2] >= top_1_height:
                    pts.append(c)
            #print(pts)
            pts = np.array(pts)
            y_min = pts[:, 1].min()
            y_max = pts[:, 1].max()
            x_max = pts[:, 0].max()
            x_min = pts[:, 0].min()
            connector_radius = (x_max - x_min) / 10 * uniform(0.8, 1)
            connector = butil.spawn_cylinder(radius=connector_radius, depth=connector_radius * uniform(1, 4))
            connector.rotation_euler[1] = np.pi / 2
            connector.location = (co[:, 0].max() + co[:, 0].min()) / 2, y_min - connector_radius, top_1_height
            butil.apply_transform(connector, True)
            y_min = read_co(connector)[:, 1].min()
            print(self.aux_leg)
            if self.aux_leg[1]['prismatic'] == 'True':
                res = save_obj_parts_add(connector, params.get("path", None), params.get("i", "unknown"), "connector", use_bpy=True, first=True, parent_obj_id="world", joint_info=
                            {
                                "name": get_joint_name("revolute_prismatic"),
                                    "type": "revolute_prismatic",
                                    "axis" : (1, 0, 0),
                                    "axis_1": (0, 0, 1),
                                    #"origin_shift": (0, 0, (self.h_max + self.h_min) / 2 - self.leg_length),
                                    "limit": {
                                        "lower": -np.pi / 8,
                                        "upper": -np.pi / 48,
                                        "lower_1": (-self.h_min / 2) if self.h_min else -0.05,
                                        "upper_1": (self.h_min / 2) if self.h_min else 0.05
                                    }
                            })
            else:
                res = save_obj_parts_add(connector, params.get("path", None), params.get("i", "unknown"), "connector", use_bpy=True, first=True, parent_obj_id="world", joint_info=
                            {
                                "name": get_joint_name("revolute"),
                                    "type": "revolute",
                                    "axis" : (1, 0, 0),
                                    #"origin_shift": (0, 0, (self.h_max + self.h_min) / 2 - self.leg_length),
                                    "limit": {
                                        "lower": -np.pi / 8,
                                        "upper": -np.pi / 48,
                                        #"lower_1": (-self.h_min / 2) if self.h_min else -0.05,
                                        #"upper_1": (self.h_min / 2) if self.h_min else 0.05
                                    }
                            })
            #res = save_obj_parts_add(connector, params.get("path", None), params.get("i", "unknown"), "connector", use_bpy=True, first=True, parent_obj_id=None)
        save_obj_parts_add(legs, params.get("path", None), params.get("i", "unknown"), "screen_leg", use_bpy=True, first=False if self.use_aux_leg else True, material=[self.support_surface])
        self.surface.apply(obj, selection="!screen", rough=True, metal_color="bw")
        self.support_surface.apply(
            obj, selection="leg", rough=True, metal_color="bw"
        )
        butil.apply_transform(obj, True)
        y = read_co(obj)[:, 1].max()
        obj.location = 0, y_min - y, 0
        butil.apply_transform(obj, True)
        co = read_co(obj).copy()
        if self.use_aux_leg:
            res = save_obj_parts_add(obj, params.get("path", None), params.get("i", "unknown"), "screen", use_bpy=True, first=False, parent_obj_id=res[0], joint_info={
                "name": get_joint_name("fixed"),
                "type": "fixed",
            })
        else:
            res = save_obj_parts_add(obj, params.get("path", None), params.get("i", "unknown"), "screen", use_bpy=True, first=False, parent_obj_id="world", joint_info=
                            {
                                "name": get_joint_name("revolute_prismatic"),
                                    "type": "revolute_prismatic",
                                    "axis" : (1, 0, 0),
                                    "axis_1": (0, 0, 1),
                                    "origin_shift": (0, 0, (self.h_max + self.h_min) / 2 - self.leg_length),
                                    "limit": {
                                        "lower": -np.pi / 8,
                                        "upper": -np.pi / 48,
                                        "lower_1": (-self.h_min / 2) if self.h_min else -0.05,
                                        "upper_1": (self.h_min / 2) if self.h_min else 0.05
                                    }
                            })
        z_min = co[:, 2].min()
        button_type = np.random.choice(['circle', 'square'])
        button_type = "square"
        button_scale = self.bottom_margin * 0.5 * uniform(0.9, 1.0)
        location = (co[:, 0].max() * 0.9, -button_scale / 10 + y_min - y, self.bottom_margin / 2)

        button_num = np.random.randint(0, 5)
        gap = button_scale * random.uniform(0.5, 1.2)
        use_aux_button = np.random.choice([True, False], p=[0.9, 0.1])
        all_same = True
        if use_aux_button:
            all_same = np.random.choice([True, False], p=[0.5, 0.5])
        aux_button = None
        for i in range(button_num):
            if button_type == "square":
                button = butil.spawn_cube()
            else:
                button = butil.spawn_cylinder(radius=button_scale / 2, depth=button_scale / 5)
                button.rotation_euler[0] = np.pi / 2
                butil.apply_transform(button, True)
            button.scale = button_scale, button_scale / 5, button_scale
            #butil.apply_transform(button, True)
            button.location = location
            butil.apply_transform(button, True)
            location = (location[0] - gap - button_scale, location[1], location[2])
            if use_aux_button:
                if not all_same:
                    aux_button = butil.deep_clone_obj(random_auxiliary('buttons')[0], keep_materials=False, keep_modifiers=False)
                elif aux_button is None:
                    aux_button = butil.deep_clone_obj(random_auxiliary('buttons')[0], keep_materials=False, keep_modifiers=False)
                #aux_button = aux_button[0]
                aux_button.rotation_euler = np.pi / 2, 0, np.pi / 2
                butil.apply_transform(aux_button)
                co = read_co(button)
                co_ = read_co(aux_button)
                scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
                scale_t = co_[:, 0].max() - co_[:, 0].min(), co_[:, 1].max() - co_[:, 1].min(), co_[:, 2].max() - co_[:, 2].min()
                scale = scale[0] / scale_t[0], scale[1] / scale_t[1], scale[2] / scale_t[2] 
                aux_button.scale = scale
                butil.apply_transform(aux_button)
                aux_button.location = co[:, 0].max() - (co[:, 0].max() - co[:, 0].min()) / 2, co[:, 1].max() - (co[:, 1].max() - co[:, 1].min()) / 2, co[:, 2].max() - (co[:, 2].max() - co[:, 2].min()) / 2
                butil.apply_transform(aux_button)
                button = aux_button
            self.button_surface.apply(button, rough=True)
            save_obj_parts_add(butil.deep_clone_obj(button, keep_materials=True, keep_modifiers=True), params.get("path", None), params.get("i", "unknown"), "button", use_bpy=True, first=False, parent_obj_id=res[0], joint_info=
                               {
                                   "name": get_joint_name("prismatic"),
                                   "type": "prismatic",
                                   "axis": (0, 1, 0),
                                   "limit": {
                                        "lower": -button_scale / 8,
                                        "upper": 0
                                   }
                               })

        parts = [obj]
        name = ["screen"]
        co = read_co(legs)
        parts.append(legs)
        self.surface.apply(legs, selection="!screen", rough=True, metal_color="bw")
        self.support_surface.apply(
            legs , rough=True, #metal_color="bw"
        )
        #name.extend(["legs"] * len(legs))
        # save_objects_obj(
        #     parts,
        #     params.get("path", None),
        #     params.get("i", "unknown"),
        #     name=name,
        #     obj_name="TV",
        #     first=True,
        # )
        obj = join_objects(parts)
        join_objects_save_whole(
            obj,
            params.get("path", None),
            params.get("i", "unknown"),
            use_bpy=True
        )
        obj.rotation_euler[2] = np.pi / 2
        butil.apply_transform(obj)
        return obj

    def make_screen(self, obj):
        cutter = new_cube()
        cutter.location = 0, -1, 1
        butil.apply_transform(cutter, True)
        cutter.scale = self.width / 2, 1, self.height / 2
        cutter.location = 0, 1e-3, self.bottom_margin
        butil.apply_transform(cutter, True)
        butil.modify_mesh(obj, "BOOLEAN", object=cutter, operation="DIFFERENCE")
        butil.delete(cutter)
        areas = read_area(obj)
        screen = np.zeros(len(areas), int)
        y = read_normal(obj)[:, 1] < 0
        screen[np.argmax(areas + 1e5 * y)] = 1
        fc2f = face_corner2faces(obj)
        unwrap_faces(obj, screen)
        bbox = compute_uv_direction(obj, "x", "z", screen[fc2f])
        write_attr_data(obj, "screen", screen, domain="FACE", type="INT")
        self.screen_surface.apply(obj, "screen", bbox)

    def make_base(self, path=None, i="unknown"):
        obj = new_cube()
        obj.location = 0, 1, 1
        butil.apply_transform(obj, True)
        obj.scale = self.total_width / 2, self.depth / 2, self.total_height / 2
        butil.apply_transform(obj)
        butil.modify_mesh(obj, "BEVEL", width=self.screen_bevel_width, segments=8)
        if not self.has_depth_extrude:
            self.h_max = self.total_height
            self.h_min = 0
            return obj
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [f for f in bm.faces if f.normal[1] > 0.5]
            bmesh.ops.delete(bm, geom=geom, context="FACES_KEEP_BOUNDARY")
            bmesh.update_edit_mesh(obj.data)
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.region_to_loop()
        height_min, height_max = (
            self.total_height * uniform(0.1, 0.3),
            self.total_height * uniform(0.5, 0.7),
        )
        self.h_min = height_min
        self.h_max = height_max
        width = self.total_width * uniform(0.3, 0.6)
        extra = new_plane()
        extra.scale = width / 2, (height_max - height_min) / 2, 1
        extra.rotation_euler[0] = -np.pi / 2
        extra.location = 0, self.depth_extrude + self.depth, self.total_height / 2
        obj = join_objects([obj, extra])
        # obj = save_obj_parts_join_objects(
        #     [obj, extra], path, i, name=["base_part", "base_part"], obj_name="TV"
        # )
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=32, profile_shape_factor=-uniform(0.0, 0.4)
            )
        x, y, z = read_co(obj).T
        self.add_h = (
            (height_max + height_min - self.total_height)
            / 2
            * np.clip(y - self.depth, 0, None)
            / self.depth_extrude
        )
        z += (
            (height_max + height_min - self.total_height)
            / 2
            * np.clip(y - self.depth, 0, None)
            / self.depth_extrude
        )
        write_co(obj, np.stack([x, y, z], -1))
        return obj

    def add_two_legs(self):
        vertices = (
            (
                -self.total_width / 2 * self.leg_width * uniform(0, 0.6),
                0,
                self.total_height * uniform(0.3, 0.5),
            ),
            (0, 0, -self.leg_length),
            (0, self.leg_length_y / 2, -self.leg_length),
            (0, -self.leg_length_y / 2, -self.leg_length),
        )
        edges = (0, 1), (1, 2), (1, 3)
        leg = mesh2obj(data2mesh(vertices, edges))
        surface.add_geomod(
            leg, geo_radius, apply=True, input_args=[self.leg_radius, 16]
        )
        x, y, z = read_co(leg).T
        write_co(
            leg,
            np.stack(
                [
                    x,
                    y,
                    np.maximum(
                        z, -self.leg_length - self.leg_radius * uniform(0.0, 0.6)
                    ),
                ],
                -1,
            ),
        )
        leg_ = deep_clone_obj(leg)
        butil.select_none()
        leg.location = (
            self.total_width / 2 * self.leg_width,
            (1 - self.leg_width) * self.depth_extrude,
            0,
        )
        butil.apply_transform(leg, True)
        mirror(leg_)
        leg_.location = (
            -self.total_width / 2 * self.leg_width,
            (1 - self.leg_width) * self.depth_extrude,
            0,
        )
        butil.apply_transform(leg_, True)
        return [leg, leg_]

    def add_single_leg(self):
        leg = new_cube()
        leg.location = 0, 1, 1
        butil.apply_transform(leg, True)
        leg.location = 0, self.depth_extrude / 2, -self.leg_length
        leg.scale = [
            self.total_width * uniform(0.05, 0.1),
            self.leg_radius,
            (self.leg_length + self.total_height * uniform(0.3, 0.5)) / 2,
        ]
        butil.apply_transform(leg, True)
        butil.modify_mesh(leg, "BEVEL", width=self.leg_bevel_width, segments=8)
        base = new_cube()
        base.location = 0, self.depth_extrude / 2, -self.leg_length
        base.scale = [
            self.total_width * uniform(0.15, 0.3),
            self.leg_length_y / 2,
            self.leg_radius,
        ]
        butil.apply_transform(base, True)
        butil.modify_mesh(base, "BEVEL", width=self.leg_bevel_width, segments=8)
        return [leg, base]

    def finalize_assets(self, assets):
        self.surface.apply(assets, selection="!screen", rough=True, metal_color="bw")
        self.support_surface.apply(
            assets, selection="leg", rough=True, metal_color="bw"
        )


class MonitorFactory(TVFactory):
    def __init__(self, factory_seed, coarse=False):
        super(MonitorFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.width = log_uniform(0.4, 0.8)
            self.leg_type = "single-legged"
