# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen


import bpy
import numpy as np
from numpy.random import normal as N
from numpy.random import randint as RI
from numpy.random import uniform as U

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.assets.utils.misc import generate_text
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.bevelling import (
    add_bevel,
    complete_bevel,
    complete_no_bevel,
    get_bevel_edges,
)
from infinigen.core.util.blender import delete
from infinigen.core.util.math import FixedSeed

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
    get_joint_name,
    saved_obj
)
from infinigen.assets.utils.auxiliary_parts import random_auxiliary


class OvenFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=[1.0, 1.0, 1.0]):
        super(OvenFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions
        with FixedSeed(factory_seed):
            self.params, self.geometry_node_params = self.sample_parameters(dimensions)
            self.ps, self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )
        self.geometry_node_params.update(self.material_params)
        self.use_aux_botton = np.random.choice([True, False], p=[0.8, 0.2])
        if self.use_aux_botton:
            self.aux_botton = random_auxiliary("revolute_botton")


    def get_material_params(self):
        material_assignments = AssetList["OvenFactory"]()
        params = {
            "Surface": material_assignments["surface"].assign_material(),
            "Back": material_assignments["back"].assign_material(),
            "WhiteMetal": material_assignments["white_metal"].assign_material(),
            "SuperBlackGlass": material_assignments["black_glass"].assign_material(),
            "Glass": material_assignments["glass"].assign_material(),
        }
        wrapped_params = {
            k: surface.shaderfunc_to_material(v) for k, v in params.items()
        }

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = np.random.uniform() < scratch_prob
        is_edge_wear = np.random.uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return params, wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        # depth, width, height = dimensions
        depth = 1 + N(0, 0.1)
        width = 1 + N(0, 0.1)
        height = 1 + N(0, 0.1)
        door_thickness = U(0.05, 0.1) * depth
        door_rotation = 0  # Set to 0 for now

        rack_radius = U(0.01, 0.02) * depth
        rack_h_amount = RI(2, 4)
        rack_d_amount = RI(4, 6)

        panel_height = U(0.2, 0.4) * height
        panel_thickness = U(0.15, 0.25) * depth
        botton_amount = RI(1, 4) * 2
        botton_radius = U(0.05, 0.1) * width / botton_amount * 2
        botton_thickness = U(0.02, 0.04) * depth / botton_amount * 2
        heat_radius_ratio = U(0.1, 0.2)
        brand_name = generate_text()

        use_gas = RI(2)
        n_grids = RI(2, 5)
        grids = [RI(1, 4) for i in range(n_grids)]
        branches = 2 * RI(2, 9)
        grate_thickness = U(0.01, 0.03)
        center_ratio = U(0.05, 0.15)
        middle_ratio = U(0.5, 0.7)

        params = {
            "UseGas": use_gas,
            "Grids": grids,
            "Branches": branches,
            "GrateThickness": grate_thickness,
            "CenterRatio": center_ratio,
            "MiddleRatio": middle_ratio,
            "Depth": depth,
            "Width": width,
            "Height": height,
            "DoorThickness": door_thickness,
            "DoorRotation": door_rotation,
            "RackRadius": rack_radius,
            "RackHAmount": rack_h_amount,
            "RackDAmount": rack_d_amount,
            "PanelHeight": panel_height,
            "PanelThickness": panel_thickness,
            "BottonAmount": botton_amount,
            "BottonRadius": botton_radius,
            "BottonThickness": botton_thickness,
            "HeaterRadiusRatio": heat_radius_ratio,
            "BrandName": brand_name,
        }
        geometry_node_params = {
            k: params[k]
            for k in params.keys()
            if k
            not in [
                "UseGas",
                "Grids",
                "Branches",
                "GrateThickness",
                "CenterRatio",
                "MiddleRatio",
            ]
        }
        return params, geometry_node_params

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        # x, y, z = self.params["Depth"], self.params["Width"], self.params["Height"]
        # box = new_bbox(-x/2 - 0.05, x/2 + self.params["DoorThickness"] + 0.1, -y/2, y/2, 0, z + 0.1)
        # tagging.tag_object(box, f'{PREFIX}{t.Subpart.SupportSurface.value}', read_normal(box)[:, -1] > .5)
        # box_top = new_bbox(-x/2 - 0.05, -x/2 - 0.05 + self.params["PanelThickness"], -y/2, y/2, z + 0.1, z+ 0.1 + 0.5)
        # box_top.rotation_euler[1] = -0.1
        # box = butil.join_objects([box, box_top])
        obj = butil.spawn_cube()
        return butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_oven_geometry(
                use_gas=self.params["UseGas"], is_placeholder=True
            ),
            ng_inputs=self.geometry_node_params,
            apply=True,
        )

    def create_asset(self, **params):
        self.params["UseGas"] = False
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_oven_geometry(
                preprocess=True, use_gas=self.params["UseGas"]
            ),
            ng_inputs=self.geometry_node_params,
            apply=True,
        )
        bevel_edges = get_bevel_edges(obj)
        delete(obj)
        obj = butil.spawn_cube()
        butil.modify_mesh(
            obj,
            "NODES",
            node_group=nodegroup_oven_geometry(use_gas=self.params["UseGas"]),
            ng_inputs=self.geometry_node_params,
            apply=True,
        )
        #obj = add_bevel(obj, bevel_edges, offset=0.01)
        if not self.params["UseGas"]:
            self.params.update(params)
            self.params.update(self.ps)
            return obj
        width, depth = (
            self.params["Width"],
            self.params["Depth"] + 2 * self.params["DoorThickness"],
        )
        grate_width, grate_depth = width * 0.8, depth * 0.6
        grate_thickness = self.params["GrateThickness"]
        grates = gas_grates(
            width,
            depth,
            grate_width,
            grate_depth,
            self.params["Height"] + self.params["DoorThickness"] - grate_thickness,
            grate_thickness,
            self.params["Grids"],
            self.params["Branches"],
            self.params["CenterRatio"],
            self.params["MiddleRatio"],
        )
        grates.data.materials.append(self.geometry_node_params["WhiteMetal"])
        obj.data.materials.append(self.geometry_node_params["Back"])
        with butil.SelectObjects(obj):
            obj.active_material_index = len(obj.material_slots) - 1
            for i in range(len(obj.material_slots)):
                bpy.ops.object.material_slot_move(direction="UP")
        hollow = butil.spawn_cube(
            size=1,
            location=(
                depth / 2,
                width / 2,
                self.params["Height"] + self.params["DoorThickness"],
            ),
            scale=(
                grate_depth + grate_thickness,
                grate_width + grate_thickness,
                grate_thickness * 2,
            ),
        )
        with butil.SelectObjects(hollow):
            bpy.ops.object.modifier_add(type="BEVEL")
            bpy.context.object.modifiers["Bevel"].segments = 8
            bpy.context.object.modifiers["Bevel"].width = grate_thickness
            bpy.ops.object.modifier_apply(modifier="Bevel")
        # with butil.SelectObjects(obj):
        #     bpy.ops.object.modifier_add(type="BOOLEAN")
        #     bpy.context.object.modifiers["Boolean"].object = hollow
        #     bpy.context.object.modifiers["Boolean"].use_hole_tolerant = True
        #     bpy.ops.object.modifier_apply(modifier="Boolean")
        #grates.location = hollow.location
        butil.apply_transform(grates, True)
        butil.delete(hollow)
        joint_info = {
                "name": get_joint_name("limited_floating"),
                "type": "limited_floating",
                "axis": (1, 0, 0),
                "limit": {
                    "lower_1": 0,
                    "upper_1": self.params['Depth'] * 0.75,
                    "lower": 0,
                    "upper": 0,
                    "lower_2": 0,
                    "upper_2": (self.params['Height'] - self.params['PanelHeight']) / (self.params['RackHAmount']) * (+ 0.5),
                },
            }
        save_obj_parts_add([grates], params.get("path"), params.get("i"), "part", first=True, use_bpy=True, material=[self.scratch, self.edge_wear], joint_info=joint_info, parent_obj_id="world")
        butil.join_objects([obj, grates], check_attributes=True)
        self.params.update(params)
        self.params.update(self.ps)

        return obj

    def finalize_assets(self, assets):
        global saved_obj
        print(self.params)
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
        first = False
        material = None
        print(self.params['WhiteMetal'])
        parent_id = "world"
        for i in range(5, 0, -1):
            if i == 1:
                material = None#[[self.params['Glass'],'In'], [self.params['Surface'], 'Out'], [self.params['WhiteMetal'], 'handle'], [self.params['WhiteMetal'], 'text'], self.scratch, self.edge_wear]
                joint_info = {
                    "name": get_joint_name("revolute"),
                    "type": "revolute",
                    "axis": (0, 1, 0),
                    "limit": {
                        "lower": 0,
                        "upper": np.pi / 2,
                    },
                    "origin_shift": (0, 0, -self.params['Height'] / 2),
                }
                door = butil.spawn_cube()
                butil.modify_mesh(
                    door,
                    "NODES",
                    node_group=nodegroup_oven_geometry(
                        preprocess=False, use_gas=self.params["UseGas"], return_part_name="door"
                    ),
                    ng_inputs=self.geometry_node_params,
                    apply=True,
                )
                a = save_obj_parts_add([door], self.params.get("path"), self.params.get("i"), "door", first=first, use_bpy=True, material=[self.scratch, self.edge_wear], joint_info=joint_info, parent_obj_id=parent_id)
            elif i == 2:
                continue
            elif i == 3:
                material = None#[[self.params['Back'], 'back'], self.scratch, self.edge_wear]
                joint_info = {
                    "name": get_joint_name("fixed"),
                    "type": "fixed",
                }
                heater = butil.spawn_cube()
                butil.modify_mesh(
                    heater,
                    "NODES",
                    node_group=nodegroup_oven_geometry(
                        preprocess= False, use_gas=self.params["UseGas"], return_part_name="heater"
                    ),
                    ng_inputs=self.geometry_node_params,
                    apply=True,
                )
                a = save_obj_parts_add([heater], self.params.get("path"), self.params.get("i"), "heater", first=first, use_bpy=True, material=[self.scratch, self.edge_wear], joint_info=joint_info, parent_obj_id=parent_id)
            elif i == 4:
                material = None#[[self.params['Back'], 'In'], [self.params['Surface'], 'Out'], [self.params['WhiteMetal'], 'handle'], self.scratch, self.edge_wear]
                joint_info = {
                    "name": get_joint_name("revolute"),
                    "type": "revolute",
                    "axis": (0, 1, 0),
                    "limit": {
                        "lower": 0,#-np.pi / 2,
                        "upper": 0,
                    },
                    "origin_shift": (-self.params['PanelThickness'] / 2, 0, -self.params['PanelHeight'] / 2),
                }
                panel = butil.spawn_cube()
                butil.modify_mesh(
                    panel,
                    "NODES",
                    node_group=nodegroup_oven_geometry(
                        preprocess=False, use_gas=self.params["UseGas"], return_part_name="panel"
                    ),
                    ng_inputs=self.geometry_node_params,
                    apply=True,
                )
                panel.location[2] += 0.005
                butil.apply_transform(panel, True)
                a = save_obj_parts_add([panel], self.params.get("path"), self.params.get("i"), "panel", first=first, use_bpy=True, material=[self.scratch, self.edge_wear], joint_info=joint_info, parent_obj_id=parent_id)
            elif i == 5:
                material = None#[self.params['Surface'], self.scratch, self.edge_wear]
                joint_info = {
                    "name": get_joint_name("fixed"),
                    "type": "fixed",
                }
                body = butil.spawn_cube()
                butil.modify_mesh(
                    body,
                    "NODES",
                    node_group=nodegroup_oven_geometry(
                        preprocess=False, use_gas=self.params["UseGas"], return_part_name="body"
                    ),
                    ng_inputs=self.geometry_node_params,
                    apply=True,
                )
                a = save_obj_parts_add([body], self.params.get("path"), self.params.get("i"), "body", first=first, use_bpy=True, material=[self.scratch, self.edge_wear], joint_info=joint_info, parent_obj_id=parent_id)
            
            #a = node_utils.save_geometry_new(assets, "part", i, self.params.get("i"), self.params.get("path"), first, True, False, material = material, parent_obj_id=parent_id, joint_info=joint_info)
            if a:
                first = False
                if i == 4:
                    p_id = a[0] 
        parent_id = p_id

        min_b = 100
        max_b = -1
        for i in range(1, self.params['BottonAmount'] + 1):
            joint_info = {
                "name": get_joint_name("revolute"),
                "type": "revolute",
                "axis": (1, 0, 0),
                "limit": {
                    "lower": -np.pi / 4,
                    "upper": np.pi / 4,
                },
            }
            def get_co(obj, _):
                nonlocal min_b, max_b
                co = read_co(obj)
                min_b = min(min_b, np.min(co[:, 1]))
                max_b = max(max_b, np.max(co[:, 1]))
                if not self.use_aux_botton:
                    return
                new_botton = butil.deep_clone_obj(self.aux_botton[0])
                new_botton.rotation_euler = (np.pi / 2, 0, np.pi / 2)
                butil.apply_transform(new_botton, True)
                scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
                location = co[:, 0].min() + scale[0] / 2, co[:, 1].min() + scale[1] / 2, co[:, 2].min() + scale[2] / 2
                new_botton.scale = scale
                butil.apply_transform(new_botton, True)
                new_botton.location = location
                butil.apply_transform(new_botton, True)
                return new_botton

            a = node_utils.save_geometry_new(assets, "button", i, self.params.get("i"), self.params.get("path"), first, True, False, material = [self.params['WhiteMetal'], self.scratch, self.edge_wear], parent_obj_id=parent_id, joint_info=joint_info, apply=get_co)
            if a:
                first = False
                break
        shift = self.params['Width'] / (self.params['BottonAmount'] + 3)
        print(bpy.context.object)
        saved_obj = bpy.context.object
        for i in range(2, self.params['BottonAmount'] + 3):
            co = read_co(saved_obj)
            co[:, 1] += shift
            write_co(saved_obj, co)
            if i == self.params['BottonAmount'] / 2 + 1 or i == self.params['BottonAmount'] / 2 + 2:
                continue
            joint_info = {
                "name": get_joint_name("revolute"),
                "type": "revolute",
                "axis": (1, 0, 0),
                "limit": {
                    "lower": -np.pi / 4,
                    "upper": np.pi / 4,
                },
            }
            co_ = read_co(saved_obj)
            min_b = min(min_b, np.min(co_[:, 1]))
            max_b = max(max_b, np.max(co_[:, 1]))
            save_obj_parts_add([saved_obj], self.params.get("path"), self.params.get("i"), "botton", first=False, use_bpy=True, material=[self.params['WhiteMetal'], self.scratch, self.edge_wear], joint_info=joint_info, parent_obj_id=parent_id)
        button_scale = self.params['BottonRadius'] * np.random.uniform(0.5, 1.2)
        button_gap = button_scale * 0.5
        button_number = int((max_b - min_b - button_scale) / (button_scale + button_gap))
        real_gap = (max_b - min_b - button_number * button_scale) / (button_number - 1)
        button_thickness = np.random.uniform(0.003, 0.005)
        x = co_[:, 0].min() + button_thickness / 2
        z = co_[:, 2].min() - button_scale
        use_aux_button = np.random.choice([True, False], p=[0.9, 0.1])
        use_aux_button = False
        all_same = True
        if use_aux_button:
            all_same = np.random.choice([True, False], p=[0.5, 0.5])
        aux_button = None
        for number in range(button_number):
            shape = np.random.choice(['square', 'circle'])
            if shape == 'square':
                button = butil.spawn_cube()
            elif shape == 'circle':
                button = butil.spawn_cylinder()
                button.rotation_euler = (0, np.pi / 2, 0)
                butil.apply_transform(button, True)
                button.scale = (1, 0.5, 0.5)
                butil.apply_transform(button, True)
            button.scale = (button_thickness, button_scale, button_scale * 0.5)
            button.location = (x, min_b + button_scale / 2 + number * (button_scale + real_gap),z)
            butil.apply_transform(button, True)
            if use_aux_button:
                if not all_same:
                    aux_button = butil.deep_clone_obj(random_auxiliary('buttons')[0], keep_materials=False, keep_modifiers=False)
                elif aux_button is None:
                    aux_button = butil.deep_clone_obj(random_auxiliary('buttons')[0], keep_materials=False, keep_modifiers=False)
                #aux_button = aux_button[0]
                aux_button.rotation_euler = np.pi / 2, 0, np.pi / 2
                butil.apply_transform(aux_button, True)
                co = read_co(button)
                co_ = read_co(aux_button)
                scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
                scale_t = co_[:, 0].max() - co_[:, 0].min(), co_[:, 1].max() - co_[:, 1].min(), co_[:, 2].max() - co_[:, 2].min()
                s = scale[0] / scale_t[0], scale[1] / scale_t[1], scale[2] / scale_t[2] 
                aux_button.scale = s
                butil.apply_transform(aux_button, True)
                aux_button.location = co[:, 0].max() + scale[0] / 2, co[:, 1].min() + scale[1] / 2, co[:, 2].min() + scale[2] / 2
                butil.apply_transform(aux_button, True)
                button = aux_button
            save_obj_parts_add([button], self.params.get("path"), self.params.get("i"), "button", first=False, use_bpy=True, material=[self.params['WhiteMetal'], self.scratch, self.edge_wear], parent_obj_id=parent_id, joint_info={
                "name": get_joint_name("prismatic"),
                "type": "prismatic",
                "axis": [1, 0, 0],
                "limit":{
                    "lower": -button_thickness * 0.9,
                    "upper": 0
                }
            })
        for i in range(1, self.params['BottonAmount'] + 1):
            joint_info = {
                "name": get_joint_name("fixed"),
                "type": "fixed"
            }
            a = node_utils.save_geometry_new(assets, "text_", i, self.params.get("i"), self.params.get("path"), first, True, False, material = [self.params['WhiteMetal'], self.scratch, self.edge_wear], parent_obj_id=parent_id, joint_info=joint_info)
            if a:
                first = False
                break
        shift = self.params['Width'] / (self.params['BottonAmount'] + 3)
        print(bpy.context.object)
        saved_obj = bpy.context.object
        for i in range(2, self.params['BottonAmount'] + 3):
            co = read_co(saved_obj)
            co[:, 1] += shift
            write_co(saved_obj, co)
            if i == self.params['BottonAmount'] / 2 + 1 or i == self.params['BottonAmount'] / 2 + 2:
                continue
            joint_info = {
                "name": get_joint_name("fixed"),
                "type": "fixed"
            }
            save_obj_parts_add([saved_obj], self.params.get("path"), self.params.get("i"), "text_", first=False, use_bpy=True, material=[self.params['WhiteMetal'], self.scratch, self.edge_wear], parent_obj_id=parent_id, joint_info=joint_info)
        print(self.params['RackHAmount'])
        parent_id = "world"
        for i in range(1, self.params['RackHAmount'] + 1):
            # joint_info = {
            #     "name": get_joint_name("limited_floating"),
            #     "type": "limited_floating",
            #     "axis": (1, 0, 0),
            #     "limit": {
            #         "lower_1": 0,
            #         "upper_1": self.params['Depth'] * 0.75,
            #         "lower": 0,
            #         "upper": 0,
            #         "lower_2": (self.params['Height'] - self.params['PanelHeight']) / (self.params['RackHAmount']) * (- 0.5),
            #         "upper_2": (self.params['Height'] - self.params['PanelHeight']) / (self.params['RackHAmount']) * (+ 0.5),
            #     },
            # }
            joint_info = {
                "name": get_joint_name("primatic"),
                "type": "prismatic",
                "axis": (1, 0, 0),
                "limit": {
                    "lower": 0,
                    "upper": self.params['Depth'] * 0.75,
                },
            }
            racks= butil.spawn_cube()
            butil.modify_mesh(
                racks,
                "NODES",
                node_group=nodegroup_oven_geometry(
                    preprocess=False, use_gas=self.params["UseGas"], return_part_name="racks"
                ),
                ng_inputs=self.geometry_node_params,
                apply=True,
            )
            a = node_utils.save_geometry_new(racks, "rack", i, self.params.get("i"), self.params.get("path"), False, True, False, material = [self.params['Surface'], self.scratch, self.edge_wear], joint_info=joint_info, parent_obj_id="world")
            if a:
                first = False
        #save_obj_parts_add([assets], self.params.get("path"), self.params.get("i"), "part", first=False, use_bpy=True, material=[self.scratch, self.edge_wear])
        node_utils.save_geometry_new(assets, 'whole', 0, self.params.get("i"), self.params.get("path"), first, True, False)
        save_obj_parts_add([assets], self.params.get("path"), self.params.get("i"), "part", first=False, use_bpy=True, material=[self.scratch, self.edge_wear])
        return assets

        


def gas_grates(
    width,
    depth,
    grate_width,
    grate_depth,
    height,
    thickness,
    grids,
    branches,
    center_ratio,
    middle_ratio,
):
    high_height = height + thickness * 0.9
    grates = []
    for i, n in enumerate(grids):
        cubes = [
            butil.spawn_cube(
                size=1,
                location=(
                    depth / 2,
                    grate_width / len(grids) * i
                    + (width - grate_width) / 2
                    + thickness / 2,
                    height,
                ),
                scale=(grate_depth + thickness, thickness, thickness),
                name=None,
            ),
            butil.spawn_cube(
                size=1,
                location=(
                    depth / 2,
                    grate_width / len(grids) * (i + 1)
                    + (width - grate_width) / 2
                    - thickness / 2,
                    height,
                ),
                scale=(grate_depth + thickness, thickness, thickness),
                name=None,
            ),
        ]
        for j in range(n + 1):
            cubes.append(
                butil.spawn_cube(
                    size=1,
                    location=(
                        grate_depth / n * j + (depth - grate_depth) / 2,
                        grate_width / len(grids) * (i + 0.5)
                        + (width - grate_width) / 2,
                        high_height,
                    ),
                    scale=(thickness, grate_width / len(grids), thickness),
                )
            )
        for j in range(n):
            min_dist = min(grate_width / len(grids) / 2, grate_depth / n / 2)
            line_len = max(grate_width / len(grids) / 2, grate_depth / n / 2) - min_dist
            center_dist = min_dist * center_ratio
            middle_dist = min_dist * middle_ratio
            if grate_width / len(grids) / 2 > grate_depth / n / 2:
                x_center, y_center = center_dist, line_len + center_dist
                x_middle, y_middle = middle_dist, line_len + middle_dist
                x_full, y_full = min_dist, line_len + min_dist
            else:
                x_center, y_center = center_dist + line_len, center_dist
                x_middle, y_middle = middle_dist + line_len, middle_dist
                x_full, y_full = min_dist + line_len, min_dist
            center = (
                (grate_depth / n * (j + 0.5) + (depth - grate_depth) / 2),
                grate_width / len(grids) * (i + 0.5) + (width - grate_width) / 2,
            )
            for k in range(branches):
                angle = 2 * np.pi / branches * k
                x0, y0 = x_center * np.cos(angle), y_center * np.sin(angle)
                x1, y1 = x_middle * np.cos(angle), y_middle * np.sin(angle)
                location = (
                    center[0] + (x0 + x1) / 2,
                    center[1] + (y0 + y1) / 2,
                    high_height,
                )
                scale = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5, thickness, thickness
                actual_angle = np.arctan2(y1 - y0, x1 - x0)
                obj = butil.spawn_cube(size=1, location=location, scale=scale)
                bpy.context.object.rotation_euler[2] = actual_angle
                cubes.append(obj)
                x0, y0 = x1, y1
                if x_full - abs(x0) < y_full - abs(y0):
                    x1, y1 = x_full * np.sign(x0), y0
                else:
                    x1, y1 = x0, y_full * np.sign(y0)
                location = (
                    center[0] + (x0 + x1) / 2,
                    center[1] + (y0 + y1) / 2,
                    high_height,
                )
                scale = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5, thickness, thickness
                actual_angle = np.arctan2(y1 - y0, x1 - x0)
                obj = butil.spawn_cube(size=1, location=location, scale=scale)
                bpy.context.object.rotation_euler[2] = actual_angle
                cubes.append(obj)
            grates.append(
                butil.spawn_cylinder(
                    center_dist + thickness,
                    thickness / 2,
                    location=(center[0], center[1], height),
                )
            )
        obj = butil.boolean(cubes)
        for i in range(1, len(cubes)):
            butil.delete(cubes[i])
        with butil.SelectObjects(obj):
            bpy.ops.object.modifier_add(type="REMESH")
            remesh_type = "VOXEL"
            bpy.context.object.modifiers["Remesh"].mode = remesh_type
            bpy.context.object.modifiers["Remesh"].voxel_size = 0.004
            bpy.ops.object.modifier_apply(modifier="Remesh")
            bpy.ops.object.modifier_add(type="SMOOTH")
            bpy.context.object.modifiers["Smooth"].iterations = 8
            bpy.context.object.modifiers["Smooth"].factor = 1
            bpy.ops.object.modifier_apply(modifier="Smooth")
        grates.append(obj)
    obj = butil.boolean(grates)
    for i in range(1, len(grates)):
        butil.delete(grates[i])
    return obj


@node_utils.to_nodegroup(
    "nodegroup_hollow_cube", singleton=False, type="GeometryNodeTree"
)
def nodegroup_hollow_cube(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVectorTranslation", "Size", (0.1000, 10.0000, 4.0000)),
            ("NodeSocketVector", "Pos", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "Resolution", 2),
            ("NodeSocketFloat", "Thickness", 0.0000),
            ("NodeSocketBool", "Switch1", False),
            ("NodeSocketBool", "Switch2", False),
            ("NodeSocketBool", "Switch3", False),
            ("NodeSocketBool", "Switch4", False),
            ("NodeSocketBool", "Switch5", False),
            ("NodeSocketBool", "Switch6", False),
        ],
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Size"]}
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Thickness"],
            "Y": subtract,
            "Z": subtract_1,
        },
    )

    cube_2 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_4,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_2.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": group_input.outputs["Pos"]}
    )

    add = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_1, 1: separate_xyz_1.outputs["X"]}
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Size"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    separate_xyz_2 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": scale.outputs["Vector"]}
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": subtract_2}
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_1,
            "Translation": combine_xyz_5,
        },
    )

    switch_2 = nw.new_node(
        Nodes.Switch, input_kwargs={1: group_input.outputs["Switch3"], 14: transform_2}
    )

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": subtract_3,
            "Z": group_input.outputs["Thickness"],
        },
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_2,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    subtract_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_3 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_2, "Y": add_3, "Z": subtract_4}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_4,
            "Translation": combine_xyz_3,
        },
    )

    switch_1 = nw.new_node(
        Nodes.Switch, input_kwargs={1: group_input.outputs["Switch2"], 14: transform_1}
    )

    subtract_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": subtract_5,
            "Z": group_input.outputs["Thickness"],
        },
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    add_6 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply_1, 1: separate_xyz_1.outputs["Z"]}
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_4, "Y": add_5, "Z": add_6}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute, "Translation": combine_xyz_1},
    )

    switch = nw.new_node(
        Nodes.Switch, input_kwargs={1: group_input.outputs["Switch1"], 14: transform}
    )

    subtract_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    subtract_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: multiply},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Thickness"],
            "Y": subtract_6,
            "Z": subtract_7,
        },
    )

    cube_3 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_6,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_3.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_3.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    subtract_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Y"], 1: separate_xyz_1.outputs["Y"]},
    )

    subtract_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
        attrs={"operation": "SUBTRACT"},
    )

    combine_xyz_7 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": subtract_8, "Y": add_7, "Z": subtract_9}
    )

    transform_3 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_5,
            "Translation": combine_xyz_7,
        },
    )

    switch_3 = nw.new_node(
        Nodes.Switch, input_kwargs={1: group_input.outputs["Switch4"], 14: transform_3}
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": group_input.outputs["Thickness"],
            "Z": separate_xyz.outputs["Z"],
        },
    )

    cube_4 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_9,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_4.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_4.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["X"], 1: separate_xyz_2.outputs["X"]},
    )

    add_9 = nw.new_node(
        Nodes.Math, input_kwargs={0: separate_xyz_1.outputs["Y"], 1: multiply_1}
    )

    add_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_1.outputs["Z"], 1: separate_xyz_2.outputs["Z"]},
    )

    combine_xyz_8 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_8, "Y": add_9, "Z": add_10}
    )

    transform_4 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_2,
            "Translation": combine_xyz_8,
        },
    )

    switch_4 = nw.new_node(
        Nodes.Switch, input_kwargs={1: group_input.outputs["Switch5"], 14: transform_4}
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": separate_xyz.outputs["X"],
            "Y": group_input.outputs["Thickness"],
            "Z": separate_xyz.outputs["Z"],
        },
    )

    cube_5 = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": combine_xyz_10,
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_5.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_5.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    add_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["X"], 1: separate_xyz_1.outputs["X"]},
    )

    subtract_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: multiply_1},
        attrs={"operation": "SUBTRACT"},
    )

    add_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz_2.outputs["Z"], 1: separate_xyz_1.outputs["Z"]},
    )

    combine_xyz_11 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_11, "Y": subtract_10, "Z": add_12}
    )

    transform_5 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_3,
            "Translation": combine_xyz_11,
        },
    )

    switch_5 = nw.new_node(
        Nodes.Switch, input_kwargs={1: group_input.outputs["Switch6"], 14: transform_5}
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [
                switch_2.outputs[6],
                switch_1.outputs[6],
                switch.outputs[6],
                switch_3.outputs[6],
                switch_4.outputs[6],
                switch_5.outputs[6],
            ]
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_o", singleton=False, type="GeometryNodeTree")
def nodegroup_o(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"End": (0.0000, 0.0000, 0.0020)}
    )

    group_input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketFloatDistance", "Size", 1.0000)]
    )

    curve_circle_1 = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": group_input.outputs["Size"]}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": curve_circle_1.outputs["Curve"],
        },
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh, input_kwargs={"Mesh": curve_to_mesh, "Offset Scale": 0.0030}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": extrude_mesh.outputs["Mesh"]},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_heater", singleton=False, type="GeometryNodeTree")
def nodegroup_heater(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    curve_line_1 = nw.new_node(
        Nodes.CurveLine, input_kwargs={"End": (0.0000, 0.0000, 0.0010)}
    )

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "width", 0.5000),
            ("NodeSocketFloat", "depth", 0.0000),
            ("NodeSocketFloat", "radius_ratio", 0.2000),
            ("NodeSocketFloat", "arrangement_ratio", 0.5000),
            ("NodeSocketShader", "SuperBlackGlass", None),
        ],
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"], 1: group_input.outputs["depth"]},
        attrs={"operation": "MINIMUM"},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: group_input.outputs["radius_ratio"]},
        label="Multiply",
        attrs={"operation": "MULTIPLY"},
    )

    curve_circle_1 = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": multiply})

    curve_to_mesh_1 = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line_1,
            "Profile Curve": curve_circle_1.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": curve_to_mesh_1,
            "Material": group_input.outputs["SuperBlackGlass"],
        },
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": set_material}
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: minimum, 1: group_input.outputs["arrangement_ratio"]},
        label="Multiply",
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"], 1: multiply_1},
        attrs={"operation": "DIVIDE"},
    )

    floor = nw.new_node(
        Nodes.Math, input_kwargs={0: divide}, attrs={"operation": "FLOOR"}
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"], 1: multiply_1},
        attrs={"operation": "DIVIDE"},
    )

    floor_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_1}, attrs={"operation": "FLOOR"}
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: floor, 1: floor_1}, attrs={"operation": "MULTIPLY"}
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance, "Amount": multiply_2},
        attrs={"domain": "INSTANCE"},
    )

    divide_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["depth"], 1: floor_1},
        attrs={"operation": "DIVIDE"},
    )

    divide_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: floor},
        attrs={"operation": "DIVIDE"},
    )

    floor_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: divide_3}, attrs={"operation": "FLOOR"}
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: floor_2, 1: divide_2},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_2, 2: multiply_3},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    divide_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"], 1: floor},
        attrs={"operation": "DIVIDE"},
    )

    modulo = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: floor},
        attrs={"operation": "MODULO"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: modulo, 1: divide_4},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: divide_4, 2: multiply_4},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": multiply_add, "Y": multiply_add_1}
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": set_position},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_oven_rack", singleton=False, type="GeometryNodeTree"
)
def nodegroup_oven_rack(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloatDistance", "Width", 2.0000),
            ("NodeSocketFloatDistance", "Height", 2.0000),
            ("NodeSocketFloatDistance", "Radius", 0.0200),
            ("NodeSocketInt", "Amount", 5),
        ],
    )

    quadrilateral = nw.new_node(
        "GeometryNodeCurvePrimitiveQuadrilateral",
        input_kwargs={
            "Width": group_input.outputs["Width"],
            "Height": group_input.outputs["Height"],
        },
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply})

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_4 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_1})

    curve_line = nw.new_node(
        Nodes.CurveLine, input_kwargs={"Start": combine_xyz_3, "End": combine_xyz_4}
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": curve_line}
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={
            "Geometry": geometry_to_instance,
            "Amount": group_input.outputs["Amount"],
        },
        attrs={"domain": "INSTANCE"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_2, 1: group_input.outputs["Amount"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: divide},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_3})

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Offset": combine_xyz,
        },
    )

    duplicate_elements_1 = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={
            "Geometry": geometry_to_instance,
            "Amount": group_input.outputs["Amount"],
        },
        attrs={"domain": "INSTANCE"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_4, 1: group_input.outputs["Amount"]},
        attrs={"operation": "DIVIDE"},
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements_1.outputs["Duplicate Index"], 1: divide_1},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_1 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"X": multiply_5})

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": duplicate_elements_1.outputs["Geometry"],
            "Offset": combine_xyz_1,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [quadrilateral, set_position, set_position_1]},
    )

    curve_circle = nw.new_node(
        Nodes.CurveCircle, input_kwargs={"Radius": group_input.outputs["Radius"]}
    )

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": join_geometry,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Mesh": curve_to_mesh},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_text", singleton=False, type="GeometryNodeTree")
def nodegroup_text(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVectorTranslation", "Translation", (1.5000, 0.0000, 0.0000)),
            ("NodeSocketString", "String", "BrandName"),
            ("NodeSocketFloatDistance", "Size", 0.0500),
            ("NodeSocketFloat", "Offset Scale", 0.0020),
        ],
    )

    string_to_curves = nw.new_node(
        "GeometryNodeStringToCurves",
        input_kwargs={
            "String": group_input.outputs["String"],
            "Size": group_input.outputs["Size"],
        },
        attrs={"align_y": "BOTTOM_BASELINE", "align_x": "CENTER"},
    )

    fill_curve = nw.new_node(
        Nodes.FillCurve,
        input_kwargs={"Curve": string_to_curves.outputs["Curve Instances"]},
    )

    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={
            "Mesh": fill_curve,
            "Offset Scale": group_input.outputs["Offset Scale"],
        },
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": extrude_mesh.outputs["Mesh"],
            "Translation": group_input.outputs["Translation"],
            "Rotation": (1.5708, 0.0000, 1.5708),
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform_1},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "width", 0.0000),
            ("NodeSocketFloat", "length", 0.0000),
            ("NodeSocketFloat", "thickness", 0.0200),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["width"]}
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    cube_1 = nw.new_node(
        Nodes.MeshCube, input_kwargs={"Size": group_input.outputs["width"]}
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_1.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["length"]}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_named_attribute_1, "Translation": combine_xyz},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_named_attribute, transform]},
    )

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": multiply})

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": combine_xyz_3},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["length"],
            1: group_input.outputs["width"],
        },
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["width"],
            "Y": add,
            "Z": group_input.outputs["thickness"],
        },
    )

    cube_2 = nw.new_node(Nodes.MeshCube, input_kwargs={"Size": combine_xyz_1})

    store_named_attribute_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_2.outputs["Mesh"],
            "Name": "uv_map",
            3: cube_2.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["length"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["thickness"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["width"], 1: multiply_2}
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": multiply_1, "Z": add_1}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute_2,
            "Translation": combine_xyz_2,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [transform_2, transform_1]}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_center", singleton=False, type="GeometryNodeTree")
def nodegroup_center(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketGeometry", "Geometry", None),
            ("NodeSocketVector", "Vector", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketFloat", "MarginX", 0.5000),
            ("NodeSocketFloat", "MarginY", 0.0000),
            ("NodeSocketFloat", "MarginZ", 0.0000),
        ],
    )

    bounding_box = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": group_input.outputs["Geometry"]}
    )

    subtract = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: group_input.outputs["Vector"], 1: bounding_box.outputs["Min"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract.outputs["Vector"]}
    )

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["X"], 1: group_input.outputs["MarginX"]},
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    subtract_1 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Max"], 1: group_input.outputs["Vector"]},
        attrs={"operation": "SUBTRACT"},
    )

    separate_xyz_1 = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": subtract_1.outputs["Vector"]}
    )

    greater_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["X"],
            1: group_input.outputs["MarginX"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than, 1: greater_than_1}
    )

    greater_than_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Y"], 1: group_input.outputs["MarginY"]},
        attrs={"operation": "GREATER_THAN"},
    )

    greater_than_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["Y"],
            1: group_input.outputs["MarginY"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and_1 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than_2, 1: greater_than_3}
    )

    op_and_2 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and, 1: op_and_1})

    greater_than_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: separate_xyz.outputs["Z"], 1: group_input.outputs["MarginZ"]},
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    greater_than_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: separate_xyz_1.outputs["Z"],
            1: group_input.outputs["MarginZ"],
        },
        attrs={"operation": "GREATER_THAN", "use_clamp": True},
    )

    op_and_3 = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: greater_than_4, 1: greater_than_5}
    )

    op_and_4 = nw.new_node(Nodes.BooleanMath, input_kwargs={0: op_and_2, 1: op_and_3})

    op_not = nw.new_node(
        Nodes.BooleanMath, input_kwargs={0: op_and_4}, attrs={"operation": "NOT"}
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"In": op_and_4, "Out": op_not},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_cube", singleton=False, type="GeometryNodeTree")
def nodegroup_cube(nw: NodeWrangler):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketVectorTranslation", "Size", (0.1000, 10.0000, 4.0000)),
            ("NodeSocketVector", "Pos", (0.0000, 0.0000, 0.0000)),
            ("NodeSocketInt", "Resolution", 2),
        ],
    )

    cube = nw.new_node(
        Nodes.MeshCube,
        input_kwargs={
            "Size": group_input.outputs["Size"],
            "Vertices X": group_input.outputs["Resolution"],
            "Vertices Y": group_input.outputs["Resolution"],
            "Vertices Z": group_input.outputs["Resolution"],
        },
    )

    store_named_attribute_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube.outputs["Mesh"],
            "Name": "uv_map",
            3: cube.outputs["UV Map"],
        },
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    store_named_attribute = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": store_named_attribute_1, "Name": "uv_map"},
        attrs={"domain": "CORNER", "data_type": "FLOAT_VECTOR"},
    )

    multiply_add = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={
            0: group_input.outputs["Size"],
            1: (0.5000, 0.5000, 0.5000),
            2: group_input.outputs["Pos"],
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": store_named_attribute,
            "Translation": multiply_add.outputs["Vector"],
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": transform},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup(
    "nodegroup_oven_geometry", singleton=False, type="GeometryNodeTree"
)
def nodegroup_oven_geometry(
    nw: NodeWrangler,
    preprocess: bool = False,
    use_gas: bool = False,
    is_placeholder: bool = False,
    return_part_name: str =""
):
    # Code generated using version 2.6.5 of the node_transpiler

    group_input = nw.new_node(
        Nodes.GroupInput,
        expose_input=[
            ("NodeSocketFloat", "Depth", 1.0000),
            ("NodeSocketFloat", "Width", 1.0000),
            ("NodeSocketFloat", "Height", 1.0000),
            ("NodeSocketFloat", "DoorThickness", 0.0700),
            ("NodeSocketFloat", "DoorRotation", 0.0000),
            ("NodeSocketFloatDistance", "RackRadius", 0.0100),
            ("NodeSocketInt", "RackHAmount", 2),
            ("NodeSocketInt", "RackDAmount", 5),
            ("NodeSocketFloat", "PanelHeight", 0.3000),
            ("NodeSocketFloat", "PanelThickness", 0.2000),
            ("NodeSocketInt", "BottonAmount", 4),
            ("NodeSocketFloatDistance", "BottonRadius", 0.0500),
            ("NodeSocketFloat", "BottonThickness", 0.0300),
            ("NodeSocketFloat", "HeaterRadiusRatio", 0.1500),
            ("NodeSocketString", "BrandName", "BrandName"),
            ("NodeSocketMaterial", "Glass", None),
            ("NodeSocketMaterial", "Surface", None),
            ("NodeSocketMaterial", "WhiteMetal", None),
            ("NodeSocketMaterial", "SuperBlackGlass", None),
            ("NodeSocketMaterial", "Back", None),
            ("NodeSocketBool", "is_placeholder", is_placeholder),
        ],
    )

    combine_xyz_1 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["DoorThickness"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    combine_xyz_2 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Depth"]}
    )

    cube = nw.new_node(
        nodegroup_cube().name,
        input_kwargs={"Size": combine_xyz_1, "Pos": combine_xyz_2},
    )

    position = nw.new_node(Nodes.InputPosition)

    center = nw.new_node(
        nodegroup_center().name,
        input_kwargs={
            "Geometry": cube,
            "Vector": position,
            "MarginX": -1.0000,
            "MarginY": 0.1000,
            "MarginZ": 0.1500,
        },
    )

    set_material_2 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": cube,
            "Selection": center.outputs["In"],
            "Material": group_input.outputs["Glass"],
        },
    )

    set_material_3 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_material_2,
            "Selection": center.outputs["Out"],
            "Material": group_input.outputs["Surface"],
        },
    )

    # store = nw.new_node(
    #     Nodes.StoreNamedAttribute,
    #     input_kwargs={"Geometry": center.outputs['Out'], "Name": "Out", "Value": 1},
    #     attrs={"domain": "FACE", "data_type": "INT"},
    # )
    # store_1 = nw.new_node(
    #     Nodes.StoreNamedAttribute,
    #     input_kwargs={"Geometry": center.outputs['In'], "Name": "In", "Value": 1},
    #     attrs={"domain": "FACE", "data_type": "INT"},
    # )

    set_material_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material_3]},
    )

    #set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': set_material_3})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "MULTIPLY"}
    )

    handle = nw.new_node(
        nodegroup_handle().name,
        input_kwargs={"width": multiply, "length": multiply_1, "thickness": multiply_2},
    )

    add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    multiply_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_1, 1: -0.5000},
        attrs={"operation": "MULTIPLY"},
    )

    add_1 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_3, 1: multiply_4})

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.9200},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_13 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": add_1, "Z": multiply_5}
    )

    transform_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": handle,
            "Translation": combine_xyz_13,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    set_material_8 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": transform_1,
            "Material": group_input.outputs["WhiteMetal"],
        },
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_2, "Y": multiply_6, "Z": 0.0300}
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    text = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_12,
            "String": group_input.outputs["BrandName"],
            "Size": multiply_7,
        },
    )

    text = complete_no_bevel(nw, text, preprocess)

    set_material_9 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": text, "Material": group_input.outputs["WhiteMetal"]},
    )

    set_material_9 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_material_9,
            "Name": "text",
            "Value": 1,
        },
    )

    set_material_8 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_material_8,
            "Name": "handle",
            "Value": 1,
        },
        attrs={"domain": "FACE", "data_type": "INT"},
    )

    set_material_8 = complete_bevel(nw, set_material_8, preprocess)

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material_3, set_material_8, set_material_9]},
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry_3}
    )

    y = nw.scalar_multiply(
        group_input.outputs["DoorRotation"], 1 if not preprocess else 0
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": y})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": group_input.outputs["Depth"]}
    )

    rotate_instances = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={
            "Instances": geometry_to_instance,
            "Rotation": combine_xyz_3,
            "Pivot Point": combine_xyz_4,
        },
    )

    rotate_instances = nw.new_node(Nodes.RealizeInstances, [rotate_instances])

    door = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": join_geometry_3}, label="door"
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: multiply_8},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: multiply_9},
        attrs={"operation": "SUBTRACT"},
    )

    ovenrack = nw.new_node(
        nodegroup_oven_rack().name,
        input_kwargs={
            "Width": subtract,
            "Height": subtract_1,
            "Radius": group_input.outputs["RackRadius"],
            "Amount": group_input.outputs["RackDAmount"],
        },
    )

    geometry_to_instance_1 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": ovenrack}
    )

    duplicate_elements = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={
            "Geometry": geometry_to_instance_1,
            "Amount": group_input.outputs["RackHAmount"],
        },
        attrs={"domain": "INSTANCE"},
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: 1.0000},
    )

    store = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": duplicate_elements, "Name": "rack", "Value": add_3},
        attrs={"domain": "INSTANCE", "data_type": "INT"},
    )

    multiply_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: multiply_12},
        attrs={"operation": "SUBTRACT"},
    )

    add_4 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["RackHAmount"], 1: 1.0000}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: add_4},
        attrs={"operation": "DIVIDE"},
    )

    multiply_13 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_3, 1: divide}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_10, "Y": multiply_11, "Z": multiply_13},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": store.outputs["Geometry"],
            "Offset": combine_xyz_5,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_position,
            "Material": group_input.outputs["Surface"],
        },
    )

    set_material = nw.new_node(Nodes.RealizeInstances, [set_material])

    racks = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": set_material}, label="racks"
    )

    add_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_5})

    reroute_11 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    reroute_8 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["DoorThickness"]}
    )

    combine_xyz_6 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_10, "Y": reroute_11, "Z": reroute_8},
    )

    reroute_9 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Height"]}
    )

    combine_xyz_7 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": reroute_9})

    cube_1 = nw.new_node(
        nodegroup_cube().name,
        input_kwargs={"Size": combine_xyz_6, "Pos": combine_xyz_7},
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": cube_1, "Material": group_input.outputs["Back"]},
    )

    # set_shade_smooth_1 = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': set_material_5})

    subtract_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_10, 1: group_input.outputs["PanelThickness"]},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_add = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["HeaterRadiusRatio"],
            1: 2.0000,
            2: 0.1000,
        },
        attrs={"operation": "MULTIPLY_ADD"},
    )

    heater = nw.new_node(
        nodegroup_heater().name,
        input_kwargs={
            "width": reroute_11,
            "depth": subtract_3,
            "radius_ratio": group_input.outputs["HeaterRadiusRatio"],
            "arrangement_ratio": multiply_add,
        },
    )

    add_6 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_8, 1: reroute_9})

    combine_xyz_15 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": group_input.outputs["PanelThickness"], "Z": add_6},
    )

    transform_2 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": heater, "Translation": combine_xyz_15},
    )

    transform_2 = complete_no_bevel(nw, transform_2, preprocess)

    set_material_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_material_5,
            "Name": "back",
            "Value": 1,
        },
        attrs={"domain": "FACE", "data_type": "INT"},
    )

    if use_gas:
        join_geometry_2 = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_5]}
        )
    else:
        join_geometry_2 = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_5, transform_2]}
        )

    heater_1 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": join_geometry_2}, label="heater"
    )

    reroute_14 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["Width"]}
    )

    combine_xyz_9 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["PanelThickness"],
            "Y": reroute_14,
            "Z": group_input.outputs["PanelHeight"],
        },
    )

    add_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Height"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    combine_xyz_8 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": add_7})

    cube_2 = nw.new_node(
        nodegroup_cube().name,
        input_kwargs={"Size": combine_xyz_9, "Pos": combine_xyz_8},
    )

    position_1 = nw.new_node(Nodes.InputPosition)

    center_1 = nw.new_node(
        nodegroup_center().name,
        input_kwargs={
            "Geometry": cube_2,
            "Vector": position_1,
            "MarginX": -1.0000,
            "MarginY": 0.0500,
            "MarginZ": 0.0500,
        },
    )

    set_material_4 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": cube_2,
            "Selection": center_1.outputs["In"],
            "Material": group_input.outputs["Back"],
        },
    )

    set_material_100 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_material_4,
            "Selection": center_1.outputs["Out"],
            "Material": group_input.outputs["Surface"],
        },
    )

    set_material_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_material_100,
            "Name": "Out",
            "Value": 1,
        },
        attrs={"domain": "FACE", "data_type": "INT"},
    )
    set_material_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": set_material_4,
            "Name": "In",
            "Value": 1,
        },
        attrs={"domain": "FACE", "data_type": "INT"},
    )

    set_material_7 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material_100]},
    )

    # set_shade_smooth_3 = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': set_material_7})

    reroute_13 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["PanelThickness"]}
    )

    multiply_14 = nw.new_node(
        Nodes.Math, input_kwargs={0: reroute_14}, attrs={"operation": "MULTIPLY"}
    )

    bounding_box = nw.new_node(Nodes.BoundingBox, input_kwargs={"Geometry": cube_2})

    add_8 = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: bounding_box.outputs["Min"], 1: bounding_box.outputs["Max"]},
    )

    scale = nw.new_node(
        Nodes.VectorMath,
        input_kwargs={0: add_8.outputs["Vector"], "Scale": 0.5000},
        attrs={"operation": "SCALE"},
    )

    separate_xyz = nw.new_node(
        Nodes.SeparateXYZ, input_kwargs={"Vector": scale.outputs["Vector"]}
    )

    combine_xyz_16 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": reroute_13,
            "Y": multiply_14,
            "Z": separate_xyz.outputs["Z"],
        },
    )

    multiply_15 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["PanelHeight"], 1: 0.2000},
        attrs={"operation": "MULTIPLY"},
    )

    text_1 = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_16,
            "String": "12:01",
            "Size": multiply_15,
        },
    )

    set_material_7 = complete_bevel(nw, set_material_7, preprocess)
    text_1 = complete_no_bevel(nw, text_1, preprocess)

    join_geometry_5 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [set_material_7, text_1]}
    )

    combine_xyz_21 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["BottonThickness"]}
    )

    curve_line = nw.new_node(Nodes.CurveLine, input_kwargs={"End": combine_xyz_21})

    reroute_12 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BottonRadius"]}
    )

    curve_circle = nw.new_node(Nodes.CurveCircle, input_kwargs={"Radius": reroute_12})

    curve_to_mesh = nw.new_node(
        Nodes.CurveToMesh,
        input_kwargs={
            "Curve": curve_line,
            "Profile Curve": curve_circle.outputs["Curve"],
            "Fill Caps": True,
        },
    )

    add_9 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_12, 1: 0.0050})

    o = nw.new_node(nodegroup_o().name, input_kwargs={"Size": add_9})

    join_geometry_4 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [curve_to_mesh, o]}
    )

    combine_xyz_10 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_13, "Z": separate_xyz.outputs["Z"]}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": join_geometry_4,
            "Translation": combine_xyz_10,
            "Rotation": (0.0000, 1.5708, 0.0000),
        },
    )

    reroute_16 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": separate_xyz.outputs["Z"]}
    )

    reroute_15 = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": group_input.outputs["BottonRadius"]}
    )

    multiply_16 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["PanelHeight"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_15, 1: 1.0000, 2: multiply_16},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    add_10 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_16, 1: multiply_add_1})

    combine_xyz_17 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": reroute_13, "Z": add_10}
    )

    multiply_17 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["BottonRadius"], 1: 0.2500},
        attrs={"operation": "MULTIPLY"},
    )

    text_2 = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_17,
            "String": "Off",
            "Size": multiply_17,
        },
    )

    multiply_add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_15, 1: 0.7000, 2: multiply_16},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    add_11 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_16, 1: multiply_add_2})

    combine_xyz_18 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_13, "Y": multiply_add_2, "Z": add_11},
    )

    text_3 = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_18,
            "String": "High",
            "Size": multiply_17,
        },
    )

    multiply_18 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: multiply_16, 1: -1.0000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_add_3 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_15, 1: -0.7000, 2: multiply_18},
        attrs={"operation": "MULTIPLY_ADD"},
    )

    combine_xyz_19 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": reroute_13, "Y": multiply_add_3, "Z": add_11},
    )

    text_4 = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_19,
            "String": "Low",
            "Size": multiply_17,
        },
    )

    add_12 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: reroute_13, 1: group_input.outputs["BottonThickness"]},
    )

    combine_xyz_20 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_12, "Z": separate_xyz.outputs["Z"]}
    )

    multiply_19 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["BottonThickness"], 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    text_5 = nw.new_node(
        nodegroup_text().name,
        input_kwargs={
            "Translation": combine_xyz_20,
            "String": "1",
            "Size": group_input.outputs["BottonRadius"],
            "Offset Scale": multiply_19,
        },
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [text_2, text_3, text_4]},
    )
    geometry_to_instance_100 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry_6}
    )

    join_geometry_6 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [transform, text_5]},
    )

    geometry_to_instance_2 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry_6}
    )

    add_13 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["BottonAmount"], 1: 2.0000}
    )

    reroute_6 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_13})

    duplicate_elements_1 = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance_2, "Amount": reroute_6},
        attrs={"domain": "INSTANCE"},
    )

    duplicate_elements_2 = nw.new_node(
        Nodes.DuplicateElements,
        input_kwargs={"Geometry": geometry_to_instance_100, "Amount": reroute_6},
        attrs={"domain": "INSTANCE"},
    )

    add_14 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements_1.outputs["Duplicate Index"], 1: 1.0000},
    )

    add_100 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements_2.outputs["Duplicate Index"], 1: 1.0000},
    )

    store_6 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": duplicate_elements_2, "Name": "text_", "Value": add_100},
        attrs={"domain": "INSTANCE", "data_type": "INT"},
    )

    store_7 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={"Geometry": duplicate_elements_1, "Name": "button", "Value": add_14},
        attrs={"domain": "INSTANCE", "data_type": "INT"},
    )

    join_geometry_100 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [store_6, store_7]}
    )

    add_15 = nw.new_node(Nodes.Math, input_kwargs={0: reroute_6, 1: 1.0000})

    divide_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: add_15},
        attrs={"operation": "DIVIDE"},
    )

    multiply_20 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_14, 1: divide_1},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_100 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: add_100, 1: divide_1},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_11 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_20})
    combine_xyz_100 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Y": multiply_100})

    set_position_1 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": store_7.outputs["Geometry"],
            "Offset": combine_xyz_11,
        },
    )

    set_position_100 = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": store_6.outputs["Geometry"],
            "Offset": combine_xyz_100,
        },
    )
    

    multiply_21 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_13}, attrs={"operation": "MULTIPLY"}
    )

    add_16 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_21, 1: -1.0100})

    greater_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements_1.outputs["Duplicate Index"], 1: add_16},
        attrs={"operation": "GREATER_THAN"},
    )

    add_17 = nw.new_node(Nodes.Math, input_kwargs={0: multiply_21, 1: 0.9900})

    less_than = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements_1.outputs["Duplicate Index"], 1: add_17},
        attrs={"operation": "LESS_THAN"},
    )

    less_than_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements_2.outputs["Duplicate Index"], 1: add_17},
        attrs={"operation": "LESS_THAN"},
    )

    minimum = nw.new_node(
        Nodes.Math,
        input_kwargs={0: greater_than, 1: less_than},
        attrs={"operation": "MINIMUM"},
    )

    minimum_1 = nw.new_node(    
        Nodes.Math,
        input_kwargs={0: greater_than, 1: less_than_1},
        attrs={"operation": "MINIMUM"},
    )

    delete_geometry = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": set_position_1, "Selection": minimum},
        attrs={"domain": "INSTANCE"},
    )

    delete_geometry_1 = nw.new_node(
        Nodes.DeleteGeometry,
        input_kwargs={"Geometry": set_position_100, "Selection": minimum_1},
        attrs={"domain": "INSTANCE"},
    )


    realized = nw.new_node(Nodes.RealizeInstances, [delete_geometry])
    realized_1 = nw.new_node(Nodes.RealizeInstances, [delete_geometry_1])
    joined = nw.new_node(Nodes.JoinGeometry, input_kwargs={"Geometry": [realized, realized_1]})

    set_material_6 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": joined,
            "Material": group_input.outputs["WhiteMetal"],
        },
    )

    botton = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": set_material_6}, label="botton"
    )

    botton = complete_no_bevel(nw, botton, preprocess)

    if return_part_name == "panel":
        panel = nw.new_node(
            Nodes.RealizeInstances, [join_geometry_5]
        )
        group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": panel})
        return

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [join_geometry_5, botton]}
    )

    geometry_to_instance_3 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry_1}
    )

    combine_xyz_14 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Z": group_input.outputs["Height"]}
    )

    panel_bbox = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": geometry_to_instance_3}
    )

    switch_1 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["is_placeholder"],
            "False": geometry_to_instance_3,
            "True": panel_bbox,
        },
    )
    angle = 0
    rotate_instances_1 = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={
            "Instances": switch_1,
            "Rotation": (0.0000, angle, 0.0000),
            "Pivot Point": combine_xyz_14,
        },
    )

    rotate_instances_1 = nw.new_node(Nodes.RealizeInstances, [rotate_instances_1])

    panel = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": rotate_instances_1}, label="panel"
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Depth"],
            "Y": group_input.outputs["Width"],
            "Z": group_input.outputs["Height"],
        },
    )

    hollowcube = nw.new_node(
        nodegroup_hollow_cube().name,
        input_kwargs={
            "Size": combine_xyz,
            "Thickness": group_input.outputs["DoorThickness"],
            "Switch2": True,
            "Switch4": True,
        },
    )

    set_material_1 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": hollowcube,
            "Material": group_input.outputs["Surface"],
        },
    )

    subdivide_mesh = nw.new_node(
        Nodes.SubdivideMesh, input_kwargs={"Mesh": set_material_1, "Level": 0}
    )

    # set_shade_smooth_2 = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': subdivide_mesh})

    body = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": subdivide_mesh}, label="Body"
    )

    if return_part_name == "door":
        door = nw.new_node(
            Nodes.RealizeInstances, [door]
        )
        group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": door})
        return 

    body = nw.new_node(
        Nodes.RealizeInstances,
        [body],
    )

    # store_1 = nw.new_node(
    #     Nodes.StoreNamedAttribute,
    #     input_kwargs={"Geometry": door, "Name": "part", "Value": 1},
    # )

    if return_part_name == "racks":
        group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": racks})
        return
    # store_2 = nw.new_node(
    #     Nodes.StoreNamedAttribute,
    #     input_kwargs={"Geometry": racks, "Name": "part", "Value": 2},
    # )

    # store_3 = nw.new_node(
    #     Nodes.StoreNamedAttribute,
    #     input_kwargs={"Geometry": heater_1, "Name": "part", "Value": 3},
    # )

    if return_part_name == "heater":
        group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": heater_1})
        return

    # store_4 = nw.new_node(
    #     Nodes.StoreNamedAttribute,
    #     input_kwargs={"Geometry": panel, "Name": "part", "Value": 4},
    # )

    # store_5 = nw.new_node(
    #     Nodes.StoreNamedAttribute,
    #     input_kwargs={"Geometry": body, "Name": "part", "Value": 5},
    # )

    if return_part_name == "body":
        group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": body})
        return

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [door, racks, heater_1, panel, body]},
    )

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [door, racks, heater_1, body]}
    )
    body_bbox = nw.new_node(
        Nodes.BoundingBox, input_kwargs={"Geometry": join_geometry_2}
    )
    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [body_bbox, panel]}
    )

    switch_2 = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            "Switch": group_input.outputs["is_placeholder"],
            "False": join_geometry,
            "True": join_geometry_3,
        },
    )
    geometry = nw.new_node(Nodes.RealizeInstances, [switch_2])

    group_output = nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": geometry})
