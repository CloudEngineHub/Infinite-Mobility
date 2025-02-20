# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo

import bpy
import numpy as np
from numpy.random import choice, uniform, randint

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.seating.chairs.seats.curvy_seats import (
    generate_curvy_seats,
)
from infinigen.assets.objects.tables.cocktail_table import geometry_create_legs
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.assets.utils.object import join_objects, save_file_path
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.nodes.node_utils import save_geometry, save_geometry_new
import random
from infinigen.assets.objects.seating.chairs.chair import ChairFactory

from infinigen.assets.utils.object import (
    join_objects_save_whole,
    save_file_path_obj,
    save_obj_parts_add,
    add_joint,
    get_joint_name,
)
from infinigen.assets.utils.auxiliary_parts import random_auxiliary

import bmesh
co_seat = None
def geometry_assemble_chair(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    global co_seat
    generateseat = nw.new_node(
        generate_curvy_seats().name,
        input_kwargs={
            "Width": kwargs["Top Profile Width"],
            "Front Relative Width": kwargs["Top Front Relative Width"],
            "Front Bent": kwargs["Top Front Bent"],
            "Seat Bent": kwargs["Top Seat Bent"],
            "Mid Bent": kwargs["Top Mid Bent"],
            "Mid Relative Width": kwargs["Top Mid Relative Width"],
            "Back Bent": kwargs["Top Back Bent"],
            "Back Relative Width": kwargs["Top Back Relative Width"],
            "Mid Pos": kwargs["Top Mid Pos"],
            "Seat Height": kwargs["Top Thickness"],
        },
    )

    seat_instance = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": generateseat,
            "Translation": (0.0000, 0.0000, kwargs["Top Height"]),
        },
    )

    seat_instance = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": seat_instance, "Material": kwargs["TopMaterial"]},
    )

    legs = nw.new_node(geometry_create_legs(**kwargs).name)

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [seat_instance, legs]}
    )

    names = ["seat", "legs"]
    parts_legs = {
        "single_stand": 1,
        "straight": kwargs["Leg Number"] * 2,
        "wheeled": kwargs.get("Leg Pole Number", 0) + 2
    }
    parts = [3, parts_legs[kwargs["Leg Style"]]]
    if kwargs.get("aux_back", None) is not None and kwargs.get("aux_seat_whole", None) is None:
        seat_idx = 1
    else:
        seat_idx = 0

    first = True
    last_idx = None
    first_wheel = True
    if kwargs["Leg Style"] != "wheeled":
        for i, name in enumerate(names):
            named_attribute = nw.new_node(
                node_type=Nodes.NamedAttribute,
                input_args=[name],
                attrs={"data_type": "INT"},
            )
            for j in range(1, parts[i]+1):
                compare = nw.new_node(
                    node_type=Nodes.Compare,
                    input_kwargs={"A": named_attribute, "B": j},
                    attrs={"data_type": "INT", "operation": "EQUAL"},
                )
                separate_geometry = nw.new_node(
                    node_type=Nodes.SeparateGeometry,
                    input_kwargs={
                        "Geometry": join_geometry.outputs["Geometry"],
                        "Selection": compare.outputs["Result"],
                    },
                )
                
                output_geometry = separate_geometry
                a = save_geometry(
                    nw,
                    output_geometry,
                    kwargs.get("path", None),
                    name,
                    kwargs.get("i", "unknown"),
                    first=first,
                )
                if a:
                    first = False
                    last_idx = a[0]
    else:
        named_attribute_leg = nw.new_node(
                node_type=Nodes.NamedAttribute,
                input_args=["legs_wheel"],
                attrs={"data_type": "INT"},
            )
        for i, name in enumerate(names):
            named_attribute = nw.new_node(
                node_type=Nodes.NamedAttribute,
                input_args=[name],
                attrs={"data_type": "INT"},
            )
            for j in range(1, parts[i]+1):
                compare = nw.new_node(
                    node_type=Nodes.Compare,
                    input_kwargs={"A": named_attribute, "B": j},
                    attrs={"data_type": "INT", "operation": "EQUAL"},
                )
                separate_geometry = nw.new_node(
                    node_type=Nodes.SeparateGeometry,
                    input_kwargs={
                        "Geometry": join_geometry.outputs["Geometry"],
                        "Selection": compare.outputs["Result"],
                    },
                )
                def clear_material(obj):
                    obj.data.materials.clear()
                    return obj
                if i == 1 and j <= kwargs["Leg Pole Number"]:
                    for k in range(1, 5):
                        compare_1 = nw.new_node(
                            node_type=Nodes.Compare,
                            input_kwargs={"A": named_attribute_leg, "B": k},
                            attrs={"data_type": "INT", "operation": "EQUAL"},
                        )
                        separate_geometry_1 = nw.new_node(
                            node_type=Nodes.SeparateGeometry,
                            input_kwargs={
                                "Geometry": separate_geometry,
                                "Selection": compare_1.outputs["Result"],
                            },
                        )
                        output_geometry = separate_geometry_1
                        joint_info = None
                        parent_idx = None
                        if k == 3:
                            parent_idx = last_idx + 2
                            joint_info = {
                                "name": get_joint_name("revolute"),
                                "type": "continuous",
                                "axis": (0, 0, 1)
                            }
                            
                        elif k == 1:
                            origin_shift = (0, 0, 0)
                            parent_idx = last_idx + 2
                            origin_shift = origin_shift[0], origin_shift[2], origin_shift[1]
                            angle = np.pi / kwargs['Leg Pole Number'] + (j - 1) * 2 * np.pi / kwargs['Leg Pole Number']
                            def substitute(obj):
                                if kwargs.get("aux_wheel", None) is not None:
                                    wheel = butil.deep_clone_obj(kwargs['aux_wheel'])
                                    wheel.rotation_euler = (0, 0, angle)
                                    butil.apply_transform(wheel, True)
                                    co = read_co(obj)
                                    scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
                                    wheel.scale = (scale[0], scale[1], scale[2])
                                    butil.apply_transform(wheel, True)
                                    wheel.location = (co[:, 0].max() - scale[0] / 2, co[:, 1].max() - scale[1] / 2, co[:, 2].max() - scale[2] / 2)
                                    butil.apply_transform(wheel, True)
                                    obj = wheel
                                    return obj

                            joint_info = {
                                "name": get_joint_name("continuous"),
                                "type": "continuous",
                                "axis": (np.cos(angle), np.sin(angle), 0),
                                #"origin_shift": origin_shift,
                                #"substitute_mesh_idx": 9 if kwargs['Leg Pole Number'] == 5 else 5,
                                #"origin_shift": (0, -kwargs.get("Leg Wheel Width", 0) / 2, 0)
                            }
                            first_wheel = False
                        elif k == 4:
                            parent_idx = 21 if kwargs['Leg Pole Number'] == 5 else 21 - 4 * (5 - kwargs['Leg Pole Number'])
                            parent_idx += seat_idx
                            joint_info = {
                                "name": get_joint_name("fixed"),
                                "type": "fixed"
                            }
                                
                        else:
                            origin_shift = (0, 0, 0)
                            parent_idx = last_idx + 2
                            origin_shift = origin_shift[0], origin_shift[2], origin_shift[1]
                            joint_info = {
                                "name": get_joint_name("fixed"),
                                "type": "fixed",
                                #"substitute_mesh_idx": 10 if kwargs['Leg Pole Number'] == 5 else 6,
                                #"origin_shift": origin_shift
                            }
                        name_ = name
                        if name == "legs":
                            #name_ = f"{name}_{kwargs["Leg Style"]}_{j}"
                            if k == 1:
                                name_ = f"{name}_wheel"
                            if k == 2:
                                name_ = f"{name}_cap"
                            if k == 3:
                                name_ = f"{name}_spin"
                            if k == 4:
                                name_ = f"{name}_stretch"
                        a = save_geometry(  
                            nw,
                            output_geometry,
                            kwargs.get("path", None),
                            name_,
                            kwargs.get("i", "unknown"),
                            first=first,
                            joint_info=joint_info,
                            parent_obj_id=parent_idx,
                            material=kwargs["LegMaterial"],
                            apply=substitute if k == 1 else clear_material
                        )
                        if a:
                            first = False
                            last_idx = a[0]
                else:
                    material = kwargs["LegMaterial"] if i == 1 else kwargs["TopMaterial"]
                    after_separate = None
                    output_geometry = separate_geometry
                    joint_info = None
                    parent_idx = None
                    name_ = f"leg_{kwargs['Leg Style']}_down"
                    if(i == 1 and j == parts[1]):
                        parent_idx = last_idx
                        joint_info = {
                            "name": get_joint_name("prismatic"),
                            "type": "prismatic",
                            "axis": (0, 0, 1),
                            "limit": {
                                "lower": (kwargs["Leg Joint Height"] - (kwargs['Height'] - kwargs['Top Thickness'])) * 0.8,
                                "upper": 0,
                            },
                        }
                        name_ = f"leg_{kwargs['Leg Style']}_upper"
                    if i == 0:
                        def add_thickness(obj):
                            global co_seat
                            bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', 
                                object=True, 
                                obdata=True, 
                                material=True, 
                                animation=False)
                            #bpy.context.view_layer.objects.link(obj)
                            #bpy.context.collection.objects.link(obj)
                            thickness = kwargs.get("seat_thickness", 0)
                            if thickness <= 0.01:
                                return
                            butil.select_none()
                            obj.select_set(True)
                            bpy.ops.object.modifier_add(type='SOLIDIFY')
                            bpy.context.object.modifiers["Solidify"].thickness = thickness
                            bpy.context.object.modifiers["Solidify"].offset = 0
                            bpy.ops.object.modifier_apply(modifier="Solidify")
                            bpy.ops.object.modifier_add(type='SMOOTH')
                            bpy.context.object.modifiers["Smooth"].factor = 1
                            bpy.context.object.modifiers["Smooth"].iterations = 100
                            bpy.ops.object.modifier_apply(modifier="Smooth")
                            obj.name = "seat"
                            co_seat = read_co(obj)

                        substitute = None
                        if kwargs.get("aux_seat_whole", None) is not None:
                            seat = butil.deep_clone_obj(kwargs['aux_seat_whole'][0])
                            add_thickness = None
                            seat.rotation_euler = (np.pi / 2, 0, 0)
                            butil.apply_transform(seat, True)
                            def substitute(obj):
                                global co_seat
                                co = read_co(obj)
                                co_seat = co
                                scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
                                seat.scale = (scale[0], scale[1], scale[2])
                                butil.apply_transform(seat, True)
                                seat.location = (0, 0, co[:, 2].max() - scale[2] / 2)
                                seat.location[2] *= 0.99
                                butil.apply_transform(seat, True)
                                obj = seat
                                return obj
                        elif kwargs.get("aux_back", None) is not None:
                            back = butil.deep_clone_obj(kwargs['aux_back'][0])
                            seat = butil.deep_clone_obj(kwargs['aux_seat'][0])
                            add_thickness = None
                            seat.rotation_euler = (np.pi / 2, 0, 0)
                            butil.apply_transform(seat, True)
                            back.rotation_euler = (np.pi / 2, 0, 0)
                            butil.apply_transform(back, True)
                            def substitute(obj):
                                global co_seat
                                co = read_co(obj)
                                co_seat = co
                                scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
                                seat.scale = (scale[0], scale[1], scale[2] * uniform(0.05, 0.1))
                                scale_h = seat.scale[2]
                                butil.apply_transform(seat, True)
                                seat.location = (0, 0, co[:, 2].min() * 1.03)
                                butil.apply_transform(seat, True)
                                #co_seat_ = read_co(seat)
                                back.scale = (scale[0], scale[1] * uniform(0.1, 0.2), scale[2] - scale_h)
                                back_h = back.scale[2]
                                butil.apply_transform(back, True)
                                co_s = read_co(seat)
                                back.location = (0, co_s[:, 1].max(), co[:, 2].min() + back_h / 2)
                                butil.apply_transform(back, True)
                                co_b = read_co(back)
                                save_obj_parts_add(back, kwargs.get("path", None), kwargs.get("i", "unknown"), "chair_back", first=True, use_bpy=True, material=kwargs["TopMaterial"], parent_obj_id=1, joint_info={
                                    "name": get_joint_name("revolute"),
                                    "type": "revolute",
                                    "axis": (1, 0, 0),
                                    "origin_shift": (0,  - (co_b[:, 1].max() - co_b[:, 1].min()) / 2,  - (co_b[:, 2].max() - co_b[:, 2].min()) / 2),
                                    "limit":{
                                        "lower": -np.pi / 10,
                                        "upper": 0
                                    }
                                })
                                first = False
                                obj = seat
                                return {"object": obj, "first": first}
                        after_separate = add_thickness
                        name_ = f"seat"

                    a = save_geometry(
                        nw,
                        output_geometry,
                        kwargs.get("path", None),
                        name_,
                        kwargs.get("i", "unknown"),
                        first=first,
                        joint_info=joint_info,
                        parent_obj_id=parent_idx,
                        material=material,
                        after_seperate=after_separate,
                        apply=substitute if i == 0 else None
                    )
                    if a:
                        first = False
                        last_idx = a[0]
    
    add_joint(last_idx, seat_idx, {
        "name": get_joint_name("continuous"),
        "type" :    "continuous",
        "axis": (0, 0, 1),#(0, 0, 1)
        #"origin_shift": (uniform(-0.2, 0.2), uniform(-0.2, 0.2), 0),
        })
    
    # save_geometry(
    #     nw,
    #     join_geometry,
    #     kwargs.get("path", None),
    #     "whole",
    #     kwargs.get("i", "unknown"),
    # )
    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry},
        attrs={"is_active_output": True},
    )


class OfficeChairFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super(OfficeChairFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions

        with FixedSeed(factory_seed):
            self.params, leg_style = self.sample_parameters(dimensions)
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params(leg_style)
            )
        self.params.update(self.material_params)
        #self.chair_fac = ChairFactory(factory_seed, coarse=coarse)
        self.aux_wheel = random_auxiliary("wheels")[0]
        self.use_aux_wheel = choice([True, False], p=[0.95, 0.05])
        #self.use_aux_wheel = False
        if self.use_aux_wheel:
            self.params['aux_wheel'] = self.aux_wheel
        self.adjustable_back = choice([True, False], p=[0, 1])
        self.use_aux_seat = choice([True, False], p=[0.7, 0.3])
        self.use_aux_back_and_seat = choice([True, False], p=[0.5, 0.5])
        self.aux_seat = None
        if self.use_aux_seat:
            self.aux_seat = random_auxiliary("chair_seat_whole")
            self.params['aux_seat_whole'] = self.aux_seat
        elif self.use_aux_back_and_seat:
            self.aux_seat = random_auxiliary("chair_seat")
            self.params['aux_seat'] = self.aux_seat
            self.aux_back = random_auxiliary("chair_back")
            self.params['aux_back'] = self.aux_back
            
        self.use_aux_whole_arm = choice([True, False], p=[0.2, 0.8])
        if self.use_aux_whole_arm:
            self.params['aux_whole_arm'] = random_auxiliary("chair_arm_whole")[0]

        


    def get_material_params(self, leg_style):
        material_assignments = AssetList["OfficeChairFactory"](leg_style)
        params = {
            "TopMaterial": material_assignments["top"].assign_material(),
            "LegMaterial": material_assignments["leg"].assign_material(),
        }
        wrapped_params = {
            k: surface.shaderfunc_to_material(v) for k, v in params.items()
        }
        print(wrapped_params["TopMaterial"].name)

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = uniform() < scratch_prob
        is_edge_wear = uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        # all in meters

        if dimensions is None:
            x = uniform(0.4, 0.8)
            z = uniform(1.0, 1.4)
            dimensions = (x, x, z)

        x, y, z = dimensions

        top_thickness = uniform(0.5, 0.7)

        # straight has the bug that seat and legs are disjoint, so disable for now.

        # leg_style = choice(['straight', 'single_stand', 'wheeled'])
        leg_style = choice(["single_stand", "wheeled"])
        leg_style = "wheeled"

        parameters = {
            "Top Profile Width": x,
            "Top Thickness": top_thickness,
            "Top Front Relative Width": uniform(0.5, 0.8),
            "Top Front Bent": uniform(-1.5, -0.4),
            "Top Seat Bent": uniform(-1.5, -0.4),
            "Top Mid Bent": uniform(-2.4, -0.5),
            "Top Mid Relative Width": uniform(0.5, 0.9),
            "Top Back Bent": uniform(-1, -0.1),
            "Top Back Relative Width": uniform(0.6, 0.9),
            "Top Mid Pos": 0.5,#uniform(0.4, 0.6),
            # 'Top Material': choice(['leather', 'wood', 'plastic', 'glass']),
            "Height": z,
            "Top Height": z - top_thickness,
            "Leg Style": leg_style,
            "Leg NGon": choice([4, 32]),
            "Leg Placement Top Relative Scale": 0.7,
            "Leg Placement Bottom Relative Scale": uniform(1.1, 1.3),
            "Leg Height": 1.0,
        }

        if leg_style == "single_stand":
            leg_number = 1
            leg_diameter = uniform(0.7 * x, 0.9 * x)

            leg_curve_ctrl_pts = [
                (0.0, uniform(0.1, 0.2)),
                (0.5, uniform(0.1, 0.2)),
                (0.9, uniform(0.2, 0.3)),
                (1.0, 1.0),
            ]

            parameters.update(
                {
                    "Leg Number": leg_number,
                    "Leg Diameter": leg_diameter,
                    "Leg Curve Control Points": leg_curve_ctrl_pts,
                    # 'Leg Material': choice(['metal', 'wood'])
                }
            )

        elif leg_style == "straight":
            leg_diameter = uniform(0.04, 0.06)
            leg_number = 4

            leg_curve_ctrl_pts = [
                (0.0, 1.0),
                (0.4, uniform(0.85, 0.95)),
                (1.0, uniform(0.4, 0.6)),
            ]

            parameters.update(
                {
                    "Leg Number": leg_number,
                    "Leg Diameter": leg_diameter,
                    "Leg Curve Control Points": leg_curve_ctrl_pts,
                    # 'Leg Material': choice(['metal', 'wood']),
                    "Strecher Relative Pos": uniform(0.2, 0.6),
                    "Strecher Increament": choice([0, 1, 2]),
                }
            )

        elif leg_style == "wheeled":
            leg_diameter = uniform(0.03, 0.05)
            leg_number = 1
            pole_number = randint(3, 10)
            #pole_number = 2
            joint_height = uniform(0.5, 0.8) * (z - top_thickness)
            wheel_arc_sweep_angle = uniform(120, 240)
            wheel_width = uniform(0.11, 0.15)
            wheel_rot = uniform(0, 360)
            pole_length = uniform(1.6, 2.0)

            parameters.update(
                {
                    "Leg Number": leg_number,
                    "Leg Pole Number": pole_number,
                    "Leg Diameter": leg_diameter,
                    "Leg Joint Height": joint_height,
                    "Leg Wheel Arc Sweep Angle": wheel_arc_sweep_angle,
                    "Leg Wheel Width": wheel_width,
                    "Leg Wheel Rot": wheel_rot,
                    "Leg Pole Length": pole_length,

                    # 'Leg Material': choice(['metal'])
                }
            )

        else:
            raise NotImplementedError

        return parameters, leg_style
    @staticmethod
    def bevel(obj, offset=0.01):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.bevel(
            offset=offset, offset_pct=0, segments=8, release_confirm=True, face_strength_mode="ALL"
        )
        bpy.ops.object.mode_set(mode='OBJECT')

    def create_asset(self, **params):
        global co_seat
        #params.update("wheel_width", self.wheel_width)
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object

        path_dict = {
            "path": params.get("path", None),
            "i": params.get("i", "unknown"),
            "name": ["leg_decors"],
            "seat_thickness": uniform(0, 0.1)
        }
        self.params.update(path_dict)

        surface.add_geomod(
            obj, geometry_assemble_chair, apply=True, input_kwargs=self.params
        )
        tagging.tag_system.relabel_obj(obj)
        has_arm = choice([True, False], p=[0.8, 0.2])
        if self.use_aux_seat and self.aux_seat[1]['need_arm'] == "False":
            has_arm = False
        #has_arm = False
        if self.use_aux_back_and_seat and not self.use_aux_seat:
            seat_idx = 1
        else:
            seat_idx = 0
        if self.use_aux_whole_arm and has_arm:
            has_arm = False
            arm = butil.deep_clone_obj(self.params['aux_whole_arm'])
            arm.rotation_euler = (np.pi / 2, 0, 0)
            butil.apply_transform(arm, True)
            scale = co_seat[:, 0].max() - co_seat[:, 0].min(), co_seat[:, 1].max() - co_seat[:, 1].min(), co_seat[:, 2].max() - co_seat[:, 2].min()
            arm.scale = (scale[0], scale[1]* uniform(0.8, 1), scale[2] * uniform(0.2, 0.5))
            scale_h = arm.scale[2]
            butil.apply_transform(arm, True)
            arm.location = (0, 0, co_seat[:, 2].min() * 1.02 + scale_h / 2)
            butil.apply_transform(arm, True)
            save_obj_parts_add(arm, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=seat_idx, material=self.params["LegMaterial"], joint_info={
                 "name": get_joint_name("fixed"),
                 "type": "fixed"
            })
        if has_arm:
            self.params['Top Back Relative Width'] = max(self.params['Top Back Relative Width'], self.params['Top Mid Relative Width'], self.params['Top Front Relative Width']) * 1.1
            seat = bpy.data.objects['seat']
            co = read_co(seat)
            width = self.params['Top Profile Width'] * self.params['Top Back Relative Width']
            y = (co[:, 1].max() + co[:, 1].min()) / 2
            length = uniform(0.2, 0.35)
            pos = (width / 2, y, self.params['Top Height'])
            arm_l = butil.spawn_cube()
            width = uniform(0.01, 0.02)
            arm_l.scale = (width * 1.5, width * 3, length * 0.7)
            butil.apply_transform(arm_l, True)
            arm_l.location[2] = length / 2
            butil.apply_transform(arm_l, True)
            r_x = random.uniform(-0.174533, 0.174533)
            r_y = random.uniform(0, 0.174533)
            #arm_l.rotation_euler[0] = r_x
            arm_l.rotation_euler[1] = r_y
            butil.apply_transform(arm_l, True)
            arm_l.location = (pos[0] + length * 0.3 * np.sin(r_y), pos[1], pos[2])
            arm_l.location[2] -= 0.03
            butil.apply_transform(arm_l, True)
            self.bevel(arm_l)
            co = read_co(arm_l)
            id = np.argmax(co[:, 2])
            vertex_l = co[id]
            res_l = save_obj_parts_add(arm_l, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=None, material=self.params["LegMaterial"])

            arm_l_ = butil.spawn_cube()
            arm_l_.scale = (width * 1, width * 2, length * 0.3)
            butil.apply_transform(arm_l_, True)
            arm_l_.rotation_euler[1] = r_y
            butil.apply_transform(arm_l_, True)
            arm_l_.location = (pos[0] + width / 2 , pos[1], pos[2])
            butil.apply_transform(arm_l_, True)
            self.bevel(arm_l_)
            res_l_ = save_obj_parts_add(arm_l_, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=seat_idx, joint_info={
                 "name": get_joint_name("fixed"),
                 "type": "fixed"
            }, material=self.params["LegMaterial"])
            add_joint(res_l_[0], res_l[0], joint_info={
                 "name": get_joint_name("prismatic"),
                 "type": "prismatic",
                    "axis": (np.sin(r_y), 0, np.cos(r_y)),
                    "limit": {
                        "lower": -length * 0.15,
                        "upper": 0,
                    },
            })

            arm_r = butil.spawn_cube()
            arm_r.scale = (width * 1.5, width * 3, length*0.7)
            butil.apply_transform(arm_r, True)
            arm_r.location[2] = length / 2
            butil.apply_transform(arm_r, True)
            #arm_r.rotation_euler[0] = r_x
            arm_r.rotation_euler[1] = -r_y
            butil.apply_transform(arm_r, True)
            arm_r.location = (pos[0] - length * 0.3 * np.sin(r_y), pos[1], pos[2])
            arm_r.location[0] *= -1
            arm_r.location[2] -= 0.03
            butil.apply_transform(arm_r, True)
            self.bevel(arm_r)
            co = read_co(arm_r)
            id = np.argmax(co[:, 2])
            vertex_r = co[id]
            res_r = save_obj_parts_add(arm_r, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=None, material=self.params["LegMaterial"])

            arm_r_ = butil.spawn_cube()
            arm_r_.scale = (width * 1, width * 2, length * 0.3)
            butil.apply_transform(arm_r_, True)
            arm_r_.rotation_euler[1] = -r_y
            butil.apply_transform(arm_r_, True)
            arm_r_.location = (-pos[0] - width / 2 , pos[1], pos[2])
            butil.apply_transform(arm_r_, True)
            self.bevel(arm_r_)
            res_r_ = save_obj_parts_add(arm_r_, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=seat_idx, joint_info={
                 "name": get_joint_name("fixed"),
                 "type": "fixed"
            }, material=self.params["LegMaterial"])

            add_joint(res_r_[0], res_r[0], joint_info={
                    "name": get_joint_name("prismatic"),
                    "type": "prismatic",
                        "axis": (-np.sin(r_y), 0, np.cos(r_y)),
                        "limit": {
                            "lower": -length * 0.15,
                            "upper": 0,
                        },
                })
            
            base = butil.spawn_cylinder(self.params['Top Profile Width'] * self.params['Top Back Relative Width'] / 2 * 0.7, 0.02)
            base.location[2] = pos[2]
            base.location[2] -= 0.03
            stretcher = butil.spawn_cube()
            #stretcher.rotation_euler[1] = np.pi / 2
            #butil.apply_transform(stretcher, True)
            stretcher.scale = (self.params['Top Profile Width'] * self.params['Top Back Relative Width'] * 1, width * 2,0.03)
            stretcher.location = base.location
            butil.apply_transform(stretcher, True)
            butil.apply_transform(base, True)
            stretcher_ = butil.spawn_cylinder()
            stretcher_.rotation_euler[1] = np.pi / 2
            butil.apply_transform(stretcher_, True)
            stretcher_.scale = (self.params['Top Profile Width'] * self.params['Top Back Relative Width'] *0.1, 0.02 ,0.02)
            butil.apply_transform(stretcher_, True)
            stretcher_.location[2] = pos[2] - 0.03
            stretcher_.location[0] = self.params['Top Profile Width'] * self.params['Top Back Relative Width'] * 0.45
            butil.apply_transform(stretcher_, True)
            res_s_ = save_obj_parts_add(stretcher_, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=seat_idx, joint_info={
                 "name": get_joint_name("revolute_prismatic"),
                 "type": "revolute_prismatic",
                    "axis": (1, 0, 0),
                    "limit": {
                        "lower": -np.pi / 18,
                        "upper": np.pi / 18,
                        "lower_1": 0,
                        "upper_1": self.params['Top Profile Width'] * self.params['Top Back Relative Width'] * 0.1
                    },

            }, material=self.params["LegMaterial"])
            add_joint(res_s_[0], res_l_[0], joint_info={
                    "name": get_joint_name("fixed"),
                    "type": "fixed",
                })

            stretcher__ = butil.spawn_cylinder()
            stretcher__.rotation_euler[1] = np.pi / 2
            butil.apply_transform(stretcher__, True)
            stretcher__.scale = (self.params['Top Profile Width'] * self.params['Top Back Relative Width'] *0.1, 0.02 ,0.02)
            butil.apply_transform(stretcher__, True)
            stretcher__.location[2] = pos[2] - 0.03
            stretcher__.location[0] = -self.params['Top Profile Width'] * self.params['Top Back Relative Width'] * 0.45
            butil.apply_transform(stretcher__, True)
            res_s__ = save_obj_parts_add(stretcher__, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=seat_idx, joint_info={
                 "name": get_joint_name("revolute_prismatic"),
                 "type": "revolute_prismatic",
                    "axis": (1, 0, 0),
                    "limit": {
                        "lower": -np.pi / 18,
                        "upper": np.pi / 18,
                        "lower_1": -self.params['Top Profile Width'] * self.params['Top Back Relative Width'] * 0.1,
                        "upper_1": 0
                    },

            }, material=self.params["LegMaterial"])
            add_joint(res_s__[0], res_r_[0], joint_info={
                    "name": get_joint_name("fixed"),
                    "type": "fixed",
                })



            co = read_co(arm_l)
            h = co[:, 2].max()
            if r_y > 0:
                w = co[:, 0].max() - width / 2
            else:
                w = co[:, 0].min() - width / 2
            d = (co[:, 1].max() + co[:, 1].min()) / 2

            bpy.ops.object.metaball_add(type='PLANE', radius=0.5, enter_editmode=False, align='WORLD', location=(w, d, h))
            handle = bpy.context.object
            handle.name = "handle"
            bpy.ops.object.convert(target='MESH')
            handle = bpy.data.objects['handle.001']
            handle.scale = (width * 4, width * 15, width * 2)
            butil.apply_transform(handle, True)
            co = read_co(handle)
            center = (co[:, 0].max() + co[:, 0].min()) / 2, (co[:, 1].max() + co[:, 1].min()) / 2, (co[:, 2].max() + co[:, 2].min()) / 2
            save_obj_parts_add(handle, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=res_l[0], joint_info={
                "name": get_joint_name("revolute"),
                "type": "revolute",
                "axis": (1, 0, 0),
                "origin_shift": (vertex_l[0] - center[0], vertex_l[1] - center[1], vertex_l[2] - center[2]),
                "limit": {
                    "lower": -np.pi / 18,
                    "upper": np.pi / 18
                }
            }, material=self.params["LegMaterial"])

            bpy.ops.object.metaball_add(type='PLANE', radius=0.5, enter_editmode=False, align='WORLD', location=(-w, d, h))
            handle_ = bpy.context.object
            handle_.name = "handle_"
            bpy.ops.object.convert(target='MESH')
            handle_ = bpy.data.objects['handle_.001']
            handle_.scale = (width * 4, width * 15, width * 2)
            butil.apply_transform(handle_, True)
            co = read_co(handle_)
            center = (co[:, 0].max() + co[:, 0].min()) / 2, (co[:, 1].max() + co[:, 1].min()) / 2, (co[:, 2].max() + co[:, 2].min()) / 2
            save_obj_parts_add(handle_, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=res_r[0], joint_info={
                "name": get_joint_name("revolute"),
                "type": "revolute",
                "axis": (1, 0, 0),
                "origin_shift": (vertex_r[0] - center[0], vertex_r[1] - center[1], vertex_r[2] - center[2]),
                "limit": {
                    "lower": -np.pi / 18,
                    "upper": np.pi / 18
                }
            }, material=self.params["LegMaterial"])

            #arm_l = join_objects([arm_l, base, stretcher, arm_r])
            #co = read_co(arm_l)
            # for i in range(len(co)):
            #     if co[i, 2] <  self.params['Top Height'] - 0.03:
            #         co[i, 2] = self.params['Top Height'] - 0.03
            #write_co(arm_l, co)
            base = join_objects([base, stretcher])
            self.bevel(base, 0.002)
            save_obj_parts_add(base, params.get("path"), params.get("i"), "chair_arm", first=False, use_bpy=True, parent_obj_id=seat_idx, joint_info={
                 "name": get_joint_name("fixed"),
                 "type": "fixed"
            }, material=self.params["LegMaterial"])
            
            #arm_l = join_objects([arm_l, handle, handle_])
            # save_obj_parts_add(arm_l, params.get("path"), params.get("i"), "arm", first=False, use_bpy=True, parent_obj_id=0, joint_info={
            #     "name": get_joint_name("fixed"),
            #     "type": "fixed"
            # }, material=self.params["LegMaterial"])
        
        join_objects_save_whole(obj, self.params.get("path"), self.params.get("i"), "whole", use_bpy=True, join=False)

        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)
        self.params.update(params)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
        #save_obj_parts_add(assets, self.params.get("path"), self.params.get("i"),  "part", first=False, use_bpy=True, material=[self.params["LegMaterial"]])
        
