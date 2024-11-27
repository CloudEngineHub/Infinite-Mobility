# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo

import bpy
import numpy as np
from numpy.random import choice, uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.seating.chairs.seats.curvy_seats import (
    generate_curvy_seats,
)
from infinigen.assets.objects.tables.cocktail_table import geometry_create_legs
from infinigen.assets.utils.object import save_file_path
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.nodes.node_utils import save_geometry, save_geometry_new
import random

from infinigen.assets.utils.object import (
    join_objects_save_whole,
    save_file_path_obj,
    save_obj_parts_add,
    add_joint,
    get_joint_name
)

def geometry_assemble_chair(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

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
                            if kwargs["Leg Pole Number"] == 3:
                                if j == 1:
                                    origin_shift = (0.03, 0, 0.035)
                                if j == 3:
                                    origin_shift = (-0.03, 0, 0.035)
                            else:
                                if j == 1:
                                    origin_shift = (0.03, 0, 0.3)
                                if j == 2:
                                    origin_shift = (0.03, 0, 0.015)
                                if j == 4:
                                    origin_shift = (-0.03, 0, 0.015)
                                if j == 5:
                                    origin_shift = (-0.03, 0, 0.3)
                            parent_idx = last_idx + 2
                            origin_shift = origin_shift[0], origin_shift[2], origin_shift[1]
                            joint_info = {
                                "name": get_joint_name("continuous"),
                                "type": "continuous",
                                "axis": (1, 0, 0),
                                #"origin_shift": origin_shift,
                                "substitute_mesh_idx": 9 if kwargs['Leg Pole Number'] == 5 else 5,
                                #"origin_shift": (0, -kwargs.get("Leg Wheel Width", 0) / 2, 0)
                            }
                            first_wheel = False
                        elif k == 4:
                            parent_idx = 21 if kwargs['Leg Pole Number'] == 5 else 13
                            joint_info = {
                                "name": get_joint_name("fixed"),
                                "type": "fixed"
                            }
                        else:
                            origin_shift = (0, 0, 0)
                            if kwargs["Leg Pole Number"] == 3:
                                if j == 1:
                                    origin_shift = (0.03, 0, 0.035)
                                if j == 3:
                                    origin_shift = (-0.03, 0, 0.035)
                            else:
                                if j == 1:
                                    origin_shift = (0.03, 0, 0.05)
                                if j == 2:
                                    origin_shift = (0.03, 0, 0.015)
                                if j == 4:
                                    origin_shift = (-0.03, 0, 0.015)
                                if j == 5:
                                    origin_shift = (-0.03, 0, 0.05)
                            parent_idx = last_idx + 2
                            origin_shift = origin_shift[0], origin_shift[2], origin_shift[1]
                            joint_info = {
                                "name": get_joint_name("fixed"),
                                "type": "fixed",
                                "substitute_mesh_idx": 10 if kwargs['Leg Pole Number'] == 5 else 6,
                                #"origin_shift": origin_shift
                            }
                        a = save_geometry(  
                            nw,
                            output_geometry,
                            kwargs.get("path", None),
                            name,
                            kwargs.get("i", "unknown"),
                            first=first,
                            joint_info=joint_info,
                            parent_obj_id=parent_idx,
                            material=kwargs["LegMaterial"]
                        )
                        if a:
                            first = False
                            last_idx = a[0]
                else:
                    output_geometry = separate_geometry
                    joint_info = None
                    parent_idx = None
                    if(i == 1 and j == parts[1]):
                        parent_idx = last_idx
                        joint_info = {
                            "name": get_joint_name("prismatic"),
                            "type": "prismatic",
                            "axis": (0, 0, 1),
                            "limit": {
                                "lower": -0.2,
                                "upper": 0,
                                "lower_1": -0.2,
                                "upper_1": 0
                            },
                            "axis_1": (0, 0, 1),
                        }

                    a = save_geometry(
                        nw,
                        output_geometry,
                        kwargs.get("path", None),
                        name,
                        kwargs.get("i", "unknown"),
                        first=first,
                        joint_info=joint_info,
                        parent_obj_id=parent_idx,
                        material=kwargs["TopMaterial"]
                    )
                    if a:
                        first = False
                        last_idx = a[0]
    
    add_joint(last_idx, 0, {
        "name": get_joint_name("continuous"),
        "type" :    "continuous",
        "axis": (0, 0, 1)})
    save_geometry(
        nw,
        join_geometry,
        kwargs.get("path", None),
        "whole",
        kwargs.get("i", "unknown"),
    )
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
            x = uniform(0.5, 0.6)
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
            "Top Mid Pos": uniform(0.4, 0.6),
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
            pole_number = choice([3, 5])
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

    def create_asset(self, **params):
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
        }
        self.params.update(path_dict)

        surface.add_geomod(
            obj, geometry_assemble_chair, apply=True, input_kwargs=self.params
        )
        tagging.tag_system.relabel_obj(obj)

        obj.rotation_euler.z += np.pi / 2
        butil.apply_transform(obj)

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
        
