# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Yiming Zuo: primary author
# - Alexander Raistrick: implement placeholder

import bpy
import numpy as np
from numpy.random import choice, uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.tables.legs.single_stand import (
    nodegroup_generate_single_stand,
)
from infinigen.assets.objects.tables.legs.straight import (
    nodegroup_generate_leg_straight,
)
from infinigen.assets.objects.tables.legs.wheeled import nodegroup_wheeled_leg, nodegroup_wheeled_leg_nocap
from infinigen.assets.objects.tables.strechers import nodegroup_strecher
from infinigen.assets.objects.tables.table_top import nodegroup_generate_table_top
from infinigen.assets.objects.tables.table_utils import (
    nodegroup_create_anchors,
    nodegroup_create_legs_and_strechers,
)
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.core import surface, tagging
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import NoApply
from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil
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


@node_utils.to_nodegroup(
    "geometry_create_legs", singleton=False, type="GeometryNodeTree"
)
def geometry_create_legs(nw: NodeWrangler, **kwargs):
    createanchors = nw.new_node(
        nodegroup_create_anchors(**{"Profile N-gon": kwargs["Leg Number"]}).name,
        input_kwargs={
            "Profile N-gon": kwargs["Leg Number"],
            "Profile Width": kwargs["Leg Placement Top Relative Scale"]
            * kwargs["Top Profile Width"],
            "Profile Aspect Ratio": 1.0000,
        },
    )

    if kwargs["Leg Style"] == "single_stand":
        leg = nw.new_node(
            nodegroup_generate_single_stand(**kwargs).name,
            input_kwargs={
                "Leg Height": kwargs["Leg Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Resolution": 64,
            },
        )

        leg = nw.new_node(
            nodegroup_create_legs_and_strechers(**{"Profile N-gon": kwargs["Leg Number"]}).name,
            input_kwargs={
                "Anchors": createanchors,
                "Keep Legs": True,
                "Leg Instance": leg,
                "Table Height": kwargs["Top Height"],
                "Leg Bottom Relative Scale": kwargs[
                    "Leg Placement Bottom Relative Scale"
                ],
                "Align Leg X rot": True,
            },
        )

    elif kwargs["Leg Style"] == "straight":
        leg = nw.new_node(
            nodegroup_generate_leg_straight(**kwargs).name,
            input_kwargs={
                "Leg Height": kwargs["Leg Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Resolution": 32,
                "N-gon": kwargs["Leg NGon"],
                "Fillet Ratio": 0.1,
            },
        )

        if kwargs['No Strecher'] == True:
            strecher = None
        else:
            strecher = nw.new_node(
                nodegroup_strecher().name,
                input_kwargs={"Profile Width": kwargs["Leg Diameter"] * 0.5},
            )

        leg = nw.new_node(
            nodegroup_create_legs_and_strechers(**{"Profile N-gon": kwargs["Leg Number"]}).name,
            input_kwargs={
                "Anchors": createanchors,
                "Keep Legs": True,
                "Leg Instance": leg,
                "Table Height": kwargs["Top Height"],
                "Strecher Instance": strecher,
                "Strecher Index Increment": kwargs["Strecher Increament"],
                "Strecher Relative Position": kwargs["Strecher Relative Pos"],
                "Leg Bottom Relative Scale": kwargs[
                    "Leg Placement Bottom Relative Scale"
                ],
                "Align Leg X rot": True,
            },
        )

    elif kwargs["Leg Style"] == "wheeled":
        leg = nw.new_node(
            nodegroup_wheeled_leg_nocap(**kwargs).name,
            input_kwargs={
                "Joint Height": kwargs["Leg Joint Height"],
                "Leg Diameter": kwargs["Leg Diameter"],
                "Top Height": kwargs["Top Height"],
                "Wheel Width": kwargs["Leg Wheel Width"],
                "Wheel Rotation": kwargs["Leg Wheel Rot"],
                "Pole Length": kwargs["Leg Pole Length"],
                "Leg Number": kwargs["Leg Pole Number"],
            },
        )

    else:
        raise NotImplementedError

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": leg},
        attrs={"is_active_output": True},
    )

@node_utils.to_nodegroup(
    "nodegroup_assemble_table", singleton=False, type="GeometryNodeTree"
)
def geometry_assemble_table(nw: NodeWrangler, return_type_name="", **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler

    generatetabletop = nw.new_node(
        nodegroup_generate_table_top().name,
        input_kwargs={
            "Thickness": kwargs["Top Thickness"],
            "N-gon": kwargs["Top Profile N-gon"],
            "Profile Width": kwargs["Top Profile Width"],
            "Aspect Ratio": kwargs["Top Profile Aspect Ratio"],
            "Fillet Ratio": kwargs["Top Profile Fillet Ratio"],
            "Fillet Radius Vertical": kwargs["Top Vertical Fillet Ratio"],
        },
    )

    store_table = nw.new_node(
            Nodes.StoreNamedAttribute,
            input_kwargs={
                "Geometry": generatetabletop,
                "Name": "table_top",
                "Value": 1,  # Assign Cube 1 an ID of 1
            },
            attrs={"domain": "POINT", "data_type": "INT"},
        )

    tabletop_instance = nw.new_node(
        Nodes.Transform,
        input_kwargs={
            "Geometry": generatetabletop,
            "Translation": (0.0000, 0.0000, kwargs["Top Height"]),
        },
    )

    tabletop_instance = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": tabletop_instance, "Material": kwargs["TopMaterial"]},
    )

    legs = nw.new_node(geometry_create_legs(**kwargs).name)

    if return_type_name == "leg":
        legs = nw.new_node(
            Nodes.RealizeInstances,
            [legs]
        )
        group_output = nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": legs},
            attrs={"is_active_output": True},
        )
        return
    if return_type_name == "table_top":
        tabletop_instance = nw.new_node(
            Nodes.RealizeInstances,
            [tabletop_instance]
        )
        group_output = nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": tabletop_instance},
            attrs={"is_active_output": True},
        )
        return

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [tabletop_instance, legs]}
    )

    resample_curve = nw.new_node(
        Nodes.ResampleCurve, input_kwargs={"Curve": generatetabletop.outputs["Curve"]}
    )
    fill_curve = nw.new_node(Nodes.FillCurve, input_kwargs={"Curve": resample_curve})

    voff = kwargs["Top Height"] + kwargs["Top Thickness"]
    extrude_mesh = nw.new_node(
        Nodes.ExtrudeMesh,
        input_kwargs={"Mesh": fill_curve, "Offset Scale": -voff, "Individual": False},
    )
    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [extrude_mesh.outputs["Mesh"], fill_curve]},
    )
    transform_geometry_1 = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": join_geometry_1, "Translation": (0, 0, voff)},
    )
    switch = nw.new_node(
        Nodes.Switch,
        input_kwargs={
            1: kwargs["is_placeholder"],
            14: join_geometry,
            15: transform_geometry_1,
        },
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": switch},
        attrs={"is_active_output": True},
    )


class TableCocktailFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super(TableCocktailFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions

        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)

            # self.clothes_scatter = ClothesCover(factory_fn=blanket.BlanketFactory, width=log_uniform(.8, 1.2),
            #                                     size=uniform(.8, 1.2)) if uniform() < .3 else NoApply()
            self.clothes_scatter = NoApply()
            self.ms, self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )

        self.params.update(self.material_params)

    def get_material_params(self):
        material_assignments = AssetList["TableCocktailFactory"]()
        params = {
            "TopMaterial": material_assignments["top"].assign_material(),
            "LegMaterial": material_assignments["leg"].assign_material(),
        }
        wrapped_params = {
            k: surface.shaderfunc_to_material(v) for k, v in params.items()
        }

        scratch_prob, edge_wear_prob = material_assignments["wear_tear_prob"]
        scratch, edge_wear = material_assignments["wear_tear"]

        is_scratch = uniform() < scratch_prob
        is_edge_wear = uniform() < edge_wear_prob
        if not is_scratch:
            scratch = None

        if not is_edge_wear:
            edge_wear = None

        return params, wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        # all in meters
        if dimensions is None:
            x = uniform(0.5, 0.8)
            z = uniform(1.0, 1.5)
            dimensions = (x, x, z)

        x, y, z = dimensions

        NGon = choice([4, 32])
        if NGon >= 32:
            round_table = True
        else:
            round_table = False

        leg_style = choice(["straight", "single_stand"])
        if leg_style == "single_stand":
            leg_number = 1
            leg_diameter = uniform(0.7 * x, 0.9 * x)

            leg_curve_ctrl_pts = [
                (0.0, uniform(0.1, 0.2)),
                (0.5, uniform(0.1, 0.2)),
                (0.9, uniform(0.2, 0.3)),
                (1.0, 1.0),
            ]

        elif leg_style == "straight":
            leg_diameter = uniform(0.05, 0.07)

            if round_table:
                leg_number = choice([3, 4])
            else:
                leg_number = NGon

            leg_curve_ctrl_pts = [
                (0.0, 1.0),
                (0.4, 1),#uniform(0.85, 0.95)),
                (1.0, 1)#uniform(0.4, 0.6)),
            ]

        else:
            raise NotImplementedError

        top_thickness = uniform(0.02, 0.05)

        parameters = {
            "Top Profile N-gon": 32 if round_table else 4,
            "Top Profile Width": x if round_table else 1.414 * x,
            "Top Profile Aspect Ratio": 1.0,
            "Top Profile Fillet Ratio": 0.499 if round_table else uniform(0.0, 0.05),
            "Top Thickness": top_thickness,
            "Top Vertical Fillet Ratio": uniform(0.1, 0.3),
            'Top Material': choice(['marble', 'tiled_wood', 'plastic', 'glass']),
            "Height": z,
            "Top Height": z - top_thickness,
            "Leg Number": leg_number,
            "Leg Style": leg_style,
            "Leg NGon": choice([4, 32]),
            "Leg Placement Top Relative Scale": 0.7,
            "Leg Placement Bottom Relative Scale": 1,#uniform(1.1, 1.3),
            "Leg Height": 1.0,
            "Leg Diameter": leg_diameter,
            "Leg Curve Control Points": leg_curve_ctrl_pts,
            'Leg Material': choice(['metal', 'wood', 'glass']),
            "Strecher Relative Pos": uniform(0.2, 0.6),
            "Strecher Increament": choice([0, 1, 2]),
            'No Strecher': False,
        }

        return parameters

    def _execute_geonodes(self, is_placeholder, return_type_name="", **kwargs):
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        obj = bpy.context.active_object
        params = kwargs
        self.ps = params

        kwargs = {**self.params, "is_placeholder": is_placeholder}
        # surface.add_geomod(
        #     obj, geometry_assemble_table(return_type_name=return_type_name), apply=True, input_kwargs=kwargs
        # )
        butil.modify_mesh(
            obj, 
            "NODES",
            node_group=geometry_assemble_table(**kwargs),
            apply=True,
            ng_inputs={},
        )
        tagging.tag_system.relabel_obj(obj)

        return obj

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return self._execute_geonodes(is_placeholder=True, **kwargs)

    def create_asset(self, **args):
        return self._execute_geonodes(is_placeholder=False, **args)

    def finalize_assets(self, assets):
        self.clothes_scatter.apply(assets)
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
        leg = bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        leg = bpy.context.active_object
        kwargs = {**self.params, "is_placeholder": False}
        # surface.add_geomod(
        #     obj, geometry_assemble_table(return_type_name=return_type_name), apply=True, input_kwargs=kwargs
        # )
        butil.modify_mesh(
            leg, 
            "NODES",
            node_group=geometry_assemble_table(**kwargs, return_type_name="leg"),
            apply=True,
            ng_inputs={},
        )
        tagging.tag_system.relabel_obj(leg)
        if self.params['Leg Style'] == "single_stand":
            co = read_co(leg)
            h = max(co[:, 2])
            co[:, 2] *= 0.7
            write_co(leg, co)
            leg_ = bpy.ops.mesh.primitive_cylinder_add(
                radius=0.015,
                depth=h * 0.3,
            )
            leg_ = bpy.context.active_object
            leg_.location = (0, 0, h * 0.85)
            butil.apply_transform(leg_, True)
            self.clothes_scatter.apply(leg_)
            parent_id = 1
            joint_info = {
                "name": get_joint_name("revolute_prismatic"),
                "type": "revolute_prismatic",
                "axis": (0, 0, 1),
                "axis_1": (0, 0, 1),
                "limit": {
                    "lower": -np.pi,
                    "upper": np.pi,
                    "lower_1": -h * 0.3,
                    "upper_1": 0,
                },
            }
            save_obj_parts_add([leg_], self.ps['path'], self.ps['i'], "leg", first=True, use_bpy=True, material=[self.ms['LegMaterial'],self.scratch, self.edge_wear, self.clothes_scatter], parent_obj_id=parent_id, joint_info=joint_info)
        elif self.params['Leg Style'] == "straight":
            co = read_co(leg)
            h = max(co[:, 2])
            co[:, 2] *= 0.7
            write_co(leg, co)
            leg_ = butil
            bpy.ops.mesh.primitive_plane_add(
                size=2,
                enter_editmode=False,
                align="WORLD",
                location=(0, 0, 0),
                scale=(1, 1, 1),
            )
            leg_ = bpy.context.active_object
            self.params['Leg Height'] = h * 0.3
            self.params['Leg Diameter'] *= 0.9
            self.params['No Strecher'] = True
            kwargs = {**self.params, "is_placeholder": False}
            butil.modify_mesh(
                leg_, 
                "NODES",
                node_group=geometry_assemble_table(**kwargs, return_type_name="leg"),
                apply=True,
                ng_inputs={},
            )
            tagging.tag_system.relabel_obj(leg)
            parent_id = 1
            joint_info = {
                "name": get_joint_name("prismatic"),
                "type": "prismatic",
                "axis": (0, 0, 1),
                "limit": {
                    "lower": -h * 0.3,
                    "upper": 0,
                },
            }
            save_obj_parts_add([leg_], self.ps['path'], self.ps['i'], "leg", first=True, use_bpy=True, material=[self.ms['LegMaterial'],self.scratch, self.edge_wear, self.clothes_scatter], parent_obj_id=parent_id, joint_info=joint_info)
        save_obj_parts_add([leg], self.ps['path'], self.ps['i'], "leg", first=False, use_bpy=True, material=[self.ms['LegMaterial'],self.scratch, self.edge_wear, self.clothes_scatter])
        top = bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        top = bpy.context.active_object
        kwargs = {**self.params, "is_placeholder": False}
        # surface.add_geomod(
        #     obj, geometry_assemble_table(return_type_name=return_type_name), apply=True, input_kwargs=kwargs
        # )
        butil.modify_mesh(
            top, 
            "NODES",
            node_group=geometry_assemble_table(**kwargs, return_type_name="table_top"),
            apply=True,
            ng_inputs={},
        )
        tagging.tag_system.relabel_obj(top)
        parent_id = 0
        joint_info  = {
            "type": "fixed",
            "name": get_joint_name("fixed"),
        }
        save_obj_parts_add([top], self.ps['path'], self.ps['i'], "leg", first=False, use_bpy=True, parent_obj_id=parent_id, joint_info=joint_info, material=[self.ms['TopMaterial'],self.scratch, self.edge_wear, self.clothes_scatter])
        node_utils.save_geometry_new(assets, 'whole',0, self.ps['i'], self.ps['path'], first=False, use_bpy=True)



