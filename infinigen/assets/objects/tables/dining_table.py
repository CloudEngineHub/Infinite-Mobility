# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Yiming Zuo


import bpy
from numpy.random import choice, normal, uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.objects.tables.legs.single_stand import (
    nodegroup_generate_single_stand,
)
from infinigen.assets.objects.tables.legs.square import nodegroup_generate_leg_square
from infinigen.assets.objects.tables.legs.straight import (
    nodegroup_generate_leg_straight,
)
from infinigen.assets.objects.tables.strechers import nodegroup_strecher
from infinigen.assets.objects.tables.table_top import nodegroup_generate_table_top
from infinigen.assets.objects.tables.table_utils import (
    nodegroup_create_anchors,
    nodegroup_create_legs_and_strechers,
)
from infinigen.assets.utils.decorate import read_co, write_co
from infinigen.assets.utils.object import get_joint_name, save_file_path, save_obj_parts_add
from infinigen.core.nodes.node_utils import save_geometry
from infinigen.core import surface, tagging
from infinigen.core import tags as t
from infinigen.core.nodes import node_utils

# from infinigen.assets.materials import metal, metal_shader_list
# from infinigen.assets.materials.fabrics import fabric
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.surface import NoApply
from infinigen.core.util.math import FixedSeed
from infinigen.core.util import blender as butil
from infinigen.assets.utils.auxiliary_parts import random_auxiliary
import numpy as np


@node_utils.to_nodegroup(
    "geometry_create_legs", singleton=False, type="GeometryNodeTree"
)
def geometry_create_legs(nw: NodeWrangler, store_strecher=True, **kwargs):
    createanchors = nw.new_node(
        nodegroup_create_anchors(**{"Profile N-gon": kwargs["Leg Number"]}).name,
        input_kwargs={
            "Profile N-gon": kwargs["Leg Number"],
            "Profile Width": kwargs["Leg Placement Top Relative Scale"]
            * kwargs["Top Profile Width"],
            "Profile Aspect Ratio": kwargs["Top Profile Aspect Ratio"],
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
        if store_strecher:
            strecher = nw.new_node(
                nodegroup_strecher().name,
                input_kwargs={"Profile Width": kwargs["Leg Diameter"] * 0.5},
            )
        else:
            strecher = None

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

    elif kwargs["Leg Style"] == "square":
        leg = nw.new_node(
            nodegroup_generate_leg_square(**kwargs).name,
            input_kwargs={
                "Height": kwargs["Leg Height"],
                "Width": 0.707
                * kwargs["Leg Placement Top Relative Scale"]
                * kwargs["Top Profile Width"]
                * kwargs["Top Profile Aspect Ratio"],
                "Has Bottom Connector": (kwargs["Strecher Increament"] > 0),
                "Profile Width": kwargs["Leg Diameter"],
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

    else:
        raise NotImplementedError

    leg = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": leg, "Material": kwargs["LegMaterial"]},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": leg},
        attrs={"is_active_output": True},
    )

@node_utils.to_nodegroup(
    "geometry_assemble_table", singleton=False, type="GeometryNodeTree"
)
def geometry_assemble_table(nw: NodeWrangler, return_name="", store_strecher=True, **kwargs):
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
            "Geometry": store_table.outputs["Geometry"],
            "Translation": (0.0000, 0.0000, kwargs["Top Height"]),
        },
    )

    tabletop_instance = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": tabletop_instance, "Material": kwargs["TopMaterial"]},
    )

    legs = nw.new_node(geometry_create_legs(store_strecher=store_strecher, **kwargs).name)

    if return_name == "leg":
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
    if return_name == "table_top":
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
    assert True == False

    names = ["table_top", "legs"]
    parts = [1, 2 * kwargs["Leg Number"]]
    
    first = True
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
    #         a = save_geometry(
    #             nw,
    #             output_geometry,
    #             kwargs.get("path", None),
    #             name,
    #             kwargs.get("i", "unknown"),
    #             first=first,
    #         )
    #         if a:
    #             first = False
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


class TableDiningFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None):
        super(TableDiningFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions

        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)

            # self.clothes_scatter = ClothesCover(factory_fn=blanket.BlanketFactory, width=log_uniform(.8, 1.2),
            #                                     size=uniform(.8, 1.2)) if uniform() < .3 else NoApply()
            self.clothes_scatter = NoApply()
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )
            self.use_aux_top = choice([True, False], p=[0.7, 0.3])
            if self.use_aux_top:
                self.aux_top = random_auxiliary('table_top')

        self.params.update(self.material_params)

    def get_material_params(self):
        material_assignments = AssetList["TableDiningFactory"]()
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

        return wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        if dimensions is None:
            width = uniform(0.91, 1.16)

            if uniform() < 0.7:
                # oblong
                length = uniform(1.4, 2.8)
            else:
                # approx square
                length = width * normal(1, 0.1)

            dimensions = (length, width, uniform(0.65, 0.85))

        # all in meters
        x, y, z = dimensions

        NGon = 4

        leg_style = choice(["straight", "single_stand", "square"], p=[0.5, 0.1, 0.4])
        leg_style = choice(['straight', 'square'])

        if leg_style == "single_stand":
            leg_number = 2
            leg_diameter = uniform(0.22 * x, 0.28 * x)

            leg_curve_ctrl_pts = [
                (0.0, uniform(0.1, 0.2)),
                (0.5, uniform(0.1, 0.2)),
                (0.9, uniform(0.2, 0.3)),
                (1.0, 1.0),
            ]

            top_scale = uniform(0.6, 0.7)
            bottom_scale = 1.0

        elif leg_style == "square":
            leg_number = 2
            leg_diameter = uniform(0.07, 0.10)

            leg_curve_ctrl_pts = None

            top_scale = 0.8
            bottom_scale = 1.0

        elif leg_style == "straight":
            leg_diameter = uniform(0.05, 0.07)

            leg_number = 4

            leg_curve_ctrl_pts = [
                (0.0, 1.0),
                (0.4, 1), #uniform(0.85, 0.95)),
                (1.0,1)# uniform(0.4, 0.6)),
            ]

            top_scale = 0.8
            bottom_scale = uniform(1.0, 1.2)
            bottom_scale = 1

        else:
            raise NotImplementedError

        top_thickness = uniform(0.03, 0.06)

        parameters = {
            "Top Profile N-gon": NGon,
            "Top Profile Width": 1.414 * x,
            "Top Profile Aspect Ratio": y / x,
            "Top Profile Fillet Ratio": uniform(0.0, 0.02),
            "Top Thickness": top_thickness,
            "Top Vertical Fillet Ratio": uniform(0.1, 0.3),
            # 'Top Material': choice(['marble', 'tiled_wood', 'metal', 'fabric'], p=[.3, .3, .2, .2]),
            "Height": z,
            "Top Height": z - top_thickness,
            "Leg Number": leg_number,
            "Leg Style": leg_style,
            "Leg NGon": 4,
            "Leg Placement Top Relative Scale": top_scale,
            "Leg Placement Bottom Relative Scale": bottom_scale,
            "Leg Height": 1.0,
            "Leg Diameter": leg_diameter,
            "Leg Curve Control Points": leg_curve_ctrl_pts,
            # 'Leg Material': choice(['metal', 'wood', 'glass', 'plastic']),
            "Strecher Relative Pos": uniform(0.2, 0.6),
            "Strecher Increament": choice([0, 1, 2]),
        }

        return parameters

    def create_asset(self, **params):
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
        }
        obj_params = self.params
        #obj_params.update(path_dict)
        
        # butil.modify_mesh(
        #     obj,
        #     "NODES",
        #     node_group=geometry_assemble_table(**obj_params),
        #     ng_inputs={},
        #     apply=True,
        # )
        obj_params.update(path_dict)
        self.ps = obj_params
        #tagging.tag_system.relabel_obj(obj)
        assert tagging.tagged_face_mask(obj, {t.Subpart.SupportSurface}).sum() != 0

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)

        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        legs = bpy.context.active_object
        butil.modify_mesh(
            legs,
            "NODES",
            node_group=geometry_assemble_table(return_name="leg",**self.ps),
            ng_inputs={},
            apply=True,
        )
        if self.params['Leg Style'] == 'straight':
            co = read_co(legs)
            co[:, 2] *= 0.7
            h = max(co[:, 2])
            write_co(legs, co)
            bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
            )
            legs_ = bpy.context.active_object
            self.params['Leg Diameter'] *= 0.8
            butil.modify_mesh(
                legs_,
                "NODES",
                node_group=geometry_assemble_table(return_name="leg",store_strecher=False,**self.ps),
                ng_inputs={},
                apply=True,
            )
            co = read_co(legs_)
            co[:, 2] *= 0.3
            h_ = max(co[:, 2])
            co[:, 2] += h
            write_co(legs_, co)
            save_obj_parts_add([legs_], self.ps.get("path", None), self.ps.get("i", "unknown"), "leg", first=True, use_bpy=True, parent_obj_id="world", joint_info={
                "name": get_joint_name("prismatic"),
                "type": "prismatic",
                "axis": (0, 0, 1),
                "limit":{
                    "lower": -h_,
                    "upper": 0
                }
            }, material=[self.scratch, self.edge_wear])
        elif self.params['Leg Style'] == 'single_stand':
            co = read_co(legs)
            co[:, 2] *= 0.7
            h = max(co[:, 2]) 
            write_co(legs, co)
            bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
            )
            legs_ = bpy.context.active_object
            self.ps['Leg Diameter'] *= 0.12
            self.ps['Leg Number'] = 2
            self.ps['Leg Curve Control Points'] = [
                (0.0, 1.0),
                (0.4, 1), #uniform(0.85, 0.95)),
                (1.0,1)# uniform(0.4, 0.6)),
            ]
            self.ps['Leg Style'] = 'straight'
            butil.modify_mesh(
                legs_,
                "NODES",
                node_group=geometry_assemble_table(return_name="leg",store_strecher=False,**self.ps),
                ng_inputs={},
                apply=True,
            )
            co = read_co(legs_)
            print(h)
            co[:, 2] *= 0.3
            h_ = max(co[:, 2])
            co[:, 2] += h
            write_co(legs_, co)
            save_obj_parts_add([legs_], self.ps.get("path", None), self.ps.get("i", "unknown"), "leg", first=True, use_bpy=True, parent_obj_id="world", joint_info={
                "name": get_joint_name("prismatic"),
                "type": "prismatic",
                "axis": (0, 0, 1),
                "limit":{
                    "lower": -h_,
                    "upper": 0
                }
            }, material=[self.scratch, self.edge_wear])

        save_obj_parts_add([legs], self.ps.get("path", None), self.ps.get("i", "unknown"), "leg", first=False, use_bpy=True, material=[self.scratch, self.edge_wear])
        bpy.ops.mesh.primitive_plane_add(
            size=2,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        top = bpy.context.active_object
        butil.modify_mesh(
            top,
            "NODES",
            node_group=geometry_assemble_table(return_name="table_top",**self.ps),
            ng_inputs={},
            apply=True,
        )
        parent_id = "world"
        joint_info = {
            "name": get_joint_name("fixed"),
            "type": "fixed",
        }
        if self.params['Leg Style'] == 'straight':
            parent_id = 0
            joint_info = {
                "name": get_joint_name("fixed"),
                "type": "fixed",
            }
        if self.use_aux_top:
            aux_top = self.aux_top[0]
            co_top = read_co(top)
            center = (co_top[:, 0].max() + co_top[:, 0].min()) / 2, (co_top[:, 1].max() + co_top[:, 1].min()) / 2, (co_top[:, 2].max() + co_top[:, 2].min()) / 2
            scale = co_top[:, 0].max() - co_top[:, 0].min(), co_top[:, 1].max() - co_top[:, 1].min(), co_top[:, 2].max() - co_top[:, 2].min()
            aux_top.rotation_euler = (np.pi / 2, 0, np.pi / 2)
            butil.apply_transform(aux_top, True)
            aux_top.scale = scale
            butil.apply_transform(aux_top, True)
            aux_top.location = center
            butil.apply_transform(aux_top, True)
            top = aux_top
        save_obj_parts_add([top], self.ps.get("path", None), self.ps.get("i", "unknown"), "table_top", first=False, use_bpy=True, parent_obj_id=parent_id, joint_info=joint_info, material=[self.scratch, self.edge_wear])
        node_utils.save_geometry_new(assets, "whole", 0, self.ps['i'], self.ps['path'], False, True)

    # def finalize_assets(self, assets):
    #    self.clothes_scatter.apply(assets)


class SideTableFactory(TableDiningFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None):
        if dimensions is None:
            w = 0.55 * normal(1, 0.05)
            h = 0.95 * w * normal(1, 0.05)
            dimensions = (w, w, h)
        super().__init__(factory_seed, coarse=coarse, dimensions=dimensions)


class CoffeeTableFactory(TableDiningFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=None):
        if dimensions is None:
            dimensions = (uniform(1, 1.5), uniform(0.6, 0.9), uniform(0.4, 0.5))
        super().__init__(factory_seed, coarse=coarse, dimensions=dimensions)
