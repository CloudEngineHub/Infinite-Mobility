# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Hongyu Wen

import numpy as np
from numpy.random import normal as N
from numpy.random import randint as RI
from numpy.random import uniform as U

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.decorate import read_co
from infinigen.core import surface
from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_utils import save_geometry, save_geometry_new
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
    join_objects_save_whole,
    save_file_path_obj,
    save_obj_parts_add,
    add_joint,
    get_joint_name
)
import math

from infinigen.assets.utils.auxiliary_parts import random_auxiliary


class BeverageFridgeFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False, dimensions=[1.0, 1.0, 1.0]):
        super(BeverageFridgeFactory, self).__init__(factory_seed, coarse=coarse)

        self.dimensions = dimensions
        with FixedSeed(factory_seed):
            self.params = self.sample_parameters(dimensions)
            self.material_params, self.scratch, self.edge_wear = (
                self.get_material_params()
            )
        self.params.update(self.material_params)
        self.use_aux_divider = np.random.choice([True, False], p=[0.8, 0.2])
        self.aux_divider = random_auxiliary("divider_plate")

    def get_material_params(self):
        material_assignments = AssetList["BeverageFridgeFactory"]()
        params = {
            "Surface": material_assignments["surface"].assign_material(),
            "Front": material_assignments["front"].assign_material(),
            "Handle": material_assignments["handle"].assign_material(),
            "Back": material_assignments["back"].assign_material(),
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

        return wrapped_params, scratch, edge_wear

    @staticmethod
    def sample_parameters(dimensions):
        # depth = max(1 + N(0.3, 0.3), 0.4)
        # width = max(1 + N(0.5, 0.6), 0.3)
        # height = max(1 + N(0.95, 0.95), 0.9)
        # width = max(height / 1.8 + N(0, 0.3), 0.8)
        # # depth, width, height = dimensions
        # door_thickness = U(0.05, 0.1) * depth
        # door_rotation = 0  # Set to 0 for now

        # rack_radius = U(0.01, 0.02) * depth
        # rack_h_amount = RI(2, 4)
        # rack_d_amount = RI(4, 6)
        # brand_name = "BrandName"
        depth = 1 + N(0, 0.1)
        width = 1 + N(0, 0.1)
        height = 1 + N(0, 0.1)
        # depth, width, height = dimensions
        door_thickness = U(0.05, 0.1) * depth
        door_rotation = 0  # Set to 0 for now

        rack_radius = U(0.01, 0.02) * depth
        rack_h_amount = RI(2, 4)
        rack_d_amount = RI(4, 6)
        brand_name = "BrandName"

        params = {
            "Depth": depth,
            "Width": width,
            "Height": height,
            "DoorThickness": door_thickness,
            "DoorRotation": 0,
            "RackRadius": rack_radius,
            "RackHAmount": rack_h_amount,
            "RackDAmount": rack_d_amount,
            "BrandName": brand_name,
        }
        return params

    def create_asset(self, **params):
        obj = butil.spawn_cube()
        params.update({"obj": obj, "inputs": self.params})
        params['scratch'] = self.scratch
        params['edge_wear'] = self.edge_wear
        params['use_aux_divider'] = self.use_aux_divider
        params['aux_divider'] = self.aux_divider
        # params.update({"obj": obj})
        ng = nodegroup_beverage_fridge_geometry(preprocess=True, **params)
        butil.modify_mesh(
            obj, "NODES", node_group=ng, ng_inputs=self.params, apply=True
        )
        bevel_edges = get_bevel_edges(obj)
        delete(obj)
        obj = butil.spawn_cube()
        params.update({"obj": obj, "bevel_edges": bevel_edges})
        ng = nodegroup_beverage_fridge_geometry(preprocess=False, save=True, **params)
        butil.modify_mesh(
            obj, "NODES", node_group=ng, ng_inputs=self.params, apply=True
        )
        obj = add_bevel(obj, bevel_edges, offset=0.01)
        #butil.save_blend(f"test.blend", autopack=True)
        self.ps = params

        return obj

    def finalize_assets(self, assets):
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
        try: 
            door = butil.spawn_cube()
            butil.modify_mesh(
                door,
                "NODES",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=True, return_type_name="door"),
                ng_inputs=self.params,
                apply=True,
            )
            bevel_edges = get_bevel_edges(door)
            delete(door)
            door = butil.spawn_cube()
            butil.modify_mesh(
                door,
                "NODES",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=False, return_type_name="door"),
                ng_inputs=self.params,
                apply=True,
            )
            door = add_bevel(door, bevel_edges, offset=0.01)
        except:
            door = butil.spawn_cube()
            butil.modify_mesh(
                door,
                "Nodes",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=False, return_type_name="door"),
                ng_inputs=self.params,
                apply=True,
            )
        joint_info = {
                            "name": get_joint_name("revolute"),
                            "type": "revolute",
                            "axis": (0, 0, 1),
                            "limit": {
                                "lower": 0,
                                "upper": math.pi / 2
                            },
                            'origin_shift': (0, self.params['Width'] / 2, 0)
                        }
        parent_id = "world"
        save_obj_parts_add([door], self.ps['path'], self.ps['i'], "door", first=False, use_bpy=True, material=[self.scratch, self.edge_wear], joint_info=joint_info, parent_obj_id=parent_id)
        try: 
            body = butil.spawn_cube()
            butil.modify_mesh(
                body,
                "NODES",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=True, return_type_name="body"),
                ng_inputs=self.params,
                apply=True,
            )
            bevel_edges = get_bevel_edges(body)
            delete(body)
            body = butil.spawn_cube()
            butil.modify_mesh(
                body,
                "NODES",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=False, return_type_name="body"),
                ng_inputs=self.params,
                apply=True,
            )
            body = add_bevel(body, bevel_edges, offset=0.01)
        except:
            body = butil.spawn_cube()
            butil.modify_mesh(
                body,
                "NODES",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=False, return_type_name="body"),
                ng_inputs=self.params,
                apply=True,
            )
        save_obj_parts_add([body], self.ps['path'], self.ps['i'], "body", first=False, use_bpy=True, material=[self.scratch, self.edge_wear])
        try:
            heater = butil.spawn_cube()
            butil.modify_mesh(
                heater,
                "NODES",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=True, return_type_name="heater"),
                ng_inputs=self.params,
                apply=True,
            )
            bevel_edges = get_bevel_edges(heater)
            delete(heater)
            heater = butil.spawn_cube()
            butil.modify_mesh(
                heater,
                "NODES",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=False, return_type_name="heater"),
                ng_inputs=self.params,
                apply=True,
            )
            heater = add_bevel(heater, bevel_edges, offset=0.01)
        except:
            heater = butil.spawn_cube()
            butil.modify_mesh(
                heater,
                "NODES",
                node_group=nodegroup_beverage_fridge_geometry(preprocess=False, return_type_name="heater"),
                ng_inputs=self.params,
                apply=True,
            )
        save_obj_parts_add([heater], self.ps['path'], self.ps['i'], "heater", first=False, use_bpy=True, material=[self.scratch, self.edge_wear])
        save_geometry_new(assets, 'whole', 0, self.ps['i'], self.ps['path'], use_bpy=True, first =False)
        #save_obj_parts_add([assets], self.ps['path'], self.ps['i'], "part", first=False, use_bpy=True)

        


@node_utils.to_nodegroup(
    "nodegroup_oven_rack", singleton=False, type="GeometryNodeTree"
)
def nodegroup_oven_rack(nw: NodeWrangler, **kwargs):
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

    geometry_to_instance_quad = nw.new_node(
        Nodes.GeometryToInstance, input_kwargs={"Geometry": quadrilateral}
    )

    store_named_attribute_quad = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": geometry_to_instance_quad,
            "Name": "rack",
            "Value": nw.new_node(
                node_type=Nodes.Math,
                input_kwargs={0: group_input.outputs["Amount"], 1: 2},
                attrs={"operation": "MULTIPLY"},
            ),
        },
        attrs={"domain": "INSTANCE", "data_type": "INT"},
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

    # Add store named attribute here
    store_named_attribute_rack_down = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": duplicate_elements.outputs["Geometry"],
            "Name": "rack",
            "Value": duplicate_elements.outputs["Duplicate Index"],
        },
        attrs={"domain": "INSTANCE", "data_type": "INT"},
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
            "Geometry": store_named_attribute_rack_down.outputs["Geometry"],
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

    # Add store named attribute here
    store_named_attribute_rack_up = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": duplicate_elements_1.outputs["Geometry"],
            "Name": "rack",
            "Value": nw.new_node(
                node_type=Nodes.Math,
                input_kwargs={
                    0: duplicate_elements_1.outputs["Duplicate Index"],
                    1: group_input.outputs["Amount"],
                },
                attrs={"operation": "ADD"},
            ),
        },
        attrs={"domain": "INSTANCE", "data_type": "INT"},
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
            "Geometry": store_named_attribute_rack_up.outputs["Geometry"],
            "Offset": combine_xyz_1,
        },
    )

    join_geometry = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={
            "Geometry": [store_named_attribute_quad, set_position, set_position_1]
        },
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
def nodegroup_text(nw: NodeWrangler, **kwargs):
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

    # Store unique 'Text_id' for Text 1
    store_text_id = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": transform_1,
            "Name": "text",
            "Value": 1,  # Assign an ID of 1
        },
        attrs={"domain": "POINT", "data_type": "INT"},
    )

    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": store_text_id},
        attrs={"is_active_output": True},
    )


@node_utils.to_nodegroup("nodegroup_handle", singleton=False, type="GeometryNodeTree")
def nodegroup_handle(nw: NodeWrangler, **kwargs):
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

    # Store unique 'cube_id' for Cube 1
    store_cube_id = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute,  # Geometry with UV map
            "Name": "handle",
            "Value": 3,  # Assign Cube 1 an ID of 1
        },
        attrs={"domain": "POINT", "data_type": "INT"},
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

    # Store unique 'cube_id' for Cube 1
    store_cube_id_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_1,  # Geometry with UV map
            "Name": "handle",
            "Value": 1,  # Assign Cube 1 an ID of 1
        },
        attrs={"domain": "POINT", "data_type": "INT"},
    )

    combine_xyz = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"Y": group_input.outputs["length"]}
    )

    transform = nw.new_node(
        Nodes.Transform,
        input_kwargs={"Geometry": store_cube_id_1, "Translation": combine_xyz},
    )

    join_geometry_1 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [store_cube_id, transform]},
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

    # Store unique 'cube_id' for Cube 2
    store_cube_id_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_2,  # Geometry with UV map
            "Name": "handle",
            "Value": 2,  # Assign Cube 2 an ID of 2
        },
        attrs={"domain": "POINT", "data_type": "INT"},
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
            "Geometry": store_cube_id_2,
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
def nodegroup_center(nw: NodeWrangler, **kwargs):
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
def nodegroup_cube(nw: NodeWrangler, **kwargs):
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
    "nodegroup_hollow_cube", singleton=False, type="GeometryNodeTree"
)
def nodegroup_hollow_cube(nw: NodeWrangler, **kwargs):
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
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
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

    # Store unique 'cube_id' for Cube 1
    store_cube_id_1 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_1,  # Geometry with UV map
            "Name": "body",
            "Value": 1,
        },
        attrs={"domain": "POINT", "data_type": "INT"},
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
            "Geometry": store_cube_id_1,
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
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
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

    # Store unique 'cube_id' for Cube 4
    store_cube_id_4 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_4,  # Geometry with UV map
            "Name": "body",
            "Value": 4,
        },
        attrs={"domain": "POINT", "data_type": "INT"},
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
            "Geometry": store_cube_id_4,
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
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
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

    # Store unique 'cube_id' for Cube 0
    store_cube_id = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute,  # Geometry with UV map
            "Name": "body",
            "Value": 6,
        },
        attrs={"domain": "POINT", "data_type": "INT"},
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
        input_kwargs={"Geometry": store_cube_id, "Translation": combine_xyz_1},
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
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
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

    # Store unique 'cube_id' for Cube 5
    store_cube_id_5 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_5,  # Geometry with UV map
            "Name": "body",
            "Value": 5,
        },
        attrs={"domain": "POINT", "data_type": "INT"},
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
            "Geometry": store_cube_id_5,
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
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
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

    # Store unique 'cube_id' for Cube 2
    store_cube_id_2 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_2,  # Geometry with UV map
            "Name": "body",
            "Value": 2,
        },
        attrs={"domain": "POINT", "data_type": "INT"},
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
            "Geometry": store_cube_id_2,
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
            "Vertices X": 2,
            "Vertices Y": 2,
            "Vertices Z": 2,
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

    # Store unique 'cube_id' for Cube 3
    store_cube_id_3 = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": store_named_attribute_3,  # Geometry with UV map
            "Name": "body",
            "Value": 3,
        },
        attrs={"domain": "POINT", "data_type": "INT"},
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
            "Geometry": store_cube_id_3,
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


@node_utils.to_nodegroup(
    "nodegroup_beverage_fridge_geometry", singleton=False, type="GeometryNodeTree"
)
def nodegroup_beverage_fridge_geometry(
    nw: NodeWrangler, preprocess: bool = False, return_type_name="", save=False, **kwargs
):
    # Code generated using version 2.6.5 of the node_transpiler
    if "inputs" in kwargs.keys():
        group_input = nw.new_node(
            Nodes.GroupInput,
            expose_input=[
                ("NodeSocketFloat", "Depth", kwargs["inputs"]["Depth"]),
                ("NodeSocketFloat", "Width", kwargs["inputs"]["Width"]),
                ("NodeSocketFloat", "Height", kwargs["inputs"]["Height"]),
                ("NodeSocketFloat", "DoorThickness", kwargs["inputs"]["DoorThickness"]),
                ("NodeSocketFloat", "DoorRotation", kwargs["inputs"]["DoorRotation"]),
                (
                    "NodeSocketFloatDistance",
                    "RackRadius",
                    kwargs["inputs"]["RackRadius"],
                ),
                ("NodeSocketInt", "RackDAmount", kwargs["inputs"]["RackDAmount"]),
                ("NodeSocketInt", "RackHAmount", kwargs["inputs"]["RackHAmount"]),
                ("NodeSocketString", "BrandName", kwargs["inputs"]["BrandName"]),
                ("NodeSocketMaterial", "Surface", None),
                ("NodeSocketMaterial", "Front", None),
                ("NodeSocketMaterial", "Handle", None),
                ("NodeSocketMaterial", "Back", None),
            ],
        )
    else:
        group_input = nw.new_node(
            Nodes.GroupInput,
            expose_input=[
                ("NodeSocketFloat", "Depth", 1.0000),
                ("NodeSocketFloat", "Width", 1.0000),
                ("NodeSocketFloat", "Height", 1.0000),
                ("NodeSocketFloat", "DoorThickness", 0.0700),
                ("NodeSocketFloat", "DoorRotation", 0.0000),
                ("NodeSocketFloatDistance", "RackRadius", 0.0100),
                ("NodeSocketInt", "RackDAmount", 5),
                ("NodeSocketInt", "RackHAmount", 2),
                ("NodeSocketString", "BrandName", "BrandName"),
                ("NodeSocketMaterial", "Surface", None),
                ("NodeSocketMaterial", "Front", None),
                ("NodeSocketMaterial", "Handle", None),
                ("NodeSocketMaterial", "Back", None),
            ],
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
        nodegroup_hollow_cube(**kwargs).name,
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
        nodegroup_cube(**kwargs).name,
        input_kwargs={"Size": combine_xyz_1, "Pos": combine_xyz_2},
    )

    # Store unique 'cube_id' for Cube 1
    store_door = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube,  # Geometry with UV map
            "Name": "door",
            "Value": 1,
        },
        attrs={"domain": "POINT", "data_type": "INT"},
    )

    position = nw.new_node(Nodes.InputPosition)

    center = nw.new_node(
        nodegroup_center(**kwargs).name,
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
            "Geometry": store_door,
            "Selection": center.outputs["In"],
            "Material": group_input.outputs["Front"],
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

    # set_shade_smooth = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': set_material_3})

    multiply = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.8000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_2 = nw.new_node(
        Nodes.Math, input_kwargs={0: multiply}, attrs={"operation": "MULTIPLY"}
    )

    handle = nw.new_node(
        nodegroup_handle(**kwargs).name,
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
        input_kwargs={0: group_input.outputs["Width"], 1: 0.1000},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.9000},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_13 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add, "Y": multiply_3, "Z": multiply_4}
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
            "Material": group_input.outputs["Handle"],
        },
    )

    geometry_to_instance_4 = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": set_material_8}
    )

    rotate_instances_2 = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={
            "Instances": geometry_to_instance_4,
            "Rotation": (-1.5708, 0.0000, 0.0000),
            "Pivot Point": combine_xyz_13,
        },
    )

    add_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    multiply_5 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    combine_xyz_12 = nw.new_node(
        Nodes.CombineXYZ, input_kwargs={"X": add_1, "Y": multiply_5, "Z": 0.0300}
    )

    multiply_6 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: 0.0500},
        attrs={"operation": "MULTIPLY"},
    )

    text = nw.new_node(
        nodegroup_text(**kwargs).name,
        input_kwargs={
            "Translation": combine_xyz_12,
            "String": group_input.outputs["BrandName"],
            "Size": multiply_6,
            "Offset Scale": 0.0020,
        },
    )

    text = complete_no_bevel(nw, text, preprocess)

    set_material_9 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={"Geometry": text, "Material": group_input.outputs["Handle"]},
    )

    rotate_instances_2 = complete_bevel(nw, rotate_instances_2, preprocess)

    join_geometry_3 = nw.new_node(
        Nodes.JoinGeometry,
        input_kwargs={"Geometry": [set_material_3, rotate_instances_2, set_material_9]},
    )

    geometry_to_instance = nw.new_node(
        "GeometryNodeGeometryToInstance", input_kwargs={"Geometry": join_geometry_3}
    )

    z = nw.scalar_multiply(
        group_input.outputs["DoorRotation"], 1 if not preprocess else 0
    )

    combine_xyz_3 = nw.new_node(Nodes.CombineXYZ, input_kwargs={"Z": z})

    combine_xyz_4 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={
            "X": group_input.outputs["Depth"],
            "Y": group_input.outputs["Width"],
        },
    )

    rotate_instances = nw.new_node(
        Nodes.RotateInstances,
        input_kwargs={
            "Instances": geometry_to_instance,
            "Rotation": combine_xyz_3,
            "Pivot Point": combine_xyz_4,
        },
    )

    door_geometry = nw.new_node(Nodes.RealizeInstances, [rotate_instances])
    door = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": door_geometry},
        label="door",
    )

    multiply_7 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.1000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"], 1: multiply_7},
        attrs={"operation": "SUBTRACT"},
    )

    multiply_8 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 3.000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_1 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"], 1: multiply_8},
        attrs={"operation": "SUBTRACT"},
    )

    ovenrack = nw.new_node(
        nodegroup_oven_rack(**kwargs).name,
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

    multiply_9 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Depth"]},
        attrs={"operation": "MULTIPLY"},
    )

    multiply_10 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Width"]},
        attrs={"operation": "MULTIPLY"},
    )

    add_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: duplicate_elements.outputs["Duplicate Index"], 1: 1.0000},
    )

    # Store unique 'cube_id' for Cube 3
    store_rack_up_down = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": duplicate_elements,
            "Name": "rack_up_down",
            "Value": add_2,
        },
        attrs={"domain": "INSTANCE", "data_type": "INT"},
    )

    multiply_11 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["DoorThickness"], 1: 2.0000},
        attrs={"operation": "MULTIPLY"},
    )

    subtract_2 = nw.new_node(
        Nodes.Math,
        input_kwargs={0: group_input.outputs["Height"], 1: multiply_11},
        attrs={"operation": "SUBTRACT"},
    )

    add_3 = nw.new_node(
        Nodes.Math, input_kwargs={0: group_input.outputs["RackHAmount"], 1: 1.0000}
    )

    divide = nw.new_node(
        Nodes.Math,
        input_kwargs={0: subtract_2, 1: add_3},
        attrs={"operation": "DIVIDE"},
    )

    multiply_12 = nw.new_node(
        Nodes.Math, input_kwargs={0: add_2, 1: divide}, attrs={"operation": "MULTIPLY"}
    )

    combine_xyz_5 = nw.new_node(
        Nodes.CombineXYZ,
        input_kwargs={"X": multiply_9, "Y": multiply_10, "Z": multiply_12},
    )

    set_position = nw.new_node(
        Nodes.SetPosition,
        input_kwargs={
            "Geometry": store_rack_up_down.outputs["Geometry"],
            "Offset": combine_xyz_5,
        },
    )

    set_material = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": set_position,
            "Material": group_input.outputs["Handle"],
        },
    )

    input_geometry = nw.new_node(Nodes.RealizeInstances, [set_material])
    racks = nw.new_node(
        Nodes.Reroute,
        input_kwargs={"Input": input_geometry},
        label="racks",
    )

    add_4 = nw.new_node(
        Nodes.Math,
        input_kwargs={
            0: group_input.outputs["Depth"],
            1: group_input.outputs["DoorThickness"],
        },
    )

    reroute_10 = nw.new_node(Nodes.Reroute, input_kwargs={"Input": add_4})

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
        nodegroup_cube(**kwargs).name,
        input_kwargs={"Size": combine_xyz_6, "Pos": combine_xyz_7},
    )

    # Store unique 'cube_id' for Cube 3
    store_heater = nw.new_node(
        Nodes.StoreNamedAttribute,
        input_kwargs={
            "Geometry": cube_1,  # Geometry with UV map
            "Name": "heater",
            "Value": 1,
        },
        attrs={"domain": "POINT", "data_type": "INT"},
    )

    set_material_5 = nw.new_node(
        Nodes.SetMaterial,
        input_kwargs={
            "Geometry": store_heater,
            "Material": group_input.outputs["Back"],
        },
    )

    # set_shade_smooth_1 = nw.new_node(Nodes.SetShadeSmooth, input_kwargs={'Geometry': set_material_5})

    join_geometry_2 = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": set_material_5}
    )

    heater = nw.new_node(
        Nodes.Reroute, input_kwargs={"Input": join_geometry_2}, label="heater"
    )

    if return_type_name == "door":
        door = nw.new_node(
            Nodes.RealizeInstances, [door]
        )
        group_output = nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": door},
            attrs={"is_active_output": True},
        )
        return 
    elif return_type_name == "body":
        body = nw.new_node(
            Nodes.RealizeInstances, [body]
        )
        group_output = nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": body},
            attrs={"is_active_output": True},
        )
        return
    elif return_type_name == "heater":
        heater = nw.new_node(
            Nodes.RealizeInstances, [heater]
        )
        group_output = nw.new_node(
            Nodes.GroupOutput,
            input_kwargs={"Geometry": heater},
            attrs={"is_active_output": True},
        )
        return

    join_geometry = nw.new_node(
        Nodes.JoinGeometry, input_kwargs={"Geometry": [body, door, racks, store_heater]}
    )
    geometry = nw.new_node(Nodes.RealizeInstances, [join_geometry])
    join_geometry = geometry

    names = [
        #"body",
        #"door",
        #"handle",
        "rack_up_down",
        #"heater",
        #"text",
    ]
    rack_split = "rack_up_down"
    parts = [ #6, 
             #1, 
             #3, 
             kwargs['inputs']['RackHAmount'] + 1, 
             #1, 
             #1
             ]
    first_r = True
    ids = []
    if not preprocess and save:
        first = True
        for i, name in enumerate(names):
            named_attribute = nw.new_node(
                node_type=Nodes.NamedAttribute,
                input_args=[name],
                attrs={"data_type": "INT"},
            )
            capture_kwargs = {"Geometry": join_geometry, 1: named_attribute}
            if name == "rack":
                named_attribute_1 = nw.new_node(
                    node_type=Nodes.NamedAttribute,
                    input_args=[rack_split],
                    attrs={"data_type": "INT"},
                )
                capture_kwargs.update({2: named_attribute_1})
            capture_attribute = nw.new_node(
                node_type=Nodes.CaptureAttribute,
                input_kwargs=capture_kwargs,
                attrs={"domain": "POINT"},
            )
            for j in range(1, parts[i] + 1):
                compare = nw.new_node(
                    node_type=Nodes.Compare,
                    input_kwargs={"A": named_attribute, "B": j},
                    attrs={"data_type": "INT", "operation": "EQUAL"},
                )
                separate_geometry = nw.new_node(
                    node_type=Nodes.SeparateGeometry,
                    input_kwargs={
                        "Geometry": capture_attribute.outputs["Geometry"],
                        "Selection": compare.outputs["Result"],
                    },
                )
                if name == "rack_up_down":
                    joint_info = {
                        "name": get_joint_name("prismatic"),
                        "type": "prismatic",
                        "axis": (1, 0, 0),
                        "limit": {
                            "lower": 0,
                            "upper": kwargs['inputs']['Depth'] * 0.75
                        }
                    }
                    parent_id = "world"
                    def substitute(obj, **kwarg):
                        if not kwargs['use_aux_divider']:
                            return
                        co = read_co(obj)
                        sub = butil.deep_clone_obj(kwargs['aux_divider'][0])
                        sub.rotation_euler = (np.pi / 2, 0, np.pi / 2)
                        butil.apply_transform(sub, True)
                        sub.scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
                        butil.apply_transform(sub, False)
                        sub.location = (co[:, 0].min() + co[:, 0].max()) / 2, (co[:, 1].min() + co[:, 1].max()) / 2, (co[:, 2].min() + co[:, 2].max()) / 2
                        butil.apply_transform(sub, True)
                        return sub
                    a = save_geometry(
                        nw,
                        separate_geometry,
                        kwargs.get("path", None),
                        "rack",
                        kwargs.get("i", "unknown"),
                        first=first,
                        joint_info=joint_info,
                        parent_obj_id=parent_id,
                        material=[kwargs['inputs']['Handle'], kwargs['scratch'], kwargs['edge_wear']],
                        apply=substitute
                    )
                    if a:
                        first = False
                    #     if a:
                    #         first = False
                    #         ids.append(a[0])
                    #     if len(ids) == kwargs['inputs']['RackHAmount']:
                    #         first_r = False
                else:
                    if name == 'heater':
                        joint_info = {
                            "name": get_joint_name("fixed"),
                            "type": "fixed",
                        }
                        parent_id = "world"
                        material = [kwargs['inputs']['Back'], kwargs['scratch'], kwargs['edge_wear']]
                    elif name == 'body':
                        joint_info = {
                            "name": get_joint_name("fixed"),
                            "type": "fixed",
                        }
                        parent_id = "world"
                        material = [kwargs['inputs']['Surface'], kwargs['scratch'], kwargs['edge_wear']]
                    elif name == 'door':
                        joint_info = {
                            "name": get_joint_name("revolute"),
                            "type": "revolute",
                            "axis": (0, 0, 1),
                            "limit": {
                                "lower": 0,
                                "upper": math.pi / 2
                            },
                            'origin_shift': (0, kwargs['inputs']['Width'] / 2, 0)
                        }
                        parent_id = "world"
                        material = [kwargs['inputs']['Front'], kwargs['scratch'], kwargs['edge_wear']]
                    elif name == 'text':
                        joint_info = {
                            "name": get_joint_name("fixed"),
                            "type": "fixed",
                        }
                        parent_id = d_p_id
                        material = [kwargs['inputs']['Handle'], kwargs['scratch'], kwargs['edge_wear']]
                    elif name == 'handle':
                        joint_info = {
                            "name": get_joint_name("fixed"),
                            "type": "fixed",
                        }
                        parent_id = d_p_id
                        material = [kwargs['inputs']['Handle'], kwargs['scratch'], kwargs['edge_wear']]
                    output_geometry = separate_geometry
                    a = save_geometry(
                        nw,
                        output_geometry,
                        kwargs.get("path", None),
                        name,
                        kwargs.get("i", "unknown"),
                        first=first,
                        material=material,
                        joint_info=joint_info,
                        parent_obj_id=parent_id,
                    )
                    if a:
                        first = False
                        if name == 'door':
                            d_p_id = a[0]
        # save_geometry(
        #     nw,
        #     join_geometry,
        #     kwargs.get("path", None),
        #     "whole",
        #     kwargs.get("i", "unknown"),
        # )


    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": geometry},
        attrs={"is_active_output": True},  # active output must be here
    )
