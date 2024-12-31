# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Beining Han

import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.materials.shelf_shaders import (
    shader_shelves_black_wood,
    shader_shelves_black_wood_sampler,
    shader_shelves_white,
    shader_shelves_white_sampler,
    shader_shelves_wood,
    shader_shelves_wood_sampler,
)
from infinigen.assets.objects.shelves.doors import CabinetDoorBaseFactory
from infinigen.assets.objects.shelves.drawers import CabinetDrawerBaseFactory
from infinigen.assets.objects.shelves.large_shelf import LargeShelfBaseFactory
from infinigen.assets.utils.decorate import read_co
from infinigen.assets.utils.object import new_bbox
from infinigen.core import surface, tagging
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.assets.utils.object import (
    join_objects,
    new_bbox,
    new_cube,
    new_plane,
    save_objects,
    save_parts_join_objects,
    save_obj_parts_add,
    get_joint_name,
    join_objects_save_whole
)
from infinigen.assets.objects.elements.doors import LiteDoorFactory

from infinigen.assets.materials.woods.wood import apply

from infinigen.assets.utils.auxiliary_parts import random_auxiliary
import numpy.random as npr
import random

target = -1
left = False
door = True
done = True
drawer_now = 0
def geometry_nodes(nw: NodeWrangler, **kwargs):
    # Code generated using version 2.6.4 of the node_transpiler
    global target, left, door, done, drawer_now
    cabinets = []
    for i, component in enumerate(kwargs["components"]):
        frame_info = nw.new_node(
            Nodes.ObjectInfo, input_kwargs={"Object": component[0]}
        )

        attachments = []
        # print(i, component[1])
        if component[1] == "door":
            right_door_info = nw.new_node(
                Nodes.ObjectInfo, input_kwargs={"Object": component[2][0]}
            )
            left_door_info = nw.new_node(
                Nodes.ObjectInfo, input_kwargs={"Object": component[2][1]}
            )

            transform_r = nw.new_node(
                Nodes.Transform,
                input_kwargs={
                    "Geometry": right_door_info.outputs["Geometry"],
                    "Translation": component[2][2]["door_hinge_pos"][0],
                    "Rotation": (0, 0, component[2][2]["door_open_angle"]),
                },
            )
            attachments.append(transform_r)
            if i == target and done:
                left = True
                door = True
                transform_r = nw.new_node(
                        Nodes.Transform,
                        input_kwargs={
                            "Geometry": transform_r,
                            "Translation": (0, kwargs["y_translations"][i], 0),
                        },
                    )
                group_output = nw.new_node(
                    Nodes.GroupOutput,
                    input_kwargs={"Geometry": transform_r},
                    attrs={"is_active_output": True},
                )       
                if len(component[2][2]["door_hinge_pos"]) > 1:
                    done =False
                else:
                    target += 1
                return 
            if len(component[2][2]["door_hinge_pos"]) > 1:
                transform_l = nw.new_node(
                    Nodes.Transform,
                    input_kwargs={
                        "Geometry": left_door_info.outputs["Geometry"],
                        "Translation": component[2][2]["door_hinge_pos"][1],
                        "Rotation": (0, 0, component[2][2]["door_open_angle"]),
                    },
                )
                transform_l = nw.new_node(
                        Nodes.Transform,
                        input_kwargs={
                            "Geometry": transform_l,
                            "Translation": (0, kwargs["y_translations"][i], 0),
                        },
                    )
                attachments.append(transform_l)
                if i == target and not done:
                    left = False
                    door = True
                    group_output = nw.new_node(
                        Nodes.GroupOutput,
                        input_kwargs={"Geometry": transform_l},
                        attrs={"is_active_output": True},
                    )       
                    done = True
                    target += 1
                    return 
        elif component[1] == "drawer":
            # print(i, target)
            # input()
            for j, drawer in enumerate(component[2]):
                drawer_info = nw.new_node(
                    Nodes.ObjectInfo, input_kwargs={"Object": drawer[0]}
                )
                transform = nw.new_node(
                    Nodes.Transform,
                    input_kwargs={
                        "Geometry": drawer_info.outputs["Geometry"],
                        "Translation": drawer[1]["drawer_hinge_pos"],
                    },
                )
                attachments.append(transform)
                if i == target and j == drawer_now:
                    left = False
                    door = False
                    transform = nw.new_node(
                        Nodes.Transform,
                        input_kwargs={
                            "Geometry": transform,
                            "Translation": (0, kwargs["y_translations"][i], 0),
                        },
                    )
                    transform = nw.new_node(
                        Nodes.SetMaterial,
                        input_kwargs={"Geometry": transform,
                                      "Material": surface.shaderfunc_to_material(kwargs["material_params"]["frame_material"])},
                    )
                    group_output = nw.new_node(
                        Nodes.GroupOutput,
                        input_kwargs={"Geometry": transform},
                        attrs={"is_active_output": True},
                    )       
                    done = False
                    drawer_now += 1
                    if drawer_now == len(component[2]):
                        target += 1
                        done = True
                        drawer_now = 0
                    return 
        else:
            if i == target:
                left = False
                door = False
                target += 1
            continue

        join_geometry = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": attachments}
        )
        # [frame_info.outputs['Geometry']]})

        transform = nw.new_node(
            Nodes.Transform,
            input_kwargs={
                "Geometry": join_geometry,
                "Translation": (0, kwargs["y_translations"][i], 0),
            },
        )
        cabinets.append(transform)

    try:
        join_geometry_1 = nw.new_node(
            Nodes.JoinGeometry, input_kwargs={"Geometry": cabinets}
        )
    except TypeError:
        import pdb

        pdb.set_trace()
    join_geometry_1  = nw.new_node(
        Nodes.RealizeInstances, [join_geometry_1]
    )
    target = -1
    group_output = nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={"Geometry": join_geometry_1},
        attrs={"is_active_output": True},
    )


class KitchenCabinetBaseFactory(AssetFactory):
    def __init__(self, factory_seed, params={}, coarse=False):
        super(KitchenCabinetBaseFactory, self).__init__(factory_seed, coarse=coarse)
        self.frame_params = {}
        self.material_params = {}
        self.cabinet_widths = []
        self.frame_fac = LargeShelfBaseFactory(factory_seed)
        self.door_fac = CabinetDoorBaseFactory(factory_seed)
        self.drawer_fac = CabinetDrawerBaseFactory(factory_seed)
        self.drawer_only = False
        self.aux_drawer_handle = random_auxiliary("handles")[0]
        self.aux_door_handle = random_auxiliary("handles")[0]
        #self.use_auxiliary = random.choice([True, False], weight=[0.9, 0.1])
        self.use_auxiliary = npr.choice([True, False], p=[0.9, 0.1])
        self.use_auxiliary = True
        self.aux_door_fac = LiteDoorFactory(factory_seed)
        self.aux_door = self.aux_door_fac.create_asset(return_panel=True)
        self.aux_drawer = random_auxiliary("drawers")
        self.use_auxiliary_drawer = npr.choice([True, False], p=[0.3, 0.7])
        #self.use_auxiliary_drawer = True
        #self.use_auxiliary = True
        with FixedSeed(factory_seed):
            self.params = self.sample_params()

    def sample_params(self):
        pass

    def get_material_params(self):
        with FixedSeed(self.factory_seed):
            params = self.material_params.copy()
            if params.get("frame_material", None) is None:
                with FixedSeed(self.factory_seed):
                    params["frame_material"] = np.random.choice(
                        ["white", "black_wood", "wood"], p=[0.4, 0.3, 0.3]
                    )
            params["board_material"] = params["frame_material"]
            return self.get_material_func(params, randomness=True)

    def get_material_func(self, params, randomness=True):
        with FixedSeed(self.factory_seed):
            white_wood_params = shader_shelves_white_sampler()
            black_wood_params = shader_shelves_black_wood_sampler()
            normal_wood_params = shader_shelves_wood_sampler()
            if params["frame_material"] == "white":
                if randomness:
                    params["frame_material"] = lambda x: shader_shelves_white(
                        x, **white_wood_params
                    )
                else:
                    params["frame_material"] = shader_shelves_white
            elif params["frame_material"] == "black_wood":
                if randomness:
                    params["frame_material"] = lambda x: shader_shelves_black_wood(
                        x, **black_wood_params, z_axis_texture=True
                    )
                else:
                    params["frame_material"] = lambda x: shader_shelves_black_wood(
                        x, z_axis_texture=True
                    )
            elif params["frame_material"] == "wood":
                if randomness:
                    params["frame_material"] = lambda x: shader_shelves_wood(
                        x, **normal_wood_params, z_axis_texture=True
                    )
                else:
                    params["frame_material"] = lambda x: shader_shelves_wood(
                        x, z_axis_texture=True
                    )

            if params["board_material"] == "white":
                if randomness:
                    params["board_material"] = lambda x: shader_shelves_white(
                        x, **white_wood_params
                    )
                else:
                    params["board_material"] = shader_shelves_white
            elif params["board_material"] == "black_wood":
                if randomness:
                    params["board_material"] = lambda x: shader_shelves_black_wood(
                        x, **black_wood_params
                    )
                else:
                    params["board_material"] = shader_shelves_black_wood
            elif params["board_material"] == "wood":
                if randomness:
                    params["board_material"] = lambda x: shader_shelves_wood(
                        x, **normal_wood_params
                    )
                else:
                    params["board_material"] = shader_shelves_wood

            params["panel_meterial"] = params["frame_material"]
            params["knob_material"] = params["frame_material"]
            return params

    def get_frame_params(self, width, i=0):
        params = self.frame_params.copy()
        params["shelf_cell_width"] = [width]
        params.update(self.material_params.copy())
        return params

    def get_attach_params(self, attach_type, i=0):
        param_sets = []
        if attach_type == "none":
            pass
        elif attach_type == "door":
            params = dict()
            shelf_width = (
                self.frame_params["shelf_width"]
                + self.frame_params["side_board_thickness"] * 2
            )
            if shelf_width <= 0.6:
                params["door_width"] = shelf_width
                params["has_mid_ramp"] = False
                params["edge_thickness_1"] = 0.01
                params["door_hinge_pos"] = [
                    (
                        self.frame_params["shelf_depth"] / 2.0 + 0.0025,
                        -shelf_width / 2.0,
                        self.frame_params["bottom_board_height"],
                    )
                ]
                params["door_open_angle"] = 0
            else:
                params["door_width"] = shelf_width / 2.0 - 0.0005
                params["has_mid_ramp"] = False
                params["edge_thickness_1"] = 0.01
                params["door_hinge_pos"] = [
                    (
                        self.frame_params["shelf_depth"] / 2.0 + 0.008,
                        -shelf_width / 2.0,
                        self.frame_params["bottom_board_height"],
                    ),
                    (
                        self.frame_params["shelf_depth"] / 2.0 + 0.008,
                        shelf_width / 2.0,
                        self.frame_params["bottom_board_height"],
                    ),
                ]
                params["door_open_angle"] = 0

            params["door_height"] = (
                self.frame_params["division_board_z_translation"][-1]
                - self.frame_params["division_board_z_translation"][0]
                + self.frame_params["division_board_thickness"]
            )
            params.update(self.material_params.copy())
            param_sets.append(params)
        elif attach_type == "drawer":
            for i, h in enumerate(self.frame_params["shelf_cell_height"]):
                params = dict()
                drawer_h = (
                    self.frame_params["division_board_z_translation"][i + 1]
                    - self.frame_params["division_board_z_translation"][i]
                    - self.frame_params["division_board_thickness"]
                )
                drawer_depth = self.frame_params["shelf_depth"]
                params["drawer_board_width"] = self.frame_params["shelf_width"]
                params["drawer_board_height"] = drawer_h
                params["drawer_depth"] = drawer_depth
                params["drawer_hinge_pos"] = [
                    self.frame_params["shelf_depth"] / 2.0,
                    0,
                    (
                        self.frame_params["division_board_thickness"] / 2.0
                        + self.frame_params["division_board_z_translation"][i]
                    ),
                ]
                #params["drawer_hinge_pos"][2] *= 1.1
                params.update(self.material_params.copy())
                param_sets.append(params)
        else:
            raise NotImplementedError

        return param_sets

    def get_cabinet_params(self, i=0):
        x_translations = []
        accum_w, thickness = (
            0,
            self.frame_params.get("side_board_thickness", 0.005),
        )  # instructed by Beining
        for w in self.cabinet_widths:
            accum_w += thickness + w / 2.0
            x_translations.append(accum_w)
            accum_w += thickness + w / 2.0 + 0.0005
        return x_translations

    def create_cabinet_components(self, i, drawer_only=False, path=None):
        # update material params
        self.material_params = self.get_material_params()

        components = []
        for k, w in enumerate(self.cabinet_widths):
            # create frame
            frame_params = self.get_frame_params(w, i=i)
            self.frame_fac.params = frame_params
            frame, frame_params = self.frame_fac.create_asset(i=i, ret_params=True)
            frame.name = f"cabinet_frame_{k}"
            self.frame_params = frame_params

            # create attach
            if drawer_only:
                attach_type = np.random.choice(["drawer", "door"], p=[0.5, 0.5])
            else:
                attach_type = np.random.choice(
                    ["drawer", "door", "none"], p=[0.4, 0.4, 0.2]
                )

            attach_params = self.get_attach_params(attach_type, i=i)
            if attach_type == "door":
                self.door_fac.params = attach_params[0]
                self.door_fac.params["door_left_hinge"] = False
                right_door, door_obj_params = self.door_fac.create_asset(
                    i=i, ret_params=True, auxiliary_knob=butil.deep_clone_obj(self.aux_door_handle), aux_door=butil.deep_clone_obj(self.aux_door)
                )
                right_door.name = f"cabinet_right_door_{k}"
                self.door_fac.params = door_obj_params
                self.door_fac.params["door_left_hinge"] = True
                part = butil.deep_clone_obj(self.aux_door_handle)
                door = butil.deep_clone_obj(self.aux_door)

                left_door, _ = self.door_fac.create_asset(i=i, ret_params=True, auxiliary_knob=part, aux_door=door)
                left_door.name = f"cabinet_left_door_{k}"
                components.append(
                    [frame, "door", [right_door, left_door, attach_params[0]]]
                )

            elif attach_type == "drawer":
                drawers = []
                for j, p in enumerate(attach_params):
                    self.drawer_fac.params = p
                    if self.use_auxiliary:
                        if self.use_auxiliary_drawer:
                            drawer = self.drawer_fac.create_asset(i=i, auxiliary_knob=butil.deep_clone_obj(self.aux_drawer_handle), path=path, auxiliary_drawer=[butil.deep_clone_obj(self.aux_drawer[0]), self.aux_drawer[1]])
                        else:
                            drawer = self.drawer_fac.create_asset(i=i, auxiliary_knob=butil.deep_clone_obj(self.aux_drawer_handle), path=path)
                    else:
                        drawer = self.drawer_fac.create_asset(i=i)
                    drawer.name = f"drawer_{k}_layer{j}"
                    drawers.append([drawer, p])
                components.append([frame, "drawer", drawers])

            elif attach_type == "none":
                components.append([frame, "none"])

            else:
                raise NotImplementedError

        return components

    def create_asset(self, i=0, **params):
        idx_ = i
        components = self.create_cabinet_components(i=i, drawer_only=self.drawer_only, path=params.get("path", None))
        cabinet_params = self.get_cabinet_params(i=i)
        join_objs = []

        contain_attach = False
        for com in components:
            if com[1] == "none":
                continue
            else:
                contain_attach = True

        if contain_attach:
            bpy.ops.mesh.primitive_plane_add(
                size=1,
                enter_editmode=False,
                align="WORLD",
                location=(0, 0, 0),
                scale=(1, 1, 1),
            )
            obj = bpy.context.active_object
            surface.add_geomod(
                obj,
                geometry_nodes,
                attributes=[],
                input_kwargs={
                    "components": components,
                    "y_translations": cabinet_params,
                    "material_params": self.material_params,
                },
                apply=True,
            )

            join_objs += [obj]
        global target
        target = 0
        first = True
        for i in range(10 * len(components)):
            bpy.ops.mesh.primitive_plane_add(
                size=1,
                enter_editmode=False,
                align="WORLD",
                location=(0, 0, 0),
                scale=(1, 1, 1),
            )
            obj = bpy.context.active_object
            surface.add_geomod(
                obj,
                geometry_nodes,
                attributes=[],
                input_kwargs={
                    "components": components,
                    "y_translations": cabinet_params,
                    "material_params": self.material_params,
                },
                apply=True,
            )
            if target == -1:
                break
            global door, left, done
            parent_id = "world"
            joint_info = {
                "name": get_joint_name("fixed"),
                "type": "fixed",
            }
            if not door:
                parent_id = "world"
                joint_info = {
                    "name": get_joint_name("prismatic"),
                    "type": "prismatic",
                    "axis": (1, 0, 0),
                    "limit":{
                        "lower": 0,
                        "upper": self.dimensions[0] * 0.75
                    }
                }
                #butil.save_blend("scene.blend", autopack=True)
                #input()
            else:
                co = read_co(obj)
                width = co[:, 1].max() - co[:, 1].min()
                joint_info = {
                    "name": get_joint_name("revolute"),
                    "type": "revolute",
                    "axis": (0, 0, 1),
                    "limit":{
                        "lower": -np.pi / 2 if left else 0,
                        "upper": 0 if left else np.pi / 2
                    },
                    "origin_shift": [0, -width / 2 if left else width / 2, 0]
                }
            # while(len(obj.data.materials) > 0):
            #     obj.data.materials.pop()
            obj_ = butil.deep_clone_obj(obj, keep_materials=False, keep_modifiers=False)
            #bpy.context.collection.objects.link(obj_)
            apply(obj_, shader=self.material_params["frame_material"], selection="frame")
            tagging.tag_system.relabel_obj(obj_)
            #obj_.data.materials.append(surface.shaderfunc_to_material(self.material_params["frame_material"]))
            save_obj_parts_add(obj_, params.get("path", None), idx_, use_bpy=True, first=first, parent_obj_id=parent_id, joint_info=joint_info)
            first = False
        first =True
        for i, c in enumerate(components):
            if c[1] == "door":
                butil.delete(c[2][:-2])
            elif c[1] == "drawer":
                butil.delete([x[0] for x in c[2]])
            c[0].location = (0, cabinet_params[i], 0)
            butil.apply_transform(c[0], loc=True)
            save_obj_parts_add(c[0], params.get("path", None), idx_, use_bpy=True, first=False)
            first =False
            join_objs.append(c[0])

            # butil.delete(c[:1])
        obj = butil.join_objects(join_objs)
        tagging.tag_system.relabel_obj(obj)
        #save_obj_parts_add(self.aux_drawer_handle, params.get("path", None), idx_, use_bpy=True, first=first)
        join_objects_save_whole(obj, params.get("path", None), idx_, use_bpy=True, join=False)
        return obj


class KitchenCabinetFactory(KitchenCabinetBaseFactory):
    def __init__(
        self, factory_seed, params={}, coarse=False, dimensions=None, drawer_only=False
    ):
        self.dimensions = dimensions
        super().__init__(factory_seed, params, coarse)
        self.drawer_only = drawer_only

    def sample_params(self):
        params = dict()
        if self.dimensions is None:
            dimensions = (uniform(0.25, 0.35), uniform(1.0, 4.0), uniform(0.5, 1.3))
            self.dimensions = dimensions
        else:
            dimensions = self.dimensions
        params["Dimensions"] = dimensions

        params["bottom_board_height"] = 0.06
        params["shelf_depth"] = params["Dimensions"][0] - 0.01
        num_h = int((params["Dimensions"][2] - 0.06) / 0.3)
        params["shelf_cell_height"] = [
            (params["Dimensions"][2] - 0.06) / num_h for _ in range(num_h)
        ]

        self.frame_params = params

        n_cells = max(int(params["Dimensions"][1] / 0.45), 1)
        intervals = np.random.uniform(0.55, 1.0, size=(n_cells,))
        intervals = intervals / intervals.sum() * params["Dimensions"][1]
        self.cabinet_widths = intervals.tolist()

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        x, y, z = self.dimensions
        return new_bbox(-x / 2 * 1.2, x / 2 * 1.2, 0, y * 1.1, 0, (z + 0.06) * 1.03)