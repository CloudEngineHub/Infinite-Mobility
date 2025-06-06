# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei
import bpy
import numpy as np
from numpy.random import uniform

from infinigen.assets.material_assignments import AssetList
from infinigen.assets.utils.auxiliary_parts import random_auxiliary
from infinigen.assets.utils.decorate import (
    read_center,
    read_co,
    read_edge_center,
    read_edges,
    read_normal,
    select_edges,
    select_faces,
    select_vertices,
    subsurf,
    write_attribute,
    write_co,
)
from infinigen.assets.utils.draw import align_bezier
from infinigen.assets.utils.object import (
    get_joint_name,
    join_objects,
    new_bbox,
    new_cube,
    new_cylinder,
    # save_obj_parts_join_objects,
    save_objects_obj,
    join_objects_save_whole,
    save_obj_parts_add
)
from infinigen.core.nodes.node_utils import save_geometry
from infinigen.core.placement.factory import AssetFactory
from infinigen.core.util import blender as butil
from infinigen.core.util.blender import deep_clone_obj
from infinigen.core.util.math import FixedSeed, normalize
from infinigen.core.util.random import log_uniform



class ToiletFactory(AssetFactory):
    def __init__(self, factory_seed, coarse=False):
        super().__init__(factory_seed, coarse)
        with FixedSeed(self.factory_seed):
            self.size = uniform(0.4, 0.5)
            self.width = self.size * uniform(0.7, 0.8)
            self.height = self.size * uniform(0.8, 0.9)
            self.size_mid = uniform(0.6, 0.65)
            self.curve_scale = log_uniform(0.8, 1.2, 4)
            self.depth = self.size * uniform(0.5, 0.6)
            self.tube_scale = uniform(0.25, 0.3)
            self.thickness = uniform(0.05, 0.06)
            self.extrude_height = uniform(0.015, 0.02)
            self.stand_depth = self.depth * uniform(0.85, 0.95)
            self.stand_scale = uniform(0.7, 0.85)
            self.bottom_offset = uniform(0.5, 1.5)
            self.back_thickness = self.thickness * uniform(0, 0.8)
            self.back_size = self.size * uniform(0.55, 0.65)
            self.back_scale = uniform(0.8, 1.0)
            self.seat_thickness = uniform(0.1, 0.3) * self.thickness
            self.seat_size = self.thickness * uniform(1.2, 1.6)
            self.has_seat_cut = uniform() < 0.1
            self.tank_width = self.width * uniform(1.0, 1.2)
            self.tank_height = self.height * uniform(0.6, 1.0)
            self.tank_size = self.back_size - self.seat_size - uniform(0.02, 0.03)
            self.tank_cap_height = uniform(0.03, 0.04)
            self.tank_cap_extrude = 0 if uniform() < 0.5 else uniform(0.005, 0.01)
            self.cover_rotation = -np.pi / 2#uniform(0, np.pi / 2)
            self.hardware_type = np.random.choice(["button", "handle"])
            self.hardware_cap = uniform(0.01, 0.015)
            self.hardware_radius = uniform(0.015, 0.02)
            self.hardware_length = uniform(0.04, 0.05)
            self.hardware_on_side = uniform() < 0.5
            material_assignments = AssetList["ToiletFactory"]()
            self.surface = material_assignments["surface"].assign_material()
            self.hardware_surface = material_assignments[
                "hardware_surface"
            ].assign_material()

            is_scratch = uniform() < material_assignments["wear_tear_prob"][0]
            is_edge_wear = uniform() < material_assignments["wear_tear_prob"][1]
            self.scratch = material_assignments["wear_tear"][0] if is_scratch else None
            self.edge_wear = (
                material_assignments["wear_tear"][1] if is_edge_wear else None
            )
            self.use_aux_base = np.random.choice([True, False], p=[0.5, 0.5])
            if self.use_aux_base:
                self.aux_base = random_auxiliary("toilet_base")

    @property
    def mid_offset(self):
        return (1 - self.size_mid) * self.size

    def create_placeholder(self, **kwargs) -> bpy.types.Object:
        return new_bbox(
            -self.mid_offset - self.back_size - self.tank_cap_extrude,
            self.size_mid * self.size + self.thickness + self.thickness,
            -self.width / 2 - self.thickness * 1.1,
            self.width / 2 + self.thickness * 1.1,
            -self.height,
            max(
                self.tank_height,
                -np.sin(self.cover_rotation)
                * (self.seat_size + self.size + self.thickness + self.thickness),
            ),
        )

    def create_asset(self, **params) -> bpy.types.Object:
        upper = self.build_curve()
        lower = deep_clone_obj(upper)
        lower.scale = [self.tube_scale] * 3
        lower.location = 0, self.tube_scale * self.mid_offset / 2, -self.depth
        butil.apply_transform(lower, True)
        bottom = deep_clone_obj(upper)
        bottom.scale = [self.stand_scale] * 3
        bottom.location = (
            0,
            self.tube_scale * (1 - self.size_mid) * self.size / 2 * self.bottom_offset,
            -self.height,
        )
        butil.apply_transform(bottom, True)

        obj = self.make_tube(lower, upper, params.get("path", None), params.get("i", "unknown"))
        seat, cover = self.make_seat(obj)
        stand = self.make_stand(obj, bottom)
        back = self.make_back(obj)
        tank = self.make_tank(params.get("path", None), params.get("i", "unknown"))
        butil.modify_mesh(obj, "BEVEL", segments=2)
        match self.hardware_type:
            case "button":
                hardware = self.add_button(params.get("path", None), params.get("i", "unknown"))
            case "handle":
                hardware = self.add_handle(params.get("path", None), params.get("i", "unknown"))
        write_attribute(hardware, 1, "hardware", "FACE")
        # save_objects_obj(
        #     [obj, seat, cover, stand, back],
        #     params.get("path", None),
        #     params.get("i", "unknown"),
        #     name=["tube", "seat", "cover", "stand", "back"],
        #     obj_name="Toilet",
        #     first=False
        # )
        self.surface.apply(obj, clear=True, metal_color="plain")
        self.hardware_surface.apply(obj, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(obj)
        if self.edge_wear:
            self.edge_wear.apply(obj)
        if not self.use_aux_base:
            save_obj_parts_add([obj], params.get("path", None), params.get("i", "unknown"),"toilet_tube", first=False, use_bpy=True)
        else:
            obj_ = butil.deep_clone_obj(obj)
        self.surface.apply(seat, clear=True, metal_color="plain")
        self.hardware_surface.apply(seat, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(seat)
        if self.edge_wear:
            self.edge_wear.apply(seat)
        seat.location[1] -= 0.02
        depth = read_co(seat)[:, 1]
        depth = depth.max() - depth.min()
        save_obj_parts_add([seat], params.get("path", None), params.get("i", "unknown"),"toilet_seat", first=False, use_bpy=True, parent_obj_id="world", joint_info={
            "name": get_joint_name("revolute"),
            "type": "revolute",
            "axis": (1, 0, 0),
            "limit":{
                "lower": -np.pi / 2,
                "upper": 0
            },
            "origin_shift": (0, depth / 2, 0)
        })
        self.surface.apply(cover, clear=True, metal_color="plain")
        self.hardware_surface.apply(cover, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(cover)
        if self.edge_wear:
            self.edge_wear.apply(cover)
        cover.location[2] += 0.02
        butil.apply_transform(cover, True)
        save_obj_parts_add([cover], params.get("path", None), params.get("i", "unknown"),"toilet_cover", first=False, use_bpy=True, parent_obj_id="world", joint_info={
            "name": get_joint_name("revolute"),
            "type": "revolute",
            "axis": (-1, 0, 0),
            "limit":{
                "lower": -np.pi / 2,
                "upper": 0 #- np.pi / 32
            },
            "origin_shift": (0, 0, -depth  / 2)
        })
        self.surface.apply(stand, clear=True, metal_color="plain")
        self.hardware_surface.apply(stand, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(stand)
        if self.edge_wear:
            self.edge_wear.apply(stand)
        if not self.use_aux_base:
            save_obj_parts_add([stand], params.get("path", None), params.get("i", "unknown"),"toilet_control", first=False, use_bpy=True)
        else:
            stand_ = butil.deep_clone_obj(stand)
        if self.use_aux_base:
            base = self.aux_base[0]
            base_o = butil.join_objects([obj_, stand_])
            co = read_co(base_o)
            scale = co[:, 0].max() - co[:, 0].min(), co[:, 1].max() - co[:, 1].min(), co[:, 2].max() - co[:, 2].min()
            location = co[:, 0].min() + scale[0] / 2, co[:, 1].min() + scale[1] / 2, co[:, 2].min() + scale[2] / 2
            base.rotation_euler = (np.pi / 2, 0, 0)
            butil.apply_transform(base, True)
            base.scale = scale
            butil.apply_transform(base, True)
            base.location = location
            butil.apply_transform(base, True)
            self.surface.apply(base, clear=True, metal_color="plain")
            save_obj_parts_add([base], params.get("path", None), params.get("i", "unknown"),"toilet_base", first=False, use_bpy=True)
        self.surface.apply(back, clear=True, metal_color="plain")
        self.hardware_surface.apply(back, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(back)
        if self.edge_wear:
            self.edge_wear.apply(back)
        save_obj_parts_add([back], params.get("path", None), params.get("i", "unknown"),"toilet_back", first=False, use_bpy=True)
        obj = join_objects([obj, seat, cover, stand, back, tank, hardware])
        join_objects_save_whole(
            obj,
            params.get("path", None),
            params.get("i", "unknown"),
            join=False,
            use_bpy=True
        )
        #save_obj_parts_add([obj], params.get("path", None), params.get("i", "unknown"),"part", first=False, use_bpy=True)
        # obj.rotation_euler[-1] = np.pi / 2
        # butil.apply_transform(obj)
        return obj

    def build_curve(self):
        x_anchors = [0, self.width / 2, 0]
        y_anchors = [-self.size_mid * self.size, 0, self.mid_offset]
        axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([1, 0, 0])]
        obj = align_bezier([x_anchors, y_anchors, 0], axes, self.curve_scale)
        butil.modify_mesh(obj, "MIRROR", use_axis=(True, False, False))
        return obj

    def make_tube(self, lower, upper, path=None, i="unknown"):
        obj = join_objects([upper, lower])
        # save_obj_parts_add([upper, lower], path, i, name="tube")
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=np.random.randint(12, 16),
                profile_shape_factor=uniform(0.1, 0.2),
                interpolation="SURFACE",
            )
        butil.modify_mesh(
            obj,
            "SOLIDIFY",
            thickness=self.thickness,
            offset=1,
            solidify_mode="NON_MANIFOLD",
            nonmanifold_boundary_mode="FLAT",
        )
        normal = read_normal(obj)
        select_faces(obj, normal[:, -1] > 0.9)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.extrude_region_move(
                TRANSFORM_OT_translate={
                    "value": (0, 0, self.thickness + self.extrude_height)
                }
            )
        x, y, z = read_co(obj).T
        write_co(obj, np.stack([x, y, np.clip(z, None, self.extrude_height)], -1))
        return obj

    def make_seat(self, obj):
        seat = self.make_plane(obj)
        cover = deep_clone_obj(seat)
        butil.modify_mesh(seat, "SOLIDIFY", thickness=self.extrude_height, offset=1)
        if self.has_seat_cut:
            cutter = new_cube()
            cutter.scale = [self.thickness] * 3
            cutter.location = 0, -self.thickness / 2 - self.size_mid * self.size, 0
            butil.apply_transform(cutter, True)
            butil.select_none()
            butil.modify_mesh(seat, "BOOLEAN", object=cutter, operation="DIFFERENCE")
            butil.delete(cutter)
        butil.modify_mesh(seat, "BEVEL", segments=2)

        x, y, _ = read_edge_center(cover).T
        i = np.argmin(np.abs(x) + np.abs(y))
        selection = np.full(len(x), False)
        selection[i] = True
        select_edges(cover, selection)
        with butil.ViewportMode(cover, "EDIT"):
            bpy.ops.mesh.loop_multi_select()
            bpy.ops.mesh.fill_grid()
        butil.modify_mesh(cover, "SOLIDIFY", thickness=self.extrude_height, offset=1)
        cover.location = [
            0,
            -self.mid_offset - self.seat_size + self.extrude_height / 2,
            -self.extrude_height / 2,
        ]
        butil.apply_transform(cover, True)
        cover.rotation_euler[0] = self.cover_rotation
        cover.location = [
            0,
            self.mid_offset + self.seat_size - self.extrude_height / 2,
            self.extrude_height * 1.5,
        ]
        butil.apply_transform(cover, True)
        butil.modify_mesh(cover, "BEVEL", segments=2)
        return seat, cover

    def make_plane(self, obj):
        select_faces(obj, lambda x, y, z: z > self.extrude_height * 2 / 3)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        seat = next(o for o in bpy.context.selected_objects if o != obj)
        butil.select_none()
        select_vertices(seat, lambda x, y, z: y > self.mid_offset + self.seat_thickness)
        with butil.ViewportMode(seat, "EDIT"):
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, self.seat_size + self.thickness * 2, 0)
                }
            )
        x, y, z = read_co(seat).T
        write_co(
            seat,
            np.stack([x, np.clip(y, None, self.mid_offset + self.seat_size), z], -1),
        )
        return seat

    def make_stand(self, obj, bottom, path=None, i="unknown"):
        co = read_co(obj)[read_edges(obj).reshape(-1)].reshape(-1, 2, 3)
        horizontal = np.abs(normalize(co[:, 0] - co[:, 1])[:, -1]) < 0.1
        x, y, z = read_edge_center(obj).T
        under_depth = z < -self.stand_depth
        i = np.argmin(y - horizontal - under_depth)
        selection = np.full(len(co), False)
        selection[i] = True
        select_edges(obj, selection)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.loop_multi_select()
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        stand = next(o for o in bpy.context.selected_objects if o != obj)
        stand = join_objects([stand, bottom])
        # save_obj_parts_add([stand, bottom], path, i, name=["stand"])
        with butil.ViewportMode(stand, "EDIT"):
            bpy.ops.mesh.select_mode(type="EDGE")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.bridge_edge_loops(
                number_cuts=np.random.randint(12, 16),
                profile_shape_factor=uniform(0.0, 0.15),
            )
        return stand

    def make_back(self, obj):
        back = read_center(obj)[:, 1] > self.mid_offset - self.back_thickness
        back_facing = read_normal(obj)[:, 1] > 0.1
        butil.select_none()
        select_faces(obj, back & back_facing)
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.region_to_loop()
            bpy.ops.mesh.duplicate_move()
            bpy.ops.mesh.separate(type="SELECTED")
        back = next(o for o in bpy.context.selected_objects if o != obj)
        butil.modify_mesh(back, "CORRECTIVE_SMOOTH")
        butil.select_none()
        with butil.ViewportMode(back, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.extrude_edges_move(
                TRANSFORM_OT_translate={
                    "value": (0, self.back_size + self.thickness * 2, 0)
                }
            )
            bpy.ops.transform.resize(value=(self.back_scale, 1, 1))
            bpy.ops.mesh.edge_face_add()
        back.location[1] -= 0.01
        butil.apply_transform(back, True)
        x, y, z = read_co(back).T
        write_co(
            back,
            np.stack([x, np.clip(y, None, self.mid_offset + self.back_size), z], -1),
        )
        return back

    def make_tank(self, path=None, i="unknown"):
        tank = new_cube()
        tank.scale = self.tank_width / 2, self.tank_size / 2, self.tank_height / 2
        tank.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height / 2,
        )
        butil.apply_transform(tank, True)
        subsurf(tank, 2, True)
        butil.modify_mesh(tank, "BEVEL", segments=2)
        cap = new_cube()
        cap.scale = (
            self.tank_width / 2 + self.tank_cap_extrude,
            self.tank_size / 2 + self.tank_cap_extrude,
            self.tank_cap_height / 2,
        )
        cap.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height,
        )
        butil.apply_transform(cap, True)
        butil.modify_mesh(
            cap, "BEVEL", width=uniform(0, self.extrude_height), segments=4
        )
        #save_obj_parts_add([tank, cap], path, i, name=["tank", "cap"])
        self.finalize_assets(tank)
        self.finalize_assets(cap)
        save_obj_parts_add([tank], path, i, "tank_cap", first=True, use_bpy=True, parent_obj_id="world", joint_info={
            "name": get_joint_name("fixed"),
            "type": "fixed",
        })
        save_obj_parts_add([cap], path, i, "tank_body", first=False, use_bpy=True, parent_obj_id="world", joint_info={
            "name": get_joint_name("prismatic"),
            "type": "prismatic",
            "axis": (0, 0, 1),
            "limit":{
                "lower": 0,
                "upper": 0.2
            }
        })
        tank = join_objects([tank, cap])
        return tank

    def add_button(self, path, i):
        obj = new_cylinder()
        obj.scale = (
            self.hardware_radius,
            self.hardware_radius,
            self.tank_cap_height/2 + 1e-3,
        )
        obj.location = (
            0,
            self.mid_offset + self.back_size - self.tank_size / 2,
            self.tank_height + self.tank_cap_height / 4,
        )
        butil.apply_transform(obj, True)
        save_obj_parts_add([obj], path, i, name="button", first=False, use_bpy=True, parent_obj_id=1, joint_info={
            "name": get_joint_name("prismatic"),
            "type": "prismatic",
            "axis": (0, 0, 1),
            "limit":{
                "lower": -self.tank_cap_height / 4,
                "upper": 0
            }
        })
        return obj

    def add_handle(self, path=None, i="unknown"):
        obj = new_cylinder()
        obj.scale = self.hardware_radius, self.hardware_radius, self.hardware_cap
        obj.rotation_euler[0] = np.pi / 2
        butil.apply_transform(obj, True)
        lever = new_cylinder()
        lever.scale = (
            self.hardware_radius / 2,
            self.hardware_radius / 2,
            self.hardware_length,
        )
        lever.rotation_euler[1] = np.pi / 2
        lever.location = [
            -self.hardware_radius * uniform(0, 0.5),
            -self.hardware_cap,
            -self.hardware_radius * uniform(0, 0.5),
        ]
        butil.apply_transform(lever, True)
        u = uniform(0.005, 0.01)
        a = uniform(0.01, 0.02)
        b = uniform(0.02, 0.03)
        c = uniform(0.01, 0.02)
        d = uniform(0.02, 0.03)
        for obj_1 in [obj, lever]:
            if self.hardware_on_side:
                obj_1.location = [
                    -self.tank_width / 2 + self.hardware_radius + a,
                    self.mid_offset + self.back_size - self.tank_size,
                    self.tank_height - self.hardware_radius - b,
                ]
            else:
                obj_1.location = [
                    -self.tank_width / 2,
                    self.mid_offset
                    + self.back_size
                    - self.tank_size
                    + self.hardware_radius
                    + c,
                    self.tank_height - self.hardware_radius - d,
                ]
                obj_1.rotation_euler[-1] = -np.pi / 2
            butil.apply_transform(obj_1, True)
            butil.modify_mesh(obj_1, "BEVEL", width=u, segments=2)
        self.finalize_assets(obj)
        self.finalize_assets(lever)
        save_obj_parts_add([obj], path, i, name="lever_spin", first=False, use_bpy=True, parent_obj_id="world", joint_info={
            "name": get_joint_name("revolute"),
            "type": "revolute",
            "axis": (0, 1, 0) if self.hardware_on_side else (1, 0, 0),
            "limit":{
                "lower": 0,
                "upper": np.pi / 2
            }})
        save_obj_parts_add([lever], path, i, name="lever_handle", first=False, use_bpy=True, parent_obj_id=2, joint_info={
            "name": get_joint_name("fixed"),
            "type": "fixed",
        })
        #save_obj_parts_add([obj, lever], path, i, name=["handle", "handle_lever"], first=False)
        obj_1 = join_objects([obj, lever])
        return obj_1

    def finalize_assets(self, assets):
        self.surface.apply(assets, clear=True, metal_color="plain")
        self.hardware_surface.apply(assets, "hardware", metal_color="natural")
        if self.scratch:
            self.scratch.apply(assets)
        if self.edge_wear:
            self.edge_wear.apply(assets)
