# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors:
# - Zeyu Ma, Lahav Lipson: Stationary camera selection
# - Alexander Raistrick: Refactor into proposal/validate, camera animation
# - Lingjie Mei: get_camera_trajectory


import logging
import typing
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path

import bpy
import gin
import imageio
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from numpy.random import uniform as U
from tqdm import tqdm

from infinigen.core.nodes import node_utils
from infinigen.core.nodes.node_wrangler import Nodes, NodeWrangler
from infinigen.core.rendering.post_render import colorize_depth
from infinigen.core.tagging import tag_system
from infinigen.core.util import blender as butil
from infinigen.core.util import camera
from infinigen.core.util.blender import SelectObjects, delete
from infinigen.core.util.logging import Timer
from infinigen.core.util.organization import SelectionCriterions
from infinigen.core.util.random import random_general
from infinigen.tools.suffixes import get_suffix

from . import animation_policy

logger = logging.getLogger(__name__)

CAMERA_RIGS_DIRNAME = "CameraRigs"


@gin.configurable
def get_sensor_coords(cam, H, W, sparse=False):
    camd = cam.data
    f_in_m = camd.lens / 1000
    scene = bpy.context.scene
    resolution_x_in_px = W
    resolution_y_in_px = H

    scale = scene.render.resolution_percentage / 100
    sensor_width_in_m = camd.sensor_width / 1000
    sensor_height_in_m = camd.sensor_height / 1000
    assert abs(sensor_width_in_m / sensor_height_in_m - W / H) < 1e-4, (
        sensor_width_in_m,
        sensor_height_in_m,
        W,
        H,
    )

    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if camd.sensor_fit == "VERTICAL":
        # the sensor height is fixed (sensor fit is horizontal),
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = (
            resolution_x_in_px * scale / sensor_width_in_m / pixel_aspect_ratio
        )  # pixels per milimeter
        s_v = resolution_y_in_px * scale / sensor_height_in_m

    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal),
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_m
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_m

    u_0 = resolution_x_in_px * scale / 2  # cx (in pixels) Usually is just W/2
    v_0 = resolution_y_in_px * scale / 2  # cx (in pixels) Usually is just H/2
    xx, yy = np.meshgrid(np.arange(W).astype(float), np.arange(H).astype(float))
    coords_x = (xx - u_0) / s_u  # relative, in mm
    coords_y = (yy - v_0 + 1) / s_v  # relative, in mm

    coords_z = np.full(coords_x.shape, -f_in_m)
    relative_cam_coords = np.stack((coords_x, coords_y, coords_z), axis=-1)

    cam_coords_vectors = np.empty((H, W), dtype=Vector)
    pixel_locs = np.stack((np.meshgrid(np.arange(W), np.arange(H))), axis=-1).reshape(
        (W * H, 2)
    )  # np.array(list(product(range(H), range(W))))
    if sparse:
        ii = np.random.choice(H * W, size=1000)
        pixel_locs = pixel_locs[ii]

    for x, y in tqdm(pixel_locs, desc="Building Camera Vectors", disable=True):
        pixelVector = Vector(relative_cam_coords[y, x])
        cam_coords_vectors[y, x] = cam.matrix_world @ pixelVector

    return cam_coords_vectors, pixel_locs


def adjust_camera_sensor(cam):
    scene = bpy.context.scene
    W = scene.render.resolution_x
    H = scene.render.resolution_y
    sensor_width = 18 * (W / H)
    assert sensor_width.is_integer(), (18, W, H)
    cam.data.sensor_height = 18
    cam.data.sensor_width = int(sensor_width)


def spawn_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.clip_end = 1e4
    adjust_camera_sensor(cam)
    return cam


def camera_name(rig_id, cam_id):
    return f"{CAMERA_RIGS_DIRNAME}/{rig_id}/{cam_id}"


@gin.configurable
def spawn_camera_rigs(
    camera_rig_config,
    n_camera_rigs,
):
    def spawn_rig(i):
        rig_parent = butil.spawn_empty(f"{CAMERA_RIGS_DIRNAME}/{i}")
        for j, config in enumerate(camera_rig_config):
            cam = spawn_camera()
            cam.name = camera_name(i, j)
            cam.parent = rig_parent

            cam.location = config["loc"]
            cam.rotation_euler = config["rot_euler"]

        return rig_parent

    camera_rigs = [spawn_rig(i) for i in range(n_camera_rigs)]
    butil.group_in_collection(camera_rigs, CAMERA_RIGS_DIRNAME)

    return camera_rigs


def get_cameras_ids() -> list[tuple]:
    res = []
    col = bpy.data.collections[CAMERA_RIGS_DIRNAME]
    rigs = [o for o in col.objects if o.name.count("/") == 1]
    for i, root in enumerate(rigs):
        for j, subcam in enumerate(root.children):
            assert subcam.name == camera_name(i, j)
            res.append((i, j))

    return res


def get_camera(rig_id, subcam_id, checkonly=False):
    col = bpy.data.collections[CAMERA_RIGS_DIRNAME]
    name = camera_name(rig_id, subcam_id)
    if name in col.objects.keys():
        return col.objects[name]
    if checkonly:
        return None
    raise ValueError(
        f"Could not get_camera({rig_id=}, {subcam_id=}). {list(col.objects.keys())=}"
    )


@node_utils.to_nodegroup(
    "nodegroup_camera_info", singleton=True, type="GeometryNodeTree"
)
def nodegroup_active_cam_info(nw: NodeWrangler):
    info = nw.new_node(Nodes.ObjectInfo, [bpy.context.scene.camera])
    nw.new_node(
        Nodes.GroupOutput,
        input_kwargs={k: info.outputs[k] for k in info.outputs.keys()},
    )


def set_active_camera(rig_id, subcam_id):
    camera = get_camera(rig_id, subcam_id)
    bpy.context.scene.camera = camera

    ng = (
        nodegroup_active_cam_info()
    )  # does not create a new node group, retrieves singleton
    ng.nodes["Object Info"].inputs["Object"].default_value = camera

    return bpy.context.scene.camera


def positive_gaussian(mean, std):
    while True:
        val = np.random.normal(mean, std)
        if val > 0:
            return val


def set_camera(
    camera,
    location,
    rotation,
    focus_dist,
    frame,
):
    camera.location = location
    camera.rotation_euler = rotation
    if focus_dist is not None:
        camera.data.dof.focus_distance = (
            focus_dist  # this should come before view_layer.update()
        )
    bpy.context.view_layer.update()

    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    if focus_dist is not None:
        camera.data.dof.keyframe_insert(data_path="focus_distance", frame=frame)


def terrain_camera_query(
    cam, scene_bvh, terrain_tags_queries, vertexwise_min_dist, min_dist=0
):
    dists = []
    sensor_coords, pix_it = get_sensor_coords(cam, sparse=True)
    terrain_tags_queries_counts = {q: 0 for q in terrain_tags_queries}

    for x, y in pix_it:
        direction = (sensor_coords[y, x] - cam.matrix_world.translation).normalized()
        _, _, index, dist = scene_bvh.ray_cast(cam.matrix_world.translation, direction)
        if dist is None:
            continue
        dists.append(dist)
        if dist < min_dist or (
            vertexwise_min_dist is not None and dist < vertexwise_min_dist[index]
        ):
            logger.debug(f"Found {dist=} < {min_dist=}")
            dists = None  # means dist < min
            break
        for q in terrain_tags_queries:
            terrain_tags_queries_counts[q] += terrain_tags_queries[q][index]

    n_pix = pix_it.shape[0]

    return dists, terrain_tags_queries_counts, n_pix


@dataclass
class CameraProposal:
    loc: np.array
    rot: np.array
    focal_length: float

    def apply(self, cam):
        cam.location = self.loc
        cam.rotation_euler = self.rot
        cam.data.lens = self.focal_length


@gin.configurable
def camera_pose_proposal(
    scene_bvh,
    location_sample: typing.Callable | tuple,
    altitude=("uniform", 1.5, 2.5),
    roll=0,
    yaw=("uniform", -180, 180),
    pitch=90,
    focal_length=50,
    override_loc=None,
):
    if isinstance(location_sample, tuple):
        location_sample = Vector(location_sample)

        def location_sample():
            return location_sample

    if override_loc is not None:
        loc = Vector(random_general(override_loc))
    elif altitude is None:
        loc = location_sample()
    else:
        loc = location_sample()
        curr_alt = animation_policy.get_altitude(loc, scene_bvh)
        if curr_alt is None:
            logger.debug(f"camera_pose_proposal got {curr_alt=} for {loc=}")
            # butil.spawn_empty("fail")
            return None
        desired_alt = random_general(altitude)
        loc[2] = loc[2] + desired_alt - curr_alt

    rot = np.deg2rad([random_general(pitch), random_general(roll), random_general(yaw)])
    focal_length = random_general(focal_length)
    return CameraProposal(loc, rot, focal_length)


@gin.configurable
def keep_cam_pose_proposal(
    cam,
    terrain,
    scene_bvh,
    placeholders_kd,
    camera_selection_answers,
    vertexwise_min_dist,
    camera_selection_ratio,
    min_placeholder_dist=0,
    min_terrain_distance=0,
    terrain_coverage_range=(0.5, 1),
):
    if terrain is not None:  # TODO refactor
        terrain_sdf = terrain.compute_camera_space_sdf(
            np.array(cam.location).reshape((1, 3))
        )

    if not cam.type == "CAMERA":
        cam = cam.children[0]
    if not cam.type == "CAMERA":
        raise ValueError(f"{cam.name=} had {cam.type=}")

    bpy.context.view_layer.update()

    # Reject cameras too close to any placeholder vertex
    v, i, dist_to_placeholder = placeholders_kd.find(cam.matrix_world.translation)
    if dist_to_placeholder is not None and dist_to_placeholder < min_placeholder_dist:
        logger.debug(f"keep_cam_pose_proposal rejects {dist_to_placeholder=}, {v, i}")
        return None

    dists, camera_selection_answers_counts, n_pix = terrain_camera_query(
        cam,
        scene_bvh,
        camera_selection_answers,
        vertexwise_min_dist,
        min_dist=min_terrain_distance,
    )

    if dists is None:
        logger.debug("keep_cam_pose_proposal rejects terrain dists")
        return None

    coverage = len(dists) / n_pix
    if coverage < terrain_coverage_range[0] or coverage > terrain_coverage_range[1]:
        logger.debug(
            f"keep_cam_pose_proposal rejects {coverage=} for {terrain_coverage_range=}"
        )
        return None

    if terrain is not None and terrain_sdf <= 0:
        logger.debug(f"keep_cam_pose_proposal rejects {terrain_sdf=}")
        return None

    if rparams := camera_selection_ratio:
        for q in rparams:
            if type(q) is tuple and q[0] == SelectionCriterions.CloseUp:
                closeup = len([d for d in dists if d < q[1]]) / n_pix
                if closeup < rparams[q][0] or closeup > rparams[q][1]:
                    logger.debug(f"keep_cam_pose_proposal rejects {closeup=} for {q=}")
                    return None
            else:
                minv, maxv = rparams[q][0], rparams[q][1]
                if q in camera_selection_answers_counts:
                    ratio = camera_selection_answers_counts[q] / n_pix
                    if ratio < minv or ratio > maxv:
                        logger.debug(
                            f"keep_cam_pose_proposal rejects {ratio=} for {q=}"
                        )
                        return None

    return np.std(dists) + 1.5 * np.min(dists)


@gin.configurable
class AnimPolicyGoToProposals:
    def __init__(
        self, speed=("uniform", 1.5, 2.5), min_dist=4, max_dist=10, retries=30
    ):
        self.speed = speed
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.retries = retries

    def __call__(self, camera_rig, frame_curr, retry_pct, bvh):
        margin = Vector((self.max_dist, self.max_dist, self.max_dist))
        bbox = (camera_rig.location - margin, camera_rig.location + margin)

        for _ in range(self.retries):
            res = camera_pose_proposal(bvh, bbox)
            if res is None:
                continue
            dist = np.linalg.norm(np.array(res.loc) - np.array(camera_rig.location))
            if dist < self.min_dist:
                continue
            break
        else:
            raise animation_policy.PolicyError(
                f"{__name__} found no keyframe after {self.retries=}"
            )

        time = dist / random_general(self.speed)
        return Vector(res.loc), Vector(res.rot), time, "BEZIER"


@gin.configurable
def compute_base_views(
    cam,
    n_views,
    terrain,
    scene_bvh,
    location_sample: typing.Callable,
    placeholders_kd=None,
    camera_selection_answers={},
    vertexwise_min_dist=None,
    camera_selection_ratio=None,
    min_candidates_ratio=20,
    max_tries=30000,
    visualize=False,
):
    potential_views = []
    n_min_candidates = int(min_candidates_ratio * n_views)
    with tqdm(total=n_min_candidates, desc="Searching for camera viewpoints") as pbar:
        for it in range(1, max_tries):
            props = camera_pose_proposal(
                scene_bvh=scene_bvh, location_sample=location_sample
            )

            if props is None:
                logger.debug(
                    f"{camera_pose_proposal.__name__} returned {props=} for {it=}"
                )
                continue

            props.apply(cam)

            criterion = keep_cam_pose_proposal(
                cam,
                terrain,
                scene_bvh,
                placeholders_kd,
                camera_selection_answers=camera_selection_answers,
                vertexwise_min_dist=vertexwise_min_dist,
                camera_selection_ratio=camera_selection_ratio,
            )

            if visualize:
                criterion_str = f"{criterion:.2f}" if criterion is not None else "None"
                marker = butil.spawn_empty(f"attempt_{it}_{criterion_str}")
                marker.location = cam.location
                marker.rotation_euler = cam.rotation_euler

            if criterion is None:
                logger.debug(f"{it=} {criterion=}")
                continue

            # Compute focus distance
            destination = cam.matrix_world @ Vector((0.0, 0.0, -1.0))
            forward_dir = (destination - cam.location).normalized()
            *_, straight_ahead_dist = scene_bvh.ray_cast(cam.location, forward_dir)

            potential_views.append((criterion, deepcopy(props), straight_ahead_dist))
            pbar.update(1)

            if len(potential_views) >= n_min_candidates:
                break

    if len(potential_views) < n_views:
        if visualize:
            butil.save_blend("compute_base_views-failed.blend")
        raise ValueError(f"Could not find {n_views} camera views")

    views = sorted(potential_views, reverse=True)

    return views[:n_views]


def build_bvh_and_attrs(objs, tags_queries):
    dup_objs = []
    for obj in objs:
        with SelectObjects(obj):
            bpy.ops.object.duplicate(linked=0, mode="TRANSLATION")
            dup_objs.append(bpy.context.view_layer.objects.active)
    for obj in dup_objs:
        with butil.ViewportMode(obj, "EDIT"):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.quads_convert_to_tris(
                quad_method="BEAUTY", ngon_method="BEAUTY"
            )
    with SelectObjects(dup_objs[0]):
        for obj in dup_objs[1:]:
            obj.select_set(True)
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    bvh = BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())
    from infinigen.terrain.utils import Mesh

    with butil.ViewportMode(obj, "EDIT"):
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    mesh = Mesh(obj=obj)
    delete(obj)

    camera_selection_answers = {}
    for q0 in tags_queries:
        if type(q0) is not tuple:
            q = (q0,)
        else:
            q = q0
        if q[0] in [SelectionCriterions.CloseUp]:
            continue
        if q[0] == SelectionCriterions.Altitude:
            min_altitude, max_altitude = q[1:3]
            altitude = mesh.vertices[:, 2]
            camera_selection_answers[q0] = mesh.facewise_mean(
                (altitude > min_altitude) & (altitude < max_altitude)
            )
        else:
            camera_selection_answers[q0] = np.zeros(len(mesh.faces), dtype=bool)
            for key in tag_system.tag_dict:
                if set(q).issubset(set(key.split("."))):
                    camera_selection_answers[q0] |= (
                        mesh.face_attributes["MaskTag"] == tag_system.tag_dict[key]
                    ).reshape(-1)
    return bvh, camera_selection_answers


def camera_selection_preprocessing(
    terrain,
    scene_objs,
    tags_ratio: dict = None,
    ranges_ratio: dict = None,
    anim_criterion_keys: dict = None,
):
    if tags_ratio is None:
        tags_ratio = {}
    if ranges_ratio is None:
        ranges_ratio = {}
    if anim_criterion_keys is None:
        anim_criterion_keys = {}

    # preprocessing code adapted from mazeyu's original gin-oriented solution
    tags_ratio = {
        k: (*v, anim_criterion_keys.get(k, False)) for k, v in tags_ratio.items()
    }
    ranges_ratio = {
        v[:-2]: (v[-2], v[-1], anim_criterion_keys.get(k, False))
        for k, v in ranges_ratio.items()
    }

    all_selection_ratios = {**tags_ratio, **ranges_ratio}

    with Timer("Building placeholders KDTree"):
        placeholders = list(
            chain.from_iterable(
                c.all_objects
                for c in bpy.data.collections
                if c.name.startswith("placeholders:")
            )
        )
        placeholders = [p for p in placeholders if p.type == "MESH"]
        logger.info(f"Building placeholder kd for {len(placeholders)} objects")
        placeholders_kd = butil.joined_kd(placeholders, include_origins=True)

    if terrain is None:
        scene_bvh, camera_selection_answers = build_bvh_and_attrs(
            scene_objs, all_selection_ratios.keys()
        )
        vertexwise_min_dist = None
    else:
        scene_bvh, camera_selection_answers, vertexwise_min_dist = (
            terrain.build_terrain_bvh_and_attrs(all_selection_ratios.keys())
        )

    return dict(
        terrain=terrain,
        scene_bvh=scene_bvh,
        camera_selection_ratio=all_selection_ratios,
        camera_selection_answers=camera_selection_answers,
        vertexwise_min_dist=vertexwise_min_dist,
        placeholders_kd=placeholders_kd,
    )


@node_utils.to_nodegroup("geo_distrib", singleton=True, type="GeometryNodeTree")
def geo_distrib_random_points(nw: NodeWrangler):
    input = nw.new_node(
        Nodes.GroupInput, expose_input=[("NodeSocketGeometry", "Geometry", None)]
    )
    distribute = nw.new_node(
        Nodes.DistributePointsOnFaces,
        input_kwargs={"Mesh": input.outputs["Geometry"], "Density": 500},
    )
    verts = nw.new_node(Nodes.PointsToVertices, [distribute])
    nw.new_node(Nodes.GroupOutput, input_kwargs={"Geometry": verts})


def sample_random_locs(surface: bpy.types.Object, eps=0.01):
    # HACK implementation - uses blender geonodes' uniform surface sample, im fairly sure theres a numpy impl somewhere in the repo
    surface = butil.copy(surface)
    butil.apply_transform(surface, loc=True, rot=True, scale=True)
    butil.modify_mesh(
        surface, "NODES", node_group=geo_distrib_random_points(), apply=True
    )
    locs = np.array([v.co for v in surface.data.vertices])
    locs[:, -1] += eps
    butil.delete(surface)
    return locs


@gin.configurable
def configure_cameras(
    cam_rigs,
    scene_preprocessed: dict,
    init_bounding_box: tuple[np.array, np.array] = None,
    init_surfaces: list[bpy.types.Object] = None,
):
    bpy.context.view_layer.update()
    dummy_camera = spawn_camera()

    if init_bounding_box is not None:

        def location_sample():
            return np.random.uniform(*init_bounding_box)
    elif init_surfaces is not None:
        random_locs = sample_random_locs(init_surfaces)

        def location_sample():
            loc = Vector(random_locs[np.random.randint(len(random_locs)), :])
            loc.z += 1e-3
            return loc
    else:
        raise ValueError("Either init_bounding_box or init_surfaces must be provided")

    base_views = compute_base_views(
        dummy_camera,
        n_views=len(cam_rigs),
        location_sample=location_sample,
        **scene_preprocessed,
    )

    for view, cam_rig in zip(base_views, cam_rigs):
        score, props, focus_dist = view
        cam_rig.location = props.loc
        cam_rig.rotation_euler = props.rot

        for cam in cam_rig.children:
            cam.data.lens = props.focal_length

        if focus_dist is not None:
            for cam in cam_rig.children:
                if not cam.type == "CAMERA":
                    continue
                cam.data.dof.focus_distance = focus_dist

    butil.delete(dummy_camera)


@gin.configurable
def animate_cameras(
    cam_rigs,
    bounding_box,
    scene_preprocessed,
    pois=None,
    follow_poi_chance=0.0,
    policy_registry=None,
):
    animation_ratio = {}
    animation_answers = {}
    for k in scene_preprocessed["camera_selection_ratio"]:
        if scene_preprocessed["camera_selection_ratio"][k][2]:
            animation_ratio[k] = scene_preprocessed["camera_selection_ratio"][k]
            animation_answers[k] = scene_preprocessed["camera_selection_answers"][k]

    anim_valid_pose_func = partial(
        keep_cam_pose_proposal,
        placeholders_kd=scene_preprocessed["placeholders_kd"],
        scene_bvh=scene_preprocessed["scene_bvh"],
        terrain=scene_preprocessed["terrain"],
        vertexwise_min_dist=scene_preprocessed["vertexwise_min_dist"],
        camera_selection_answers=animation_answers,
        camera_selection_ratio=animation_ratio,
    )

    for cam_rig in cam_rigs:
        if policy_registry is None:
            if U() < follow_poi_chance and pois is not None and len(pois):
                policy = animation_policy.AnimPolicyFollowObject(
                    target_obj=cam_rig, pois=pois, bvh=scene_preprocessed["scene_bvh"]
                )
            else:
                policy = animation_policy.AnimPolicyRandomWalkLookaround()
        else:
            policy = policy_registry()

        logger.info(f"Animating {cam_rig=} using {policy=}")

        animation_policy.animate_trajectory(
            cam_rig,
            scene_preprocessed["scene_bvh"],
            policy_func=policy,
            validate_pose_func=anim_valid_pose_func,
            verbose=True,
            fatal=True,
            bounding_box=bounding_box,
        )


@gin.configurable
def save_camera_parameters(camera_ids, output_folder, frame, use_dof=False):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    for camera_pair_id, camera_id in camera_ids:
        camera_obj = get_camera(camera_pair_id, camera_id)
        if use_dof is not None:
            camera_obj.data.dof.use_dof = use_dof
        # Saving camera parameters
        K = camera.get_calibration_matrix_K_from_blender(camera_obj.data)  # TODO ?
        suffix = get_suffix(
            dict(cam_rig=camera_pair_id, resample=0, frame=frame, subcam=camera_id)
        )
        output_file = output_folder / f"camview{suffix}.npz"

        height_width = np.array(
            (
                bpy.context.scene.render.resolution_y,
                bpy.context.scene.render.resolution_x,
            )
        )
        T = np.asarray(camera_obj.matrix_world, dtype=np.float64) @ np.diag(
            (1.0, -1.0, -1.0, 1.0)
        )  # Y down Z forward (aka opencv)
        np.savez(output_file, K=np.asarray(K, dtype=np.float64), T=T, HW=height_width)


if __name__ == "__main__":
    """
    This interactive section generates a depth map by raycasting through each pixel. 
    It is very useful for debugging camera.py.
    """
    cam = bpy.context.scene.camera

    scene = bpy.context.scene
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080

    adjust_camera_sensor(cam)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvhtree = BVHTree.FromObject(bpy.context.active_object, depsgraph)

    target_obj = bpy.context.active_object
    to_obj_coords = target_obj.matrix_world.inverted()
    sensor_coords, pix_it = get_sensor_coords(cam, sparse=False)

    H, W = sensor_coords.shape
    depth_output = np.zeros((H, W), dtype=np.float64)

    for x, y in tqdm(pix_it):
        destination = sensor_coords[y, x]
        direction = (destination - cam.location).normalized()
        location, normal, index, dist = bvhtree.ray_cast(cam.location, direction)
        if dist is not None:
            dist_diff = (destination - cam.location).length
            assert dist > (location - destination).length, (
                dist,
                (location - destination).length,
            )
            assert dist > dist_diff
            depth_output[H - y - 1, x] = dist - dist_diff

    color_depth = colorize_depth(depth_output)
    imageio.imwrite("color_depth.png", color_depth)
