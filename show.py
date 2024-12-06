import sapien as sapien
from sapien.utils import Viewer
import numpy as np
import math
import os


def main():
    engine = sapien.core.Engine()
    #engine.set_log_level('warning')

    if True:
        sapien.core.render_config.camera_shader_dir = "rt"
        sapien.core.render_config.viewer_shader_dir = "rt"
        sapien.core.render_config.rt_samples_per_pixel = 64
        sapien.core.render_config.rt_use_denoiser = True

    renderer = sapien.core.SapienRenderer()

    engine.set_renderer(renderer)

    scene_config = sapien.core.SceneConfig()
    scene = engine.create_scene(scene_config)
    #scene = sapien.Scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency
    # scene_config = sapien.core.physx.PhysxSceneConfig()
    # scene_config.gravity = np.array([0.0, 0.0, 0.0])
    # sapien.core.physx.set_scene_config(scene_config)

    # NOTE: How to build (rigid bodies) is elaborated in create_actors.py
    scene.add_ground(altitude=-0.5)  # Add a ground

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    #scene.set_ambient_light([-0.5, -0.5, -0.5])

    #viewer = scene.create_viewer()  # Create a viewer (window)
    viewer = Viewer(renderer)
    viewer.set_scene(scene)  # Set the viewer to observe the scene
    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=0, y=0, z=1.5)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 5), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=200, fovy=1)

    loader = scene.create_urdf_loader()
<<<<<<< HEAD
    robot = loader.load("/home/qihong/infinigen_sep_part_urdf/outputs/bottles/BottleFactory/0/scene.urdf")
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    poses = []
    steps = []
    for joint in robot.get_joints():
        if joint.type == "fixed":
            continue
        limit = joint.limit[0]
        lower = limit[0]
        upper = limit[1]
        steps.append(0.001)
        if not math.isinf(upper) and not math.isinf(lower):
            poses.append((lower + upper) / 2)
        elif not math.isinf(lower):
            poses.append(lower)
        else:
            poses.append(0)
=======
    robots = []
    dir = "/home/pjlab/datasets/partnet_mobility"
    objs = os.listdir(dir)
    objs = objs[:100]
    # for i, obj in enumerate(objs):
    # for i in range(100):
    #     robot = loader.load(f"/home/pjlab/projects/infinigen_sep_part_urdf/outputs/LiteDoorFactory/{i}/scene.urdf")
    #     # try:
    #     #     robot = loader.load(f"{dir}/{obj}/mobility.urdf")
    #     # except:
    #     #     continue
    #     r = i / 10
    #     c = i % 10
    #     robot.set_root_pose(sapien.core.Pose([-5 +  c, -5  +  r, 0], [1, 0, 0, 0]))
    #     robots.append(robot)
    robots.append(loader.load("/home/pjlab/projects/infinigen_sep_part_urdf/outputs/LiteDoorFactory/20/scene.urdf"))
    robots[0].set_root_pose(sapien.core.Pose([0, 0, 0], [1, 0, 0, 0]))
    #robot = loader.load("/home/pjlab/projects/infinigen_sep_part_urdf/outputs/OfficeChairFactory/0/scene.urdf")
    #robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    all_poses = []
    all_steps = []
    for robot in robots:
        poses = []
        steps = []
        for joint in robot.get_joints():
            if joint.type == "fixed":
                continue
            limit = joint.get_limits()
            print(joint.get_name(), limit)
            if len(limit) == 0:
                continue
            limit = limit[0]
            lower = limit[0]
            upper = limit[1]
            steps.append(0.005)
            if not math.isinf(upper) and not math.isinf(lower):
                poses.append((lower + upper) / 2)
            elif not math.isinf(lower):
                poses.append(lower)
            else:
                poses.append(0)
        all_poses.append(poses)
        all_steps.append(steps)
>>>>>>> 0e9251e2835af8d600f6051bbc1d45fdbcd6065e




    
    x = 4
    step_x = 0.01
    z = 2
    step_z = 0.01
    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            for i, robot in enumerate(robots):
                poses = all_poses[i]
                steps = all_steps[i]
                valid_idx = 0
                done = False
                for joint_idx, joint in enumerate(robot.get_joints()):
                    if joint.type == "fixed":
                        continue
                    limit = joint.get_limits()
                    if len(limit) == 0:
                        continue
                    limit = limit[0]
                    lower = limit[0]
                    upper = limit[1]
                    pos = poses[valid_idx]
                    pos += steps[valid_idx]
                    if not math.isinf(upper):
                        if steps[valid_idx] > 0:
                            #pos += steps[valid_idx]
                            if pos > upper:
                                pos = upper
                                steps[valid_idx] *= -1
                    if not math.isinf(lower):
                        if steps[valid_idx] < 0:
                            #pos += steps[valid_idx]
                            if pos < lower:
                                pos = lower
                                steps[valid_idx] *= -1
                    poses[valid_idx] = pos
                    valid_idx += 1
                robot.set_qpos(poses)
                all_poses[i] = poses
                all_steps[i] = steps
            #x += step_x
            # if x > 10 or x < 3:
            #     step_x *= -1
            #z += step_z
            if x == 30:
                step_x = 0
                step_z = 0
            # if z > 10 or z < 1:
            #     step_z *= -1
            #viewer.set_camera_xyz(x=x, y=0, z=z)
            scene.step()    
        scene.update_render()
        viewer.render()



if __name__ == "__main__":
    main()