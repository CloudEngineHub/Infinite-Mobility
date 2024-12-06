import sapien as sapien
from sapien.utils import Viewer
import numpy as np
import math


def main():
    scene = sapien.Scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency

    # NOTE: How to build (rigid bodies) is elaborated in create_actors.py
    scene.add_ground(altitude=-0.5)  # Add a ground

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    #scene.set_ambient_light([-0.5, -0.5, -0.5])

    viewer = scene.create_viewer()  # Create a viewer (window)

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-4, y=0, z=2)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    loader = scene.create_urdf_loader()
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




    
    active_joints = robot.get_active_joints()
    while not viewer.closed:
        for _ in range(4):  # render every 4 steps
            valid_idx = 0
            for joint_idx, joint in enumerate(active_joints):
                if joint.type == "fixed":
                    continue
                limit = joint.limit[0]
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
            scene.step()    
        scene.update_render()
        viewer.render()



if __name__ == "__main__":
    main()