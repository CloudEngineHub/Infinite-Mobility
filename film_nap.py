from PIL import Image
import sapien as sapien
from sapien.utils import Viewer
import numpy as np
import math
import os
import json
import random
import urdfpy
from PIL import ImageDraw, ImageFont
from reconstruct_obj_according_tourdf import generate_whole
import trimesh
#from translate import Translator

c = ''

def find_all_urdfs(path)-> list[str]:
    urdfs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".urdf"):
                urdfs.append(os.path.join(root, file))
    return urdfs
def iter_tree(r, urdf_path, tree, links, reset_material=True):
    p_name = r.name
    if len(r.visuals) >= 1:
        for v in r.visuals:
            g = v.geometry
            m = g.mesh
            f = m.filename
            path = urdf_path.replace('mobility_no_collision.urdf', f)
            path_ = path.replace('.obj', '.mtl_')
            path = path.replace('.obj', '.mtl')
            print(path, path_)
            if not reset_material and os.path.exists(path):
                os.rename(path, path_)
            elif reset_material and os.path.exists(path_):
                os.rename(path_, path)
                
    for j in tree.joints:
        if j.parent == p_name:
            c_name = j.child
            for l in links:
                if l.name == c_name:
                    iter_tree(l, urdf_path, tree, links, reset_material)


def find_all_objs_in_urdf(path, reset_material=True):
    tree = urdfpy.URDF.load(path)
    r = tree.base_link
    links = tree.links
    iter_tree(r, path, tree, links, reset_material)

def generate_text_image(image, catagory):
    global c, m
    #image = Image.fromarray(image)
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    # 选择字体和大小
    font = ImageFont.truetype('/home/pjlab/下载/经典宋体简/res.ttf', 40)
    # 添加文字到图片上
    draw.text((10, 10), m[catagory], font=font, fill=(0, 0, 0))
    #return np.asarray(image)

def add_noise_to_joint(joint):
    scale = 0.1
    if joint.type == "fixed":
        return
    
    pos_in_child = joint.get_pose_in_child()
    pos_in_parent = joint.get_pose_in_parent()
    limit = joint.get_limits()
    limit = limit[0]
    if len(limit) == 0:
        return
    if np.isinf(limit[0]) and np.isinf(limit[1]):
        pass
    else:
        if not np.isinf(limit[0]):
            limit[0] += random.uniform(-0.1, 0.1)
        if not np.isinf(limit[1]):
            limit[1] += random.uniform(-0.1, 0.1)
        joint.set_limits([limit])
    print(pos_in_child.p, pos_in_parent.p)
    print(pos_in_child.q, pos_in_parent.q)
    
    #if not joint.type == "prismatic":
    joint.set_pose_in_child(sapien.Pose([pos_in_child.p[0] + random.uniform(-scale, scale), pos_in_child.p[1] + random.uniform(-scale, scale), pos_in_child.p[2] + random.uniform(-scale, scale)], 
                                            [pos_in_child.q[0] + random.uniform(-scale, scale), pos_in_child.q[1] + random.uniform(-scale, scale), pos_in_child.q[2] + random.uniform(-scale, scale), pos_in_child.q[3] + random.uniform(-scale, scale)]))


def main(id, path):
    global c
    #engine.set_log_level('warning')

    if False:
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(64)  # change to 256 for less noise
        sapien.render.set_ray_tracing_denoiser("oidn") # change to "optix" or "oidn"


    
    phy_config = sapien.physx.PhysxSceneConfig()
    phy_config.gravity = [0, 0, 0]
    sapien.physx.set_scene_config(phy_config)
    scene = sapien.Scene()
    #scene = sapien.Scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency
    # scene_config = sapien.physx.PhysxSceneConfig()
    # scene_config.gravity = np.array([0.0, 0.0, 0.0])
    # sapien.physx.set_scene_config(scene_config)

    # NOTE: How to build (rigid bodies) is elaborated in create_actors.py
    #scene.add_ground(altitude=-10, render_half_size=[200, 200])  # Add a ground

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    #scene.set_ambient_light([-0.5, -0.5, -0.5])

    #viewer = scene.create_viewer()  # Create a viewer (window)
    #viewer = Viewer(resolutions=(640, 480))  # Create a viewer (window)
    viewer = Viewer(resolutions=[680, 720])  # Create a viewer (window)
    viewer.window.hide()
    viewer.set_scene(scene)  # Set the viewer to observe the scene
    #viewer = scene.create_viewer()
    #viewer.set_scene(scene)  # Set the viewer to observe the scene
    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    #viewer.set_camera_xyz(x=18, y=-20, z=19)
    # viewer.set_camera_xyz(x=-5, y=0, z=1)
    # #viewer.set_camera_xyz(x=0.5, y=0, z=0.2)
    # # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # # The camera now looks at the origin
    # viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 12), y= 0)
    #viewer.set_camera_xyz(x=-3, y=-0.5, z=1)
    #viewer.set_camera_xyz(x=2.5, y=0, z=0.5)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    #viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 6), y= - 0.2 )
    #viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    near, far = 0.1, 100
    width, height = 640, 480

    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    cam_pos = np.array([-2, -2, 3])
    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    # camera = scene.add_camera(
    #     name="camera",
    #     width=width,
    #     height=height,
    #     fovy=np.deg2rad(35),
    #     near=near,
    #     far=far,
    # )
    #camera.entity.set_pose(sapien.Pose(mat44))

    loader = scene.create_urdf_loader()
    robots = []
    # for i, obj in enumerate(objs):
    # #for i in range(100):
    #     #robot = loader.load(f"/home/pjlab/projects/infinigen_sep_part_urdf/outputs/TVFactory/{i}/scene.urdf")
    #     try:
    #         robot = loader.load(f"{dir}/{obj}/mobility.urdf")
    #     except:
    #         continue
    #     r = i / 10
    #     c = i % 10
    #     robot.set_root_pose(sapien.Pose([-10 + 2* c, -10  + 2 * r, 0], [1, 0, 0, 0]))
    #     robots.append(robot)
    ps = path.split('/')
    ps[-1] = 'whole.obj'
    whole_path = '/'.join(ps)
    #whole_path = f"{path}/whole.obj"
    if not os.path.exists(whole_path):
        generate_whole(path)
    mesh = trimesh.load(whole_path, force="mesh")
    center = (mesh.vertices[:, 0].max() + mesh.vertices[:, 0].min()) / 2, (mesh.vertices[:, 1].max() + mesh.vertices[:, 1].min()) / 2, (mesh.vertices[:, 2].max() + mesh.vertices[:, 2].min()) / 2
    scale = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min(), mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min(), mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
    print(scale, center)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    viewer.set_camera_rpy(r=0, p=-np.pi / 2, y= 0)
    z = center[1]
    scale = 3 / abs(scale[1])
    #x += (4 / scale)
    viewer.set_camera_xyz(x=center[0] * 2, y=center[2], z=center[1] + 3)
    #print(x, -center[2], center[1] * 1.2)
    #find_all_objs_in_urdf(f"{dir}/{objs[id]}/mobility_no_collision.urdf", False)
    robots.append(loader.load(path))
    #find_all_objs_in_urdf(f"{dir}/{objs[id]}/mobility_no_collision.urdf", True)
    robots[0].set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    r = robots[0]
    for j in robots[0].get_joints():
        pass
        #add_noise_to_joint(j)
    ls = r.get_links()
    # for l in ls:
    #     l.collision_shapes = []
    #robot = loader.load("/home/pjlab/projects/infinigen_sep_part_urdf/outputs/OfficeChairFactory/0/scene.urdf")
    #robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    all_poses = []
    all_steps = []
    all_childs = []
    valid_joints = {}
    valid_joints_rev = {}
    valid_idx = 0
    for robot in robots:
        poses = []
        steps = []
        for joint in robot.get_joints():
            if joint.type == "fixed":
                continue
            all_childs.append(joint.child_link.entity.name)
            limit = joint.get_limits()
            print(joint.type, limit)
            if len(limit) == 0:
                continue
            limit = limit[0]
            lower = limit[0]
            upper = limit[1]
            if np.isinf(lower) and np.isinf(upper):
                lower = 0
                upper = np.pi * 2
            steps.append((upper - lower) * 2 / 200)
            valid_joints[valid_idx] = joint.name
            valid_joints_rev[joint.name] = valid_idx
            # if not math.isinf(upper) and not math.isinf(lower):
            #     poses.append((lower + upper) / 2)
            # elif not math.isinf(lower):
            #     poses.append(lower)
            # else:
            #     poses.append(0)
            poses.append(0)
            valid_idx += 1
        all_poses.append(poses)
        all_steps.append(steps)

    if len(robots) == 1:
        mode = "joint by joint"
    else:
        mode = "all joints"


    mode = "all joints"
    x = 4
    step_x = 0.01
    z = 2
    step_z = 0.01
    step = 0
    done = False
    joint_idx = 0
    valid_idx = 0
    joint_dependency = {
        0:{"reverse": {1: ['smaller', 0.1], 2: ['smaller', 0.1], 2: ['smaller', 0.1]}},
        1:{"forward": {0: ['bigger', 1]}},
        2:{"forward": {0: ['bigger', 1]}},
        #2:{"forward": {3: ['bigger', 1]}}
        #4:{"forward": {1: ['bigger', 1]}},
        #7:{"forward": {1: ['bigger', 1]}},
    }

    joint_dependency = {}

    considered_joints = ['joint_revolute_9', 'joint_prismatic_6', 'joint_prismatic_7', 'joint_prismatic_8']
    print(valid_joints)
    considered_joints = [name for name in valid_joints.values() if 'prismatic' in name and 'lr' not in name]
    print(considered_joints)
    considered_joints = []

    while not viewer.closed:
        all_entity_contacted = {}
        constacts = scene.get_contacts()
        for contact in constacts:
            if contact.bodies[0].entity.name in all_entity_contacted:
                continue
            if contact.bodies[1].entity.name in all_entity_contacted:
                continue
            if contact.bodies[0].entity.name not in all_entity_contacted:
                all_entity_contacted[contact.bodies[0].entity.name] = 1
            else:
                all_entity_contacted[contact.bodies[0].entity.name] += 1
            if contact.bodies[1].entity.name not in all_entity_contacted:
                all_entity_contacted[contact.bodies[1].entity.name] = 1
            else:
                all_entity_contacted[contact.bodies[1].entity.name] += 1
        for _ in range(1):  # render every 4 steps
            if mode == "all joints":
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
                        if len(considered_joints) > 0:
                            if joint.name not in considered_joints:
                                poses[valid_idx] = 0
                                valid_idx += 1
                                continue
                        limit = limit[0]
                        lower = limit[0]
                        upper = limit[1]
                        pos = poses[valid_idx]
                        pos += steps[valid_idx]
                        l_1 = joint.child_link.entity.name
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
                        # if l_1 in all_entity_contacted:
                        #     steps[valid_idx] *= -1
                        #     pos -= steps[valid_idx]
                        #     poses[valid_idx] = pos
                        valid_idx += 1
                    print(poses)
                    robot.set_qpos(poses)
                    all_poses[i] = poses
                    all_steps[i] = steps
            else:
                robot = robots[0]
                poses = all_poses[0]
                steps = all_steps[0]
                if not done:
                    joint = robot.get_joints()[joint_idx]
                    if joint.type == "fixed":
                        done = True
                        continue
                    limit = joint.get_limits()
                    if len(limit) == 0:
                        done = True
                        continue
                    if len(considered_joints) > 0:
                        if joint.name not in considered_joints:
                            poses[valid_idx] = 0
                            done = True
                            continue
                    if joint_dependency != {}:
                        dependeny = joint_dependency.get(valid_idx, None)
                        if dependeny is not None:
                            if steps[valid_idx] < 0:
                                dependeny = dependeny.get("reverse", None)
                            else:
                                dependeny = dependeny.get("forward", None)
                            if dependeny is not None:
                                print(dependeny)
                                for dep in dependeny.keys():
                                    condition = dependeny[dep]
                                    if condition[0] == "smaller":
                                        if poses[dep] > condition[1]:
                                            done = True
                                            break
                                    elif condition[0] == "bigger":
                                        if poses[dep] < condition[1]:
                                            done = True
                                            break
                    if done:
                        continue
                    print(joint.name)

                    limit = limit[0]
                    lower = limit[0]
                    upper = limit[1]
                    pos = poses[valid_idx]
                    pos += steps[valid_idx]
                    l_1 = joint.child_link.entity.name
                    if not math.isinf(upper):
                        if steps[valid_idx] > 0:
                            #pos += steps[joint_idx]
                            if pos > upper:
                                pos = upper
                                steps[valid_idx] *= -1
                                done = True
                                continue
                    if not math.isinf(lower):
                        if steps[valid_idx] < 0:
                            #pos += steps[joint_idx]
                            if pos < lower:
                                pos = lower
                                steps[valid_idx] *= -1
                                done = True
                                continue
                    poses[valid_idx] = pos
                    # if l_1 in all_entity_contacted:
                    #     #steps[valid_idx] *= -1
                    #     pos -= steps[valid_idx]
                    #     done = True
                    #     poses[valid_idx] = pos
                    #     continue
                    robot.set_qpos(poses)
                    all_poses[0] = poses
                    all_steps[0] = steps
                else:
                    done = False
                    joint_now = robot.get_joints()[joint_idx]
                    if joint_now.type == "fixed":
                        joint_idx += 1
                        joint_idx %= len(robot.get_joints())
                        print(joint_idx)
                        continue
                    limit = joint_now.get_limits()
                    if len(limit) == 0:
                        joint_idx += 1
                        joint_idx %= len(robot.get_joints())
                        continue
                    joint_idx += 1
                    valid_idx += 1
                    joint_idx %= len(robot.get_joints())
                    valid_idx %= len(poses)




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
        #scene.update_render()  # sync pose from SAPIEN to renderer
        rgba = viewer.window.get_picture("Color")
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)
        #generate_text_image(rgba_pil, catagory)
        rgba_pil.save(f"pics___/screenshot{step}.png")
        step += 1
        print(step)
        if step == 300:
            break
        #print(scene.get_contacts())
        #print(step)
    os.system("ffmpeg -r 30 -i ./pics___/screenshot%d.png -vf palettegen ./pics___/palette.png")
    os.system(f"ffmpeg -r 30 -i ./pics___/screenshot%d.png -i ./pics___/palette.png -lavfi paletteuse ./pics___/gifs_/{id}.gif")
    os.system("rm ./pics___/palette.png")
    viewer.close()
    return True



if __name__ == "__main__":
    path = '/home/pjlab/下载/nap_urdf_s400/G'
    starting_number = 0
    #catagory = 'WashingMachine'
    res = find_all_urdfs(path)
    max_number = 400
    res = res[:max_number]
    for i in range(len(res)):
        i += starting_number
        print(f"#########################################################################start {i} object##########################################################################")
        try:
            main(i, res[i])
        except Exception as e:
            print(e)
        print(f"#########################################################################end {i} object##########################################################################")