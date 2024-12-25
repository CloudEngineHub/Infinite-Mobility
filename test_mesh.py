import sys

import os
from sample_cloud import *
import bpy
import bmesh
from scipy.spatial.transform import Rotation
from bpy_lib import *
from math import radians, pi
import random, string, json, subprocess, os
from geometry_node import *
from util import *
import time
import re
import argparse

#设置一个记录信息的类
class BlenderScriptGenerator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.objects = {}
        self.scripts = ['import bpy', 'from math import radians, pi',
                        'from bpy_lib import *\n', 'delete_all()\n']
        self.prompts = ['Remove every object from the scene.']

    def add_script(self, script, prompt=None):
        self.scripts.append(script)
        if prompt:
            self.prompts.append(prompt)

    def find_numbers(self, scripts):
        pattern = r"[+-]?\d+(\.\d*)?"
        matches = re.finditer(pattern, scripts)
        first = [m.start() for m in matches]
        matches = re.finditer(pattern, scripts)
        last = [m.end() for m in matches]
        start = []
        end = []
        for s, e in zip(first, last):
            if scripts[s - 1].isalpha():
                pass
            else:
                start.append(s)
                end.append(e)

        return start, end

    def generate_script(self):
        scripts = '\n'.join(self.scripts)
        prompts = ' '.join(self.prompts)
        return scripts, prompts, *self.find_numbers(scripts)

generator = BlenderScriptGenerator()


##下面就是生成一个类的函数
def plane_object(generator,with_rotation=False,canonical=False,set_small=False):
    name="translation"
    bevel_name="polyline"
    thickness = random.uniform(0.001,0.01) if random.uniform(0,1)<0.5 else 0.
    flip_normals=False
    if thickness<1e-10:
        flip_normals = True if random.uniform(0,1)<0.5 else False
    fill_caps = "none"
    
    #生成bevel并随机旋转一个角度
    bevel_points=generate_random_polyline_bevel()
    bevel_points=np.array(bevel_points)
    eul = mathutils.Euler((0, 0, random.uniform(0,2*pi)), 'XYZ')
    bevel_points = (np.array(eul.to_matrix())@bevel_points.T).T
    
    bevel_param={"name":bevel_name,"points":bevel_points}
    bevel_param_new = create_section_shape("polyline", bevel_param)
    bpy.data.objects.remove(bpy.data.objects[bevel_name], do_unlink=True)

    len = random.uniform(1.0, 1.2)
    points = [[0,0,0],[len,0,0]]
    points_radius = [1, 1]

    bevel_param_new, points_radius = change_bevelparam_according_radius(bevel_param_new,points_radius)
    points_radius = [round(item,2) for item in points_radius]

    #为轨迹线指定好朝向
    if canonical:
        noise = [random.gauss(0,5.0/180.0*pi) for _ in range(2)] + [0.] if random.uniform(0,1)<0.5 else [0.,0.,0.]
        #noise = [0,0,0]
        random_angle = [ noise[i]+ pi/2*random.sample([-1,0,1,2],k=1)[0] for i in range(2)] + [0.]
        rotate_rad = [normalize_to_pi(value) for value in random_angle]
    else:
        rotate_rad = [random.uniform(-pi, pi) for _ in range(3)]
    eul = mathutils.Euler(rotate_rad, 'XYZ')
    rotation_mat = np.array(eul.to_matrix())
    points = rotation_mat@np.array(points).T
    points = points.T
    #确定点的顺序，xyz优先级, 最小的点作为起始点
    if 'points' in bevel_param_new.keys():
        bevel_param_new['points'] = np.around(bevel_param_new['points'],2)
    points = np.around(points,2)

    #创建平移体
    create_section_shape("polyline", bevel_param_new)
    create_curve_translation(name, bevel_name, control_points=points, points_radius=points_radius,thickness=thickness, fill_caps=fill_caps,flip_normals=flip_normals)
    
    #根据现在的mesh调整初始点,scale+location
    curve_param = {"points":points,"thickness":thickness}
    curve_param_final, bevel_param_final = change_param_according_mesh(name,curve_param,bevel_param_new,canonical=canonical,is_straight_line=True)

    #重新生成截面和平移体 (最终版！！)
    create_section_shape("polyline", bevel_param_final)
    create_curve_translation(name, bevel_name, control_points=curve_param_final["points"], points_radius=points_radius,thickness=thickness, fill_caps=fill_caps,flip_normals=flip_normals)

    control_points = curve_param_final["points"].tolist()
    thickness = curve_param_final["thickness"]
    if "points" in bevel_param_final.keys():
        bevel_param_final["points"]=bevel_param_final["points"].tolist()
    
    #保存代码，保存的是最终版的代码
    script = f"create_section_shape('polyline', {bevel_param_final})\n"
    script += f"create_curve_translation('{name}', '{bevel_name}', control_points={control_points}, points_radius={points_radius},thickness={thickness},fill_caps='{fill_caps}',flip_normals={flip_normals})"
    prompt = "Create a translation."
    generator.add_script(script, prompt)

    
def generate_and_export(id, with_rotation=True, canonical=False, type="plane",set_small=False):
    generator.reset()
    #bpy.ops.wm.read_factory_settings()
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    except:
        bpy.ops.wm.read_homefile()
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    if type=="plane":
        plane_object(with_rotation=with_rotation,canonical=canonical,set_small=set_small)

    bpy.ops.wm.ply_export(filepath=f'{point_cloud_out_dir}/obj{id}.ply', export_triangulated_mesh=True)
    print(f"number {id} is finished")

    return generator.generate_script()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--i_start', type=int, default=0)
    parser.add_argument('--i_end', type=int, default=20)
    parser.add_argument('--root_out_dir', type=str, default="output")
    parser.add_argument('--canonical', type=bool, default=True)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--shape_type', type=str, default='plane')
    parser.add_argument('--set_small', type=bool, default=False)

    args = parser.parse_args()

    root_out_dir = args.root_out_dir
    point_cloud_out_dir = root_out_dir + "/point_cloud"
    json_out_dir = root_out_dir + "/json_out"
    shape_type = args.shape_type
    set_small = args.set_small

    os.makedirs(root_out_dir, exist_ok=True)
    os.makedirs(point_cloud_out_dir, exist_ok=True)
    os.makedirs(json_out_dir, exist_ok=True)

    list = []
    canonical = args.canonical
    start = time.time()
    i_strat = args.i_start
    i_end = args.i_end
    total = i_end - i_strat
    i = i_strat
    i_last=i-1 #记录上一次的i
      
    while i < i_end:

        script, prompt, num_s, num_e = generate_and_export(i,canonical=canonical,type=shape_type,set_small=set_small)

        list.append({'id': i,
                     'script': script,
                     'prompt': prompt,
                     'num_start': num_s,
                     'num_end': num_e,
                     'pcd_path': f'{point_cloud_out_dir}/obj{i}.npz'})

        try:
            vertices, indices = get_faces()
            #sample_and_save_points(point_cloud_out_dir, f'obj{i}')
            sample_and_save_points(vertices, indices, point_cloud_out_dir, f'obj{i}')

        except Exception as e:
            print(e)
            with open('wrong.py', 'w') as file:
                file.write(script)
            list = list[:-1]
            continue  # 如果sample出错了，当前这个重新生成

        if i == i_strat + 0.15 * total - 1:
            json_data = json.dumps(list, indent=2)
            list = []
            with open(f'{json_out_dir}/val_{i_strat}_{i_end}.json', 'w') as file:
                file.write(json_data)
        elif i == i_strat + 0.3 * total - 1:
            json_data = json.dumps(list, indent=2)
            list = []
            with open(f'{json_out_dir}/test_{i_strat}_{i_end}.json', 'w') as file:
                file.write(json_data)

        i += 1

    json_data = json.dumps(list, indent=2)
    with open(f'{json_out_dir}/train_{i_strat}_{i_end}.json', 'w') as file:
        file.write(json_data)

    end = time.time()
    print(f"Total time cost:{end - start}.")