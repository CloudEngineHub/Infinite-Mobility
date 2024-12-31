import bpy
import bmesh
from scipy.spatial.transform import Rotation
import trimesh
import random
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from math import pi
import numpy as np
#import torch
#from bpy_lib import *

import threading, time

#向下保留两位小数
def floor_to_n(arr, n=2):
    arr = np.array(arr)
    factor = 10 ** n
    # 对于正数使用 floor，负数使用 ceil
    return np.where(arr >= 0, np.floor(arr * factor) / factor, np.ceil(arr * factor) / factor)

# 判断对象是否在bound内
def is_object_in_range(obj, bound=[-1, 1]):
    bpy.context.view_layer.update()  # 重新计算world_matrix保证同步
    world_matrix = obj.matrix_world
    for vertex in obj.data.vertices:
        # 获取顶点在世界坐标系中的坐标
        vertex_world = world_matrix @ vertex.co
        for coord in vertex_world:
            if coord < bound[0] or coord > bound[1]:
                return False
    return True


# 对生成好的mesh进行旋转变换，尺度变换，以及移至原点附近
def rotation_and_scale(name, is_rotation_object=False, with_rotation=False):
    if with_rotation:
        if is_rotation_object:
            rotate = [round(random.uniform(-1 , 1 ), 2) for _ in range(2)] + [0.00]
        else:
            rotate = [round(random.uniform(-1 , 1 ), 2) for _ in range(3)]
    else:
        rotate = None
        rotate_final = None
    if with_rotation:
        rotate_angle_1 = [angle * pi for angle in rotate]
        bpy.data.objects[name].rotation_euler = rotate_angle_1
        r_1 = Rotation.from_euler("xyz", rotate_angle_1, degrees=False)
        rotation_matrix_1 = r_1.as_matrix()
        vertices, indices = get_faces()
        mesh = trimesh.Trimesh(vertices.cpu(), indices.cpu())
        R, T = trimesh.bounds.oriented_bounds(mesh, ordered=False)
        R_final = R[:3, :3] @ rotation_matrix_1
        r = Rotation.from_matrix(R_final)
        rotate_angle_final = r.as_euler("xyz", degrees=False)
        rotate_final = rotate_angle_final / pi
        rotate_final = [round(value, 2) for value in rotate_final]
        bpy.data.objects[name].rotation_euler = [angle * pi for angle in rotate_final]

    # scaling
    x, y, z = calculate_bounding_box(bpy.data.objects[name], is_curve=True)
    x0, x1 = x
    y0, y1 = y
    z0, z1 = z
    bounding_x = x1 - x0
    bounding_y = y1 - y0
    bounding_z = z1 - z0
    max_bounding = max(bounding_x, bounding_y, bounding_z)
    max_scale = 2.0 / max_bounding
    #min_scale = 1.8 / max_bounding
    #scaling_factor = random.uniform(min_scale, max_scale)
    scaling_factor = max_scale
    scale = [scaling_factor for _ in range(3)]
    scale = [floor_to_n(s, 2) for s in scale]
    bpy.data.objects[name].scale = scale

    # centralize & randomly shift
    x, y, z = calculate_bounding_box(bpy.data.objects[name], is_curve=True)
    x0, x1 = x
    y0, y1 = y
    z0, z1 = z
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    center_z = (z0 + z1) / 2
    move = [-center_x, -center_y, -center_z]
    # x1 += -center_x
    # y1 += -center_y
    # z1 += -center_z
    # move[0] += random.uniform(1.0 - x1, x1 - 1.0)
    # move[1] += random.uniform(1.0 - y1, y1 - 1.0)
    # move[2] += random.uniform(1.0 - z1, z1 - 1.0)
    move = [round(l, 2) for l in move]
    bpy.data.objects[name].location = move

    return move, scaling_factor, rotate_final


def calculate_bounding_box(obj, is_curve=False):
    bpy.context.view_layer.update()  # 重新计算world_matrix保证同步
    world_matrix = obj.matrix_world
    # 初始化最小和最大坐标
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    if is_curve:
        obj_eval = obj.evaluated_get(depsgraph=bpy.context.evaluated_depsgraph_get())
        mesh = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=bpy.context.evaluated_depsgraph_get())
        for vertex in mesh.vertices:
            # 获取顶点在世界坐标系中的坐标
            vertex_world = world_matrix @ vertex.co

            # 更新最小和最大坐标值
            min_x = min(min_x, vertex_world.x)
            min_y = min(min_y, vertex_world.y)
            min_z = min(min_z, vertex_world.z)

            max_x = max(max_x, vertex_world.x)
            max_y = max(max_y, vertex_world.y)
            max_z = max(max_z, vertex_world.z)

    else:
        for vertex in obj.data.vertices:
            # 获取顶点在世界坐标系中的坐标
            vertex_world = world_matrix @ vertex.co

            # 更新最小和最大坐标值
            min_x = min(min_x, vertex_world.x)
            min_y = min(min_y, vertex_world.y)
            min_z = min(min_z, vertex_world.z)

            max_x = max(max_x, vertex_world.x)
            max_y = max(max_y, vertex_world.y)
            max_z = max(max_z, vertex_world.z)

    # 返回bounding box的最小和最大值
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)


def detect_overlapping(name1, name2):
    bpy.context.view_layer.update()

    # Get the objects
    obj1 = bpy.data.objects[name1]
    obj2 = bpy.data.objects[name2]

    # Get their world matrix
    mat1 = obj1.matrix_world
    mat2 = obj2.matrix_world

    # Get the geometry in world coordinates
    vert1 = [mat1 @ v.co for v in obj1.data.vertices]
    poly1 = [p.vertices for p in obj1.data.polygons]

    vert2 = [mat2 @ v.co for v in obj2.data.vertices]
    poly2 = [p.vertices for p in obj2.data.polygons]

    # Create the BVH trees
    bvh1 = BVHTree.FromPolygons(vert1, poly1)
    bvh2 = BVHTree.FromPolygons(vert2, poly2)

    # Test if overlap
    return bvh1.overlap(bvh2)


def get_volume(name):
    obj = bpy.data.objects[name]
    me = obj.data

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.transform(obj.matrix_world)
    bmesh.ops.triangulate(bm, faces=bm.faces)

    volume = 0
    for f in bm.faces:
        v1 = f.verts[0].co
        v2 = f.verts[1].co
        v3 = f.verts[2].co
        volume += v1.dot(v2.cross(v3)) / 6

    bm.free()
    return volume


def count_islands(name):
    obj = bpy.data.objects[name]

    obj.hide_set(False)

    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    return len(bpy.data.objects)


#生成随机数，按概率选择使用对数采样还是均匀采样，uniform_prob用来调控两种采样方式的比例，B是shape
def uniform_and_log_rand(B, uniform_prob, log_min, log_max):
    uniform = torch.rand(B)
    uniform=0.9 * uniform + 0.1
    logits = torch.rand(B) * (log_max-log_min) + log_min
    values = 10 ** (logits)
    # pdb.set_trace()
    samples = (torch.rand(B) < uniform_prob).float()
    output = samples * uniform + (1-samples) * values
    return output

#将数字规范到[-pi,pi]
def normalize_to_pi(value):
    while value < -pi:
        value += 2 * pi
    while value > pi:
        value -= 2 * pi
    return value


#将数字规范到[-1,1]
def normalize_to_one(value):
    while value < -1:
        value += 2.
    while value > 1:
        value -= 2.
    return value

import mathutils
def change_param_according_mesh(name, param,canonical=True,is_straight_line=False,set_small = True):
    obj = bpy.data.objects[name]
# set the new_obj as active object for later process
    bpy.context.view_layer.objects.active = obj
# make sure new_obj has single user copy
    bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)
    bpy.ops.object.modifier_add(type='TRIANGULATE')
    bpy.ops.object.modifier_apply(modifier="Triangulate")
    arr = np.zeros(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", arr)
    vertices = arr.reshape(-1, 3)
    arr = np.zeros(len(obj.data.polygons) * 3)
    obj.data.polygons.foreach_get("vertices", arr)
    faces = arr.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices, faces)
    bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    rot_mat=None
    center = np.array(((vertices[:, 0].max() + vertices[:, 0].min()) / 2, (vertices[:, 1].max() + vertices[:, 1].min()) / 2, (vertices[:, 2].max() + vertices[:, 2].min()) / 2))
    mesh.vertices -= center

    #canonical情况下需要将pose转换到标准姿态下
    if canonical and not is_straight_line:
        trans = np.linalg.inv(mesh.bounding_box_oriented.transform)
        tmp = np.eye(4)
        tmp[:3,:3]=trans[:3,:3]
        trans = tmp
        #trans = mesh.bounding_box_oriented.transform
        #修改参数
        
        if "rotation" in param.keys():
            rotation_quat = param["rotation"]
            quat = mathutils.Quaternion(rotation_quat)
            ori_matrix = quat.to_matrix()
            trans_matrix = np.array(trans)[:3,:3]

            #随机进行90度旋转并50%的概率添加高斯噪声
            noise = [random.gauss(0,5.0/180.0*pi) for _ in range(3)] if random.uniform(0,1)<0.5 else [0.,0.,0.] 
            #noise = [0, 0, 0]
            random_angle = [ noise[i]+ pi/2*random.sample([-1,0,1,2],k=1)[0] for i in range(3)]
            rotate_angle_final = [normalize_to_pi(value) for value in random_angle]
            eul = mathutils.Euler(rotate_angle_final, 'XYZ')
            random_matirx = np.array(eul.to_matrix())
            final_matrix = random_matirx@trans_matrix@ori_matrix
            final_matrix4 = np.eye(4)
            final_matrix4[:3,:3] = random_matirx@trans_matrix

            rotation_matrix = mathutils.Matrix(final_matrix)
            quat_final = np.array(rotation_matrix.to_quaternion())
            param["rotation"] = quat_final
            mesh.apply_transform(final_matrix4)
    # from infinigen.core.util import blender as butil
    # obj = butil.object_from_trimesh(mesh, f"{name}_mesh")
    # bpy.context.scene.collection.objects.link(obj)
    # bpy.context.view_layer.objects.active = obj

    #scaling
    x0, x1 = mesh.vertices[:,0].min(), mesh.vertices[:,0].max()
    y0, y1 = mesh.vertices[:,1].min(), mesh.vertices[:,1].max()
    z0, z1 = mesh.vertices[:,2].min(), mesh.vertices[:,2].max()
    bounding_x = x1 - x0
    bounding_y = y1 - y0
    bounding_z = z1 - z0
    max_bounding = max(bounding_x, bounding_y, bounding_z)
    max_scale = 2.0 / max_bounding
    min_scale = 1.8 / max_bounding
    
    #是否生成0.2-1的
    if set_small:
        min_scale = 0.4 / max_bounding

    if canonical and not set_small:
        scaling_factor = max_scale if random.uniform(0.,1.)<0.5 else random.uniform(min_scale, max_scale)
    else:
        scaling_factor = random.uniform(min_scale, max_scale)

    #scaling_factor=2.0 / max_bounding
    mesh.apply_scale(scaling_factor)
    # mesh.export('mesh.ply')
    # centralize & randomly shift
    x0, x1 = mesh.vertices[:,0].min(), mesh.vertices[:,0].max()
    y0, y1 = mesh.vertices[:,1].min(), mesh.vertices[:,1].max()
    z0, z1 = mesh.vertices[:,2].min(), mesh.vertices[:,2].max()
    bounding_x = x1 - x0
    bounding_y = y1 - y0
    bounding_z = z1 - z0
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    center_z = (z0 + z1) / 2
    move = -center
    length_semi = [x1-center_x , y1-center_y, z1-center_z]
    move_add = []
    for length in length_semi:
        free = 1.0-length
        free = 0
        if free<0.1:
            move_add.append(random.uniform(-free, free))
        else: 
            add_item = random.uniform(-0.1, 0.1) + random.gauss(0,(free-0.1)/3.0)
            if add_item>free:
                add_item = free
            elif add_item < -free:
                add_item = -free
            move_add.append(add_item)

    if canonical:
        move = move if random.uniform(0.,1.0)<0.5 else [move_item + move_add_item for move_item,move_add_item in zip(move,move_add)]
    else:
        move = [move_item + move_add_item for move_item,move_add_item in zip(move,move_add)]

    # param_name = ["x_tip", "thickness", "x_anchors", "y_anchors", "z_anchors"]
    # for item in param.keys():
    #     for p in param_name:
    #         if p == item:
    #             param[item]*=scaling_factor
    #             if "anchor" not in item:
    #                 param[item] = float(floor_to_n(param[item], 2))
    

    if "location" in param.keys():
        param["location"] = np.array(param["location"])
        param["location"]+=move
        param["location"] = floor_to_n(param["location"],2)
    if "thickness" in param.keys():
        param["thickness"] = float(floor_to_n(param["thickness"],3))
    if "rotation" in param.keys():
        param["rotation"] = floor_to_n(param["rotation"],2)
        print(param["rotation"])
    param["scale"] = [d * scaling_factor for d in param["scale"]]
    
    return param

#根据mesh对齐调整平移体的坐标并添加噪声
def change_bevelparam_according_radius(bevel_param,radius):
    scaling_factor = 1.0/radius[0]
    bevel_param_name = ["a","b","width","length","radius","points"]
    for item in bevel_param_name:
        if item in bevel_param.keys():
            bevel_param[item]/=scaling_factor
    radius = [r*scaling_factor for r in radius]

    return  bevel_param,radius



#根据控制点判断是否存在相交的情况
def is_self_intersecting(coordinates):
    ###   l1 [xa, ya, xb, yb]   l2 [xa, ya, xb, yb]
    def Intersect(l1, l2):
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        a = v0[0] * v1[1] - v0[1] * v1[0]
        b = v0[0] * v2[1] - v0[1] * v2[0]

        temp = l1
        l1 = l2
        l2 = temp
        v1 = (l1[0] - l2[0], l1[1] - l2[1])
        v2 = (l1[0] - l2[2], l1[1] - l2[3])
        v0 = (l1[0] - l1[2], l1[1] - l1[3])
        c = v0[0] * v1[1] - v0[1] * v1[0]
        d = v0[0] * v2[1] - v0[1] * v2[0]

        if a*b < 0 and c*d < 0:
            return True
        else:
            return False
        
    n = len(coordinates)
    if n<4:
        return False 
    # 对每一对相邻线段进行判断
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            l1 = coordinates[i][:2] + coordinates[i + 1][:2]
            l2 = coordinates[j][:2] + coordinates[j + 1][:2]
            if Intersect(l1,l2):
                return True
    return False


#对薄板平移体来说，要根据轨迹位姿计算出截面对应的旋转角
def cal_angle_BendingSheet(rot_mat,n):
    #计算得到曲线起始点的切线向量以及旋转之后的z轴
    z = np.array([0,0,1])
    #z轴在起始点切平面上的投影
    v1 = z - np.dot(n,z)*n

    #z轴垂直于切平面的情况
    #此时实际应用后的y轴方向是世界坐标系的y轴负方向
    if np.linalg.norm(v1)<1e-10:
        #旋转之后的y轴
        v1 = np.array([0,-1,0])
        v2 = np.array(rot_mat)@np.array([0,0,1]).T
        TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
        theta = np.arccos(np.clip(np.dot(v1, v2)/ TheNorm, -1., 1.))
        cross_product = np.cross(v1, v2)
        dot_product = np.dot(cross_product, n)
        if dot_product>0:
            theta = theta
        else:
            theta = -theta
    
    else:
        #旋转整体时，bevel的y轴跟世界坐标系z轴对齐，然后旋转，所以实际显示在旋转体上的y轴方向是世界坐标系的z轴旋转后的方向
        #而实际上应用后y轴的方向是世界坐标系的z轴方向在起始点切平面上的投影
        #旋转后的z轴
        v2 = np.array(rot_mat)@np.array([0,0,1]).T

        TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
        # 计算夹角大小
        #theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
        theta = np.arccos(np.clip(np.dot(v1, v2) / TheNorm, -1., 1.))
        #判断是顺时针还是逆时针
        cross_product = np.cross(v1, v2)
        dot_product = np.dot(cross_product, n)

        if dot_product>0:
            theta = theta
        else:
            theta = -theta
    
    if np.isnan(theta) or np.isinf(theta):
        raise ValueError("theta is invalid")
    
    #将theta规范到[-1,1]
    rad = theta/pi
    
    return rad

#更一般的情况，给定一个平移体的n，求出平移体按rot_mat旋转后bevel应该旋转的角度
def cal_angle_according_rotation(rot_mat,n):
    z = np.array([0,0,1])
    #求旋转前bevel的y轴正向
    v1 = z - np.dot(n,z)*n
    if np.linalg.norm(v1)<1e-10:
        v1=np.array([0,-1,0])
    #求旋转后的y轴正向
    n2=np.array(rot_mat)@n
    v2 = z - np.dot(n2,z)*n
    if np.linalg.norm(v2)<1e-10:
        v2=np.array([0,-1,0])
    #旋转前的y轴正向进行旋转
    v3 = np.array(rot_mat)@v1

    TheNorm = np.linalg.norm(v2) * np.linalg.norm(v3)

    theta = np.arccos(np.clip(np.dot(v2, v3) / TheNorm, -1., 1.))
    cross_product = np.cross(v2, v3)
    dot_product = np.dot(cross_product, n2)
    if dot_product>0:
        theta = theta
    else:
        theta = -theta
    
    if np.isnan(theta) or np.isinf(theta):
        raise ValueError("theta is invalid")
    
    #将theta规范到[-1,1]
    rad = theta/pi
    
    return rad

#获取curve的起始点切向量
def get_curve_tangent(control_points):
    coords = control_points

    curveData = bpy.data.curves.new("tan", type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 12

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(coords) - 1) 

    for i, coord in enumerate(coords):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'AUTO'
        bezier_point.handle_right_type = 'AUTO'

    curveOB = bpy.data.objects.new("tan", curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    p1 = bpy.data.objects['tan'].data.splines[0].bezier_points[0].co
    p2 = bpy.data.objects['tan'].data.splines[0].bezier_points[0].handle_right

    tan = (p1-p2)/np.linalg.norm(np.array(p1)-np.array(p2))
    bpy.data.objects.remove(bpy.data.objects["tan"], do_unlink=True)

    return tan

#获取circle的起始点切向量
def get_circle_tangent(r,location,rotation):
    bpy.ops.curve.simple(align='WORLD', location=[0,0,0], rotation=[0,0,0], Simple_Type='Circle',shape='3D',Simple_sides=4,Simple_radius=r, outputType='BEZIER', use_cyclic_u=True,edit_mode=False)
    curve = bpy.context.object
    curve.name = 'tan'
    curve.location = location
    curve.rotation_mode = "QUATERNION"
    curve.rotation_quaternion = rotation
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    p1 = bpy.data.objects['tan'].data.splines[0].bezier_points[0].co
    p2 = bpy.data.objects['tan'].data.splines[0].bezier_points[0].handle_right

    tan = (p1-p2)/np.linalg.norm(np.array(p1)-np.array(p2))
    bpy.data.objects.remove(bpy.data.objects["tan"], do_unlink=True)

    return tan


def sort_coordinates_curve(points):
    points = list(points)
    start = points[0]
    end = points[-1]

    need_reverse = False
    if start[0]>end[0]:
        need_reverse = True
    elif start[0]==end[0]:
        if start[1]>end[1]:
            need_reverse = True
        elif start[1]==end[1]:
            if start[2]>end[2]:
                need_reverse = True
    if need_reverse:
        points.reverse()
    points_new = np.array(points)
    min_x_index = np.argmin(points_new[:, 0])
    points_new-=points_new[min_x_index]

    return points_new



#随机生成折线bevel
def generate_random_polyline_bevel():
    index = random.randint(1, 4)
    if index == 1:  # 生成折线的控制点，该折线包括三个点
        points = []
        x, y, z = 0.0, 0.0, 0.0
        points.append([x, y, z])

        x = random.uniform(0.5, 1.5)  # 限制 x 坐标在一定范围内
        y = random.uniform(-1.0, 1.0)  # y 坐标随机
        points.append([x, y, z])

        # 第三个点，x 坐标继续递增，y 坐标随机
        x = random.uniform(1.5, 2.5)
        y = random.uniform(-1.0, 1.0)
        points.append([x, y, z])
    elif index == 2: # 弯曲的薄面，由四个控制点组成，第一第二个控制点与第三第四个控制点的距离远小于第二第三个控制点的距离，且对称
        points = []

        # 生成第二个和第三个控制点，保持它们之间的距离较大
        x2 = random.uniform(1.0, 2.0)
        y2 = random.uniform(1.0, 2.0)
        point2 = [x2, y2, 0.0]  # z 坐标设为 0

        x3 = x2 + random.uniform(5.0, 7.0)  # P2和P3之间的距离较大
        y3 = y2 + random.uniform(5.0, 7.0)
        point3 = [x3, y3, 0.0]  # z 坐标设为 0

        points.append(point2)
        points.append(point3)

        # 计算第二个和第三个控制点的中点
        midpoint = ((x2 + x3) / 2, (y2 + y3) / 2, 0.0)

        # 计算方向向量在 xoy 平面上的垂直向量 (-dy, dx)
        dx = x3 - x2
        dy = y3 - y2
        perp_dir = (-dy, dx, 0.0)

        # 归一化垂直方向向量，保证对称点计算时距离合适
        norm = math.sqrt(perp_dir[0] ** 2 + perp_dir[1] ** 2)
        unit_perp_dir = (perp_dir[0] / norm, perp_dir[1] / norm, 0.0)

        # P1 相对 P2 的左右偏移范围
        x_offset1 = random.uniform(-0.5, 0.5)  # x 方向的自由偏移
        y_offset1 = random.uniform(-0.5, 0.5)  # y 方向的自由偏移

        # 设定距离 d1，生成 P1，使其在 P2 附近
        d1 = random.uniform(0.3, 0.7)  # P1 与 P2 距离较小

        # 第一控制点，允许一定的偏移自由度
        point1 = (x2 - unit_perp_dir[0] * d1 + x_offset1,
                  y2 - unit_perp_dir[1] * d1 + y_offset1,
                  0.0)

        # 设定距离 d2，生成 P4，使其在 P3 附近
        # d2 = random.uniform(0.3, 0.7)  # P4 与 P3 距离较小
        x_offset2 = random.uniform(-0.5, 0.5)  # x 方向的自由偏移
        y_offset2 = random.uniform(-0.5, 0.5)  # y 方向的自由偏移

        # 第四控制点，与 P3 距离较近，在同一侧
        point4 = [x3 - unit_perp_dir[0] * d1 + x_offset2,
                  y3 - unit_perp_dir[1] * d1 + y_offset2,
                  0.0]

        points.insert(0, point1)  # 将第一个控制点插入到列表开头
        points.append(point4)  # 将第四个控制点加入到列表末尾
    elif index == 3:# 生成四个控制点的折线，要求p1p2与p2p3垂直，p2p3与p3p4垂直,p1p2的距离大于p3p4的距离,p1p2与p3p4位于p2p3的同一侧，z为0
        points = []

        # 生成第二个和第三个控制点，保持它们之间的距离较大且在水平线上
        x2 = random.uniform(1.0, 2.0)
        y2 = random.uniform(1.0, 2.0)
        point2 = [x2, y2, 0.0]  # z 坐标设为 0

        x3 = x2 + random.uniform(2.0, 3.0)  # P2 和 P3 之间的距离较大
        y3 = y2  # 保持在水平线上
        point3 = [x3, y3, 0.0]  # z 坐标设为 0

        points.append(point2)
        points.append(point3)

        # 生成第一个控制点，使得 P1P2 和 P2P3 垂直
        # P2P3 的方向向量为 (dx, dy) = (x3 - x2, 0)
        # P1P2 垂直于 P2P3，所以 P1 应该在 P2 的上下方
        d1 = random.uniform(2.0, 4.0)  # P1P2 的距离大于 P3P4
        point1 = [x2, y2 + d1, 0.0]  # P1 在 P2 的上方

        # 生成第四个控制点，使得 P3P4 和 P2P3 垂直
        # P2P3 的方向向量为 (dx, dy) = (x3 - x2, 0)
        # P3P4 垂直于 P2P3，所以 P4 应该在 P3 的上下方，但与 P1 位于同一侧
        d2 = random.uniform(0.5, 1.5)  # P3P4 的距离小于 P1P2
        point4 = [x3, y3 + d2, 0.0]  # P4 在 P3 的上方

        points.insert(0, point1)  # 将第一个控制点插入到列表开头
        points.append(point4)  # 将第四个控制点加入到列表末尾
    else:# 生成四个控制点的折线，要求p1p2与p2p3垂直，p2p3与p3p4成一定角度,p1p2的距离大于p3p4的距离,p1p2与p3p4位于p2p3的同一侧，z为0
        points = []

        # 生成第二个和第三个控制点，保持它们之间的距离较大
        x2 = random.uniform(1.0, 2.0)
        y2 = random.uniform(1.0, 2.0)
        point2 = [x2, y2, 0.0]  # z 坐标设为 0

        x3 = x2 + random.uniform(1.0, 2.0)  # P2 和 P3 之间的距离较大
        y3 = y2  # 保持在水平线上，P2P3 是水平线
        point3 = [x3, y3, 0.0]  # z 坐标设为 0

        points.append(point2)
        points.append(point3)

        # 生成第一个控制点，使得 P1P2 和 P2P3 垂直
        d1 = random.uniform(2.0, 4.0)  # P1P2 的距离较大
        point1 = [x2, y2 + d1, 0.0]  # P1 在 P2 的上方

        # 生成第四个控制点，使得 P3P4 和 P2P3 的法线夹角为 0 到 30 度
        angle = random.uniform(60, 120)  # 随机角度 0 到 30 度
        angle_rad = math.radians(angle)
        d2 = random.uniform(1.0, 2.0)  # P3P4 的距离小于 P1P2 的距离

        # 计算 P3 到 P4 的偏移量
        offset_x = math.cos(angle_rad) * d2
        offset_y = math.sin(angle_rad) * d2

        # 第四控制点，偏移方向与 P1P2 位于 P2P3 的同一侧
        point4 = [x3 + offset_x, y3 + offset_y, 0.0]

        # 确保 P1P2 和 P3P4 在 P2P3 的同一侧
        if (point1[1] - y2) * offset_y < 0:  # 如果不在同一侧，则反转 P4 的偏移量
            point4 = [x3 + offset_x, y3 - offset_y, 0.0]

        points.insert(0, point1)  # 将第一个控制点插入到列表开头
        points.append(point4)  # 将第四个控制点加入到列表末尾

    return points
    

def generate_random_closed_concat():
    index = random.randint(1, 6)
    # index = 1
    if index == 1:  # 生成折线的控制点，该折线包括三个点
        # 定义空列表存储点
        points = []

        # 第一个点固定为 (0.0, 0.0, 0.0)
        x, y, z = 0.0, 0.0, 0.0
        points.append((x, y, z))  # 使用元组形式存储点

        # 第二个点，x 坐标递增，y 坐标随机
        x = random.uniform(2.5, 3.5)  # 限制 x 坐标在一定范围内
        # y = random.uniform(0.0, 0.0)  # y 坐标随机
        points.append((x, 0.0, 0.0))  # 以元组形式存储

        # 第三个点，x 坐标继续递增，y 坐标随机
        y = random.uniform(1.5, 2.0)
        points.append((x, y, 0.0))  # 以元组形式存储

        # 正常的三个点结束

        p1_x_random = random.uniform(1.5, 1.8)
        p1_y_random = random.uniform(1.5, 1.8)
        p2_x_random = random.uniform(0.0, 0.1)
        p2_y_random = random.uniform(0.0, 0.1)
        points.append((0+p1_x_random, 0+p1_y_random, z))  # p0 handle_left
        points.append((x-p2_x_random, y+p2_y_random, z))  # p3 handle_right
    elif index == 2:
        # 定义空列表存储点
        points = []

        # 第一个点固定为 (0.0, 0.0, 0.0)
        x, y, z = 0.0, 0.0, 0.0
        points.append((x, y, z))  # 使用元组形式存储点

        # 第二个点，x 坐标递增，y 坐标随机
        x = random.uniform(2.5, 3.5)  # 限制 x 坐标在一定范围内
        # y = random.uniform(0.0, 0.0)  # y 坐标随机
        points.append((x, 0.0, 0.0))  # 以元组形式存储

        # 第三个点，x 坐标继续递增，y 坐标随机
        y = random.uniform(1.5, 2.5)
        points.append((x, y, 0.0))  # 以元组形式存储
        p1_x_random = random.uniform(0.5, 0.8)
        p1_y_random = random.uniform(0.5, 0.8)
        p2_x_random = random.uniform(1.0, 1.5)
        p2_y_random = random.uniform(1.0, 1.5)
        points.append((x-p2_x_random, y-p2_y_random, 0.0))  # p3 handle_right
    elif index == 3:
        # 定义空列表存储点
        points = []

        # 第一个点固定为 (0.0, 0.0, 0.0)
        x, y, z = 0.0, 0.0, 0.0
        points.append((x, y, z))  # 使用元组形式存储点

        # 第二个点，x 坐标递增，y 坐标随机
        x = random.uniform(2.5, 3.5)  # 限制 x 坐标在一定范围内
        # y = random.uniform(0.0, 0.0)  # y 坐标随机
        points.append((x, 0.0, 0.0))  # 以元组形式存储
        p1_x_random = random.uniform(0.5, 0.8)
        p1_y_random = random.uniform(0.5, 0.8)
        p2_x_random = random.uniform(1.0, 1.5)
        p2_y_random = random.uniform(1.0, 1.5)
        points.append((x - p2_x_random, y+ p2_y_random, z))
    elif index == 4:
        # 定义空列表存储点
        points = []

        # 第一个点固定为 (0.0, 0.0, 0.0)
        x, y, z = 0.0, 0.0, 0.0
        points.append((x, y, z))  # 使用元组形式存储点

        x = random.uniform(2.5, 3.5)  # 限制 x 坐标在一定范围内
        x2 = random.uniform(1.5, 2.5)  # 限制 x 坐标在一定范围内
        y = random.uniform(1.5, 2.5)
        y2 = random.uniform(2.5, 3.5)  # 限制 y 坐标在一定范围内


        # 第二个点，x 坐标递增，y 坐标随机
        # y = random.uniform(0.0, 0.0)  # y 坐标随机
        points.append((x, 0.0, 0.0))  # 以元组形式存储

        # 第三个点，x 坐标继续递增，y 坐标随机
        points.append((x, y, 0.0))  # 以元组形式存储

        # 第四个点，，y 坐标随
        points.append((x2, y2, 0.0))  # 以元组形式存储
        points.append((0, y, 0.0))  # 以元组形式存储
    elif index == 5:
        # 定义空列表存储点
        points = []

        # 第一个点固定为 (0.0, 0.0, 0.0)
        # x, y, z = 0.0, 0.0, 0.0

        x = random.uniform(2.5, 3.5)  # 限制 x 坐标在一定范围内
        y = random.uniform(1.5, 2.5)

        points.append((0.0, 0.0, 0.0))  # 使用元组形式存储点

        # 第二个点，x 坐标递增，y 坐标随机
        # y = random.uniform(0.0, 0.0)  # y 坐标随机
        points.append((x, 0.0, 0.0))  # 以元组形式存储

        # 第三个点，x 坐标继续递增，y 坐标随机
        points.append((x, y, 0.0))  # 以元组形式存储
        points.append((0, y, 0.0))  # 以元组形式存储
        p1_x_random = random.uniform(1.2, 1.8)
        p1_y_random = random.uniform(1.0, 1.5)
        p2_x_random = random.uniform(0.5, 1.0)
        p2_y_random = random.uniform(0.5, 1.0)
        points.append((0+p1_x_random, y+p1_y_random, 0.0) ) # p0 handle_left
        points.append((x-p2_x_random, y+p2_y_random, 0.0)  )# p0 handle_left
    elif index == 6:
        # 定义空列表存储点
        points = []

        # 第一个点固定为 (0.0, 0.0, 0.0)
        # x, y, z = 0.0, 0.0, 0.0

        x = random.uniform(3.5, 4.5)  # 限制 x 坐标在一定范围内
        y = random.uniform(1.0, 1.5)

        points.append((0.0, 0.0, 0.0))  # 使用元组形式存储点


        # 第二个点，x 坐标递增，y 坐标随机
        # y = random.uniform(0.0, 0.0)  # y 坐标随机
        points.append((x, 0.0, 0.0))  # 以元组形式存储

        # 第三个点，x 坐标继续递增，y 坐标随机
        points.append((x, y, 0.0))  # 以元组形式存储
        points.append((0, y, 0.0))  # 以元组形式存储
        p1_x_random = random.uniform(1.2, 1.8)
        p1_y_random = random.uniform(1.0, 1.5)
        p2_x_random = random.uniform(0.5, 1.0)
        p2_y_random = random.uniform(0.5, 1.0)
        points.append((0+p1_x_random, y+p1_y_random, 0.0))
        points.append((x-p2_x_random, y+p2_y_random, 0.0))


    return points, index