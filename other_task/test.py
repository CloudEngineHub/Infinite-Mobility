from inspect import signature
import bpy
import addon_utils
addon_utils.enable("add_curve_extra_objects")
import copy
from math import radians, pi
import mathutils
import numpy as np
import bmesh
import math
from utils import *
#import torch


"""
Reorder the coordinates so that the xyz is prioritized first, with the lowest x first
"""

def sort_coordinates(coords):
    min_index = 0
    
    # find the min coord
    coords = list(coords)
    for i in range(1, len(coords)):
        if coords[i][0] < coords[min_index][0]:
            min_index = i
        elif coords[i][0] == coords[min_index][0]:
            if coords[i][1] < coords[min_index][1]:
                min_index = i
            elif coords[i][1] == coords[min_index][1]:
                if coords[i][2] < coords[min_index][2]:
                    min_index = i
    list_new = coords[min_index:] + coords[:min_index]
    points_new = np.array(list_new)
    
    return points_new
 
"""
Reorder the coordinates for non-closed graphics so that the xyz is prioritized first, with the lowest x first
"""
def sort_coordinates_endpoint(points):
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

    return points_new
"""
fill caps to mesh
"""
def make_caps(name,type="start"):
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'EDIT')

    object = bpy.data.objects[name]
    mesh = bmesh.from_edit_mesh(object.data) 
    mesh.verts.ensure_lookup_table()
    
    
    if type=="start":
        for i in range(len(mesh.verts)):
            if mesh.verts[i].is_boundary and i<len(mesh.verts)/2:
                mesh.verts[i].select=True
        bpy.ops.mesh.edge_face_add()
    elif type=="end":
        for i in range(len(mesh.verts)):
            if mesh.verts[i].is_boundary and i>len(mesh.verts)/2:
                mesh.verts[i].select=True
        bpy.ops.mesh.edge_face_add()
    
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'OBJECT')



def find_circle(points):
    """
    Calculate the center (cx, cy) and radius r of the circle passing through three points.
    Returns (cx, cy, r).
    """
    x1,y1 = points[0][0],points[0][1]
    x2,y2 = points[1][0],points[1][1]
    x3,y3 = points[2][0],points[2][1]
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    cx = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / (D+1e-10)
    cy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / (D+1e-10)
    r = math.sqrt((cx - x1)**2 + (cy - y1)**2)
    return cx, cy, r

def calculate_angle(x1, y1, x2, y2):
    """
    Calculate the angle in radians between the positive x-axis and the line from (x1, y1) to (x2, y2).
    """
    dx = x2 - x1
    dy = y2 - y1
    return math.atan2(dy, dx)

"""
Delete all objects in the scene
"""
def delete_all():
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
    except:
        bpy.ops.wm.read_homefile()
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
"""
Create primitive object
"""
def create_primitive(name, primitive_type="cube", location=None, scale=None, rotation=None, rotation_mode='XYZ', **kwargs):
    getattr(bpy.ops.mesh, f"primitive_{primitive_type}_add")(**kwargs)
    primitive = bpy.context.object
    primitive.name = name
    if location:
        primitive.location = location
    if scale:
        primitive.scale = scale
    if rotation:
        if rotation_mode=='XYZ':
            primitive.rotation_euler = [angle * pi for angle in rotation]
        elif rotation_mode=='QUATERNION':
            primitive.rotation_mode = 'QUATERNION'
            primitive.rotation_quaternion = rotation
        elif rotation_mode=='MATRIX':
            mat = np.eye(4)
            rotation = np.array(rotation).reshape([3,3])
            mat[:3,:3] = rotation
            bpy.context.view_layer.update()
            world_matrix = torch.tensor(bpy.data.objects[name].matrix_world)
            scale_now = world_matrix.norm(dim=0)[:3]
            scale_matrix = torch.eye(4)
            scale_matrix[0,0],scale_matrix[1,1],scale_matrix[2,2] =scale_now[0],scale_now[1],scale_now[2]
            scale_matrix_inv = scale_matrix.clone()
            for i in range(3):
                #对角矩阵取逆，如果对角线上元素为0，则还保持为0
                if scale_matrix_inv[i,i]>1e-10:
                    scale_matrix_inv[i,i]=1.0 / scale_matrix_inv[i,i]
            mat = scale_matrix_inv@torch.tensor(mat,dtype=torch.float32)@scale_matrix
            mat = mathutils.Matrix(np.array(mat))
            bpy.data.objects[name].matrix_world = bpy.data.objects[name].matrix_world@mat

    return primitive

"""
Perform a Boolean operation on both objects and delete the second object
"""
def boolean_operation(name1, name2, operation="UNION"):
    bpy.data.objects[name1].modifiers.new("Boolean", "BOOLEAN")
    bpy.data.objects[name1].modifiers["Boolean"].object = bpy.data.objects[name2]
    bpy.data.objects[name1].modifiers["Boolean"].operation = operation
    bpy.context.view_layer.objects.active = bpy.data.objects[name1]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    bpy.data.objects.remove(bpy.data.objects[name2], do_unlink=True)
    

"""
Create a BEZIER curve
"""
def create_curve(name, control_points=[], resolution=12):
    coords = control_points

    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(coords) - 1) 

    for i, coord in enumerate(coords):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'AUTO'
        bezier_point.handle_right_type = 'AUTO'

    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    points_sorted = sort_coordinates(control_points)
    
    return {"name":name, "points":points_sorted - points_sorted[0]}



"""
Creates a U-shaped (opening up) Bezier curve
"""
def create_symmetric_curve(name, control_points=[], resolution=12):
    copy.deepcopy(control_points)
    coords = []
    for point in control_points[::-1]:
        x, y, z = point
        coords.append((-x, y, z))
    for point in control_points[1:]:
        coords.append(point)

    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(coords) - 1) 

    for i, coord in enumerate(coords):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'AUTO'
        bezier_point.handle_right_type = 'AUTO'

    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    
    return curveOB


"""
Creates a symmetrically closed Bezier curve
"""
def create_closed_sym_curve(name, control_points=[], resolution=12):
    coords = copy.deepcopy(control_points)
    for points in control_points[1:][::-1]:
        x, y, z = points
        coords.append((-x, y, z))

    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(coords) - 1) 
    bezierSpline.use_cyclic_u = True

    for i, coord in enumerate(coords):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'AUTO'
        bezier_point.handle_right_type = 'AUTO'

    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    
    return curveOB

"""
Apply rotation to curve
"""
def curve_rotation(curve_name, radius=0.3, move=None, rotate=None):
    ng = bpy.data.node_groups.new('nodeGroupRotation', 'GeometryNodeTree')

    inNode = ng.nodes.new('NodeGroupInput')
    outNode = ng.nodes.new('NodeGroupOutput')
    c2mNode = ng.nodes.new('GeometryNodeCurveToMesh')
    transNode = ng.nodes.new('GeometryNodeTransform')
    circNode = ng.nodes.new('GeometryNodeCurvePrimitiveCircle')
    realNode = ng.nodes.new('GeometryNodeRealizeInstances')

    ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')


    ng.links.new(circNode.outputs['Curve'], c2mNode.inputs['Curve'])
    ng.links.new(inNode.outputs['Geometry'], transNode.inputs['Geometry'])
    ng.links.new(transNode.outputs['Geometry'], c2mNode.inputs['Profile Curve'])
    ng.links.new(c2mNode.outputs['Mesh'], realNode.inputs['Geometry'])
    ng.links.new(realNode.outputs['Geometry'], outNode.inputs['Geometry'])

    circNode.inputs['Radius'].default_value = radius if radius > 0 else 1
    transNode.inputs['Translation'].default_value = (0 if radius > 0 else -1, 0, 0)
    transNode.inputs['Rotation'].default_value = (pi * 0.5, 0, 0)

    modifier = bpy.data.objects[curve_name].modifiers.new('nodeGroupRotation', "NODES")
    modifier.node_group = ng

    if rotate!= None:
        bpy.data.objects[curve_name].rotation_euler = [angle * pi for angle in rotate]
    if move != None:
        bpy.data.objects[curve_name].location = move
    


"""
Translation along curve
"""
def along_curve_translation(curve_name, profile_radius=0.3, fill_caps=False,move=None, rotation=None):
    ng = bpy.data.node_groups.new('nodeGroupTranslation', 'GeometryNodeTree')

    inNode   = ng.nodes.new('NodeGroupInput')
    outNode  = ng.nodes.new('NodeGroupOutput')
    c2mNode  = ng.nodes.new('GeometryNodeCurveToMesh')
    circNode = ng.nodes.new('GeometryNodeCurvePrimitiveCircle')

    ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    ng.links.new(circNode.outputs['Curve'], c2mNode.inputs['Profile Curve'])
    ng.links.new(inNode.outputs['Geometry'], c2mNode.inputs['Curve'])
    ng.links.new(c2mNode.outputs['Mesh'], outNode.inputs['Geometry'])

    c2mNode.inputs['Fill Caps'].default_value = fill_caps
    circNode.inputs['Radius'].default_value = profile_radius

    modifier = bpy.data.objects[curve_name].modifiers.new('nodeGroupRotation', "NODES")
    modifier.node_group = ng

    if rotation!= None:
        bpy.data.objects[curve_name].rotation_euler = [angle * pi for angle in rotation]
    if move != None:
        bpy.data.objects[curve_name].location = move
    
    
"""
Add thickness to the face
"""
def solidify(name, thickness):
    bpy.data.objects[name].modifiers.new("Solidify", "SOLIDIFY")
    bpy.data.objects[name].modifiers["Solidify"].thickness = thickness
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.modifier_apply(modifier="Solidify")

"""
Change the center of the object
"""
def change_origin(name, location):
    bpy.context.scene.cursor.location = location
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')

"""
Extract the mesh points and polygons from the model generated by Blender
"""
def get_faces():
    #print("faces:",bpy.data.objects)
    names = bpy.context.scene.objects.keys()

    def get_one_obj_faces(name):  
        obj = bpy.data.objects[name]
        obj_eval = obj.evaluated_get(depsgraph=bpy.context.evaluated_depsgraph_get())
        mesh = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=bpy.context.evaluated_depsgraph_get())
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(obj.matrix_world)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.faces.ensure_lookup_table()

        vertices = []
        indices = []

        for f in bm.faces:
            indices.append([v.index for v in f.verts])
        for v in bm.verts:
            vertices.append(v.co)

        indices = np.array(indices)
        vertices = np.array(vertices)

        bm.free()

        return vertices, indices

    verts_all, faces_all= [], []
    cnt=0
    for name in names:
        verts,faces=get_one_obj_faces(name)
        verts_all+=verts.tolist()
        faces+=cnt
        faces_all+=faces.tolist()
        cnt+=len(verts)

    verts_all, faces_all = torch.tensor(np.array(verts_all),dtype=torch.float32).cuda(), torch.tensor(np.array(faces_all)).cuda()

    return verts_all,faces_all

    

"""
Create a rectangle with coordinates
"""
def create_rectangle_by_points(name,points,type = 'rectangle_points'):
    
    points = np.array(points)
    cx,cy = points[:,0].mean(), points[:,1].mean()
    width = 2*np.abs(cx - points[0][0])
    length = 2*np.abs(cy - points[0][1])
    
    
    if type == "rectangle_points":
        bpy.ops.curve.simple(align='WORLD',location=(cx, cy, 0),Simple_Type='Rectangle', Simple_width=width,shape='3D', Simple_length=length, use_cyclic_u=True)
    elif type == "rectangle_open_points":
        bpy.ops.curve.simple(align='WORLD',location=(cx, cy, 0),Simple_Type='Rectangle', Simple_width=width,shape='3D', Simple_length=length, use_cyclic_u=False)
     
    curve = bpy.context.object
    curve.name = name


"""
The arc turns to the Bézier control point, and the arc angle is an acute angle
"""
def arc2bezier(start_angle,end_angle,r,cx,cy):
    start_angle = start_angle*pi/180
    end_angle = end_angle*pi/180
    x0 = cx + np.cos(start_angle)*r
    y0 = cy + np.sin(start_angle)*r
    x3 = cx + np.cos(end_angle)*r
    y3 = cy + np.sin(end_angle)*r
    arc_angle = end_angle-start_angle
    a = 4*np.tan(arc_angle/4)/3
    x1 = x0 - a*(y0 - cy)
    y1 = y0 + a*(x0 - cx)
    x2 = x3 + a*(y3 - cy)
    y2 = y3 - a*(x3 - cx)
    
    return [x0,y0,0],[x1,y1,0],[x2,y2,0],[x3,y3,0]
    
"""
Return to the Bézier control point at any angle
"""
def get_bezier_points_by_angles(start_angle,end_angle,r,cx,cy):
    
    if start_angle > end_angle:
        tmp = start_angle
        start_angle = end_angle
        end_angle = tmp
    add_angle = end_angle - start_angle
    points_list = []
    if add_angle>90:
        times = int(np.ceil(add_angle/90))
        angle_delta = add_angle/times
        for i in range(times):
            if i != times-1:
                s_an = start_angle + i*angle_delta
                e_an = start_angle + (i+1)*angle_delta
                points0,points1,points2,points3 = arc2bezier(s_an,e_an,r,cx,cy)
                
                if i==0:
                    points_list+=[points0,points1,points2,points3]
                else:
                    points_list+=[points1,points2,points3]
            else:
                s_an = start_angle + i*angle_delta
                e_an = end_angle
                points0,points1,points2,points3 = arc2bezier(s_an,e_an,r,cx,cy)
                points_list+=[points1,points2,points3]
    else:
        points0,points1,points2,points3 = arc2bezier(start_angle,end_angle,r,cx,cy)
        points_list+=[points0,points1,points2,points3]
                    
    
    return points_list   



"""
Draw a Bezier arc based on the three 3D coordinates on the arc
"""
def create_bezier_arc_by_3Dpoints(name,points,center="POINT",points_radius=[1.0,1.0],closed=False):
    
    points = copy.deepcopy(points)
    p1, p2, p3 = np.array(points[0]), np.array(points[1]), np.array(points[2])
    n = np.cross((p1-p2),(p1-p3))
    x_axis = p1-p2
    y_axis = np.cross((p1-p2),n)
    z_axis = np.cross(x_axis,y_axis)
    if np.linalg.norm(x_axis)<1e-4:
        x_axis*=1e5
    if np.linalg.norm(y_axis)<1e-4:
        y_axis*=1e5
    if np.linalg.norm(z_axis)<1e-4:
        z_axis*=1e5
    x_axis/=np.linalg.norm(x_axis)
    y_axis/=np.linalg.norm(y_axis)
    z_axis/=np.linalg.norm(z_axis)
    
    point3D_to_xyplane=np.array([x_axis,y_axis,z_axis])
    try:
        xyplane_to_point3D = np.linalg.inv(point3D_to_xyplane)
    except:
        raise ValueError("Three-point collinear")

    p1_xy = point3D_to_xyplane@np.array(p1)
    p2_xy = point3D_to_xyplane@np.array(p2)
    p3_xy = point3D_to_xyplane@np.array(p3)
    
    points = np.array([p1_xy,p2_xy,p3_xy])
    points_z = points[0][2]
    points_2D = points[:,:2]
    
    cx,cy,r = find_circle(points_2D)
        
    angle_start = calculate_angle(cx, cy, points[0][0], points[0][1])
    angle_end = calculate_angle(cx, cy, points[2][0], points[2][1])

    
    # Calculate angles in degrees
    angle_start = math.degrees(angle_start) if angle_start>0 else math.degrees(angle_start) + 360
    angle_end = math.degrees(angle_end) if angle_end>0 else math.degrees(angle_end) + 360

    
    control_points = np.array(get_bezier_points_by_angles(angle_start,angle_end,r,cx,cy))
    control_points[:,2] = points_z 
    control_points = xyplane_to_point3D@np.array(control_points).T
    control_points = control_points.T
    if np.isnan(control_points).any() or np.isinf(control_points).any():
        raise ValueError("points can't form an arc")
    
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'

    bezierSpline = curveData.splines.new('BEZIER')

    num_cycle = int(np.ceil(len(control_points)/3))
    bezierSpline.bezier_points.add(num_cycle-1)
    
    #生成所有点的radius
    points_radius= [points_radius[0] + i * (points_radius[1] - points_radius[0]) / (num_cycle - 1) for i in range(num_cycle)]

    
    for i in range(num_cycle):
        bezierSpline.bezier_points[i].co = control_points[i*3]
        bezierSpline.bezier_points[i].radius = points_radius[i]
        #最后一个周期中只有一个点
        if i<num_cycle-1:
            bezierSpline.bezier_points[i].handle_right = control_points[i*3+1]
            bezierSpline.bezier_points[i+1].handle_left = control_points[i*3+2]
                
    if closed:
        bezierSpline.use_cyclic_u = True
    curveOB = bpy.data.objects.new(name, curveData)
    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    if center=="MEDIAN":
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    
"""
Draw a Bézier circle based on three 3D coordinates on the circle
"""
def create_bezier_circle_by_3Dpoints(name,points):
    
    p1, p2, p3 = np.array(points[0]), np.array(points[1]), np.array(points[2])
    n = np.cross((p1-p2),(p1-p3))
    x_axis = (p1-p2)
    y_axis = np.cross((p1-p2),n)
    z_axis = np.cross(x_axis,y_axis)
    x_axis/=np.linalg.norm(x_axis)
    y_axis/=np.linalg.norm(y_axis)
    z_axis/=np.linalg.norm(z_axis)
    
    point3D_to_xyplane=np.array([x_axis,y_axis,z_axis])
    xyplane_to_point3D = point3D_to_xyplane.T

    p1_xy = point3D_to_xyplane@np.array(p1)
    p2_xy = point3D_to_xyplane@np.array(p2)
    p3_xy = point3D_to_xyplane@np.array(p3)
    
    points = np.array([p1_xy,p2_xy,p3_xy])
    
    points_z = points[0][2]
    points_2D = points[:,:2]
    
    cx,cy,r = find_circle(points_2D)
    points = create_circle_points(r, cx,cy)
    
    points[:,2] = points_z
        
    curve = bpy.context.object
    curve.name = name
    splines = curve.data.splines[0]
    
    curve.data.dimensions = '3D'
    
    for i in range(4):
        splines.bezier_points[i].co = xyplane_to_point3D@points[3*i].T
        splines.bezier_points[i].handle_right = xyplane_to_point3D@points[3*i+2].T
        splines.bezier_points[i].handle_left = xyplane_to_point3D@points[3*i+1].T


"""
Creates an arc and returns three control points
"""
def create_arc_points(name, radius, start_angle, end_angle,location=[0,0,0],rotation=[0,0,0],center="POINT",sort_point=False ):
    bpy.ops.curve.simple(align='WORLD', location=location, rotation=rotation, Simple_Type='Arc',Simple_sides=3,Simple_radius=radius,Simple_startangle=start_angle, Simple_endangle=end_angle, outputType='BEZIER', use_cyclic_u=False)
    curve = bpy.context.object
    curve.name = name
    curveData = curve.data
    
    curveData.dimensions = '3D'
    mat = np.array(bpy.data.objects[name].matrix_world)

    front = mat @ np.array(list(curveData.splines[0].bezier_points[0].co) + [1])
    middle = mat @ np.array(list(curveData.splines[0].bezier_points[2].co) + [1])
    end = mat @ np.array(list(curveData.splines[0].bezier_points[4].co) + [1])
    
    points = np.array([front[:3],middle[:3],end[:3]])
    if sort_point:
        points = sort_coordinates(points)
    
    return {"name":name, "points":points - points[0],"center":center}

"""
Creates a rectangle and returns four control points
"""
def create_rectangle_points(name, width, length, location=[0,0,0],rotation=[0,0,0],closed=True,sort_point=False):
    bpy.ops.curve.simple(align='WORLD',location=location, rotation=rotation,Simple_Type='Rectangle', Simple_width=width,shape='3D', Simple_length=length, use_cyclic_u=closed)
    curve = bpy.context.object
    curve.name = name
    curveData = curve.data
    
    curveData.dimensions = '3D'
    mat = np.array(bpy.data.objects[name].matrix_world)

    points = []
    for i in range(4):
        point = mat @ np.array(list(curveData.splines[0].bezier_points[i].co) + [1])
        points.append(point[:3])
    points = np.array(points)
    if sort_point:
        points = sort_coordinates(points)
    
    return {"name":name, "points":points - points[0]}

"""
Creates a circle and returns three control points
"""
def create_circle_points(name, radius, location=[0,0,0],rotation=[0,0,0],sort_point=False ):
    bpy.ops.curve.simple(align='WORLD', location=location, rotation=rotation, Simple_Type='Circle',shape='3D',Simple_sides=6,Simple_radius=radius, outputType='BEZIER', use_cyclic_u=True)
    curve = bpy.context.object
    curve.name = name
    curveData = curve.data
    
    curveData.dimensions = '3D'
    mat = np.array(bpy.data.objects[name].matrix_world)

    front = mat @ np.array(list(curveData.splines[0].bezier_points[0].co) + [1])
    middle = mat @ np.array(list(curveData.splines[0].bezier_points[2].co) + [1])
    end = mat @ np.array(list(curveData.splines[0].bezier_points[4].co) + [1])
    
    points = [front[:3],middle[:3],end[:3]]
    points = np.array(points)
    if sort_point:
        points = sort_coordinates(points)
    
    return {"name":name, "points":points - points[0]}

"""
create an circle and return radius
"""
def create_circle(name, radius,center="MEDIAN"):
    bpy.ops.curve.simple(align='WORLD', location=[0,0,0], rotation=[0,0,0], Simple_Type='Circle',shape='3D',Simple_sides=4,Simple_radius=radius, outputType='BEZIER', use_cyclic_u=True,edit_mode=False)
    curve = bpy.context.object
    curve.name = name
    if center=="POINT":
        curve.location = [radius,0,0]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    curveData = curve.data
    
    curveData.dimensions = '3D'
        
    return {"name":name, "radius":radius, "center":center}

"""
create a rectangle and return param
"""
def create_rectangle(name, width, length, rotation_rad=0,center="MEDIAN",closed=True):
    bpy.ops.curve.simple(align='WORLD',location=[0,0,0], rotation=[0,0,0],Simple_Type='Rectangle', Simple_width=width,shape='3D', Simple_length=length, use_cyclic_u=closed,edit_mode=False)
    curve = bpy.context.object
    curve.name = name
    curveData = curve.data
    
    curveData.dimensions = '3D'
    if center=="POINT":
        curve.location=[width/2,length/2,0]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    curve.rotation_euler=[0,0,rotation_rad*pi]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    return {"name":name, "width":width,"length":length,"rotation_rad":rotation_rad,"center":center}


"""
create an oval and return param
"""
def create_oval(name, a,b,rotation_rad=0,center="MEDIAN"):
    bpy.ops.curve.simple(align='WORLD', location=[0,0,0], rotation=[0,0,0], Simple_Type='Circle',shape='3D',Simple_sides=4,Simple_radius=1.0, outputType='BEZIER', use_cyclic_u=True,edit_mode=False)
    curve = bpy.context.object
    curve.name = name
    if center=="POINT":
        curve.location = [1,0,0]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    curve.scale = [a,b,0]
    curve.rotation_euler=[0,0,rotation_rad*pi]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return {"name":name, "a":a,"b":b,"rotation_rad":rotation_rad,"center":center} 


"""
Creates a generic quad and returns four control points
"""
def create_quad(name, control_points,center="POINT", resolution=12,sort_point=False):
    
    control_points = copy.deepcopy(control_points)
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(control_points) - 1) 
    bezierSpline.use_cyclic_u = True

    for i, coord in enumerate(control_points):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'VECTOR'
        bezier_point.handle_right_type = 'VECTOR'

    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    if center=="MEDIAN":
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    points = np.array(control_points)
    if sort_point:
        points = np.array(sort_coordinates(points))    
    
    return {"name":name,"points":points - points[0],"center":center}
    #return {"name":name,"points":points_sorted - points_sorted[0]}

"""
Creates a closed curve and returns four control points
"""
def create_closed_curve(name, control_points,center="POINT", resolution=12,sort_point=False):
    
    control_points = copy.deepcopy(control_points)
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(control_points) - 1) 
    bezierSpline.use_cyclic_u = True

    for i, coord in enumerate(control_points):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'AUTO'
        bezier_point.handle_right_type = 'AUTO'

    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    if center=="MEDIAN":
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

    points = np.array(control_points)
    if sort_point:
        points = np.array(sort_coordinates(control_points))    
    
    return {"name":name,"points":points - points[0],"center":center}


"""
create a square
"""
def create_square(name,length,rotation_rad=0,center="MEDIAN"):
    bpy.ops.curve.simple(align='WORLD',location=(0,0,0), rotation=(0,0,0),Simple_Type='Rectangle', Simple_width=length,shape='3D', Simple_length=length, use_cyclic_u=True,edit_mode=False)
    curve = bpy.context.object
    curve.name = name
    curveData = curve.data
    curveData.dimensions = '3D'
    if center=="POINT":
        curve.location=[length/2,length/2,0]
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    curve.rotation_euler=[0,0,rotation_rad*pi]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    return {"name":name,"length":length,"rotation_rad":rotation_rad,"center":center}

"""
创建直线底碗形状的bevel
"""
def create_linebowl_curve(name, control_points=[], resolution=12):
    coords = copy.deepcopy(control_points)
    
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution
    
    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(coords) - 1) 

    for i, coord in enumerate(coords):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'AUTO'
        bezier_point.handle_right_type = 'AUTO'
    
    bezierSpline.bezier_points[1].handle_left_type="VECTOR"
    bezierSpline.bezier_points[1].handle_right_type="ALIGNED"
    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    
    return {"name":name, "points":control_points}

"""
创建碗形状的bevel
"""
def create_bowl_curve(name, control_points=[], resolution=12):
    coords = copy.deepcopy(control_points)
    handle = coords[1]
    try:
        coords.pop(1)
    except:
        coords = coords.tolist()
        coords.pop(1)
    
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution
    
    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(coords) - 1) 

    for i, coord in enumerate(coords):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'AUTO'
        if i==0:
            bezier_point.handle_right_type = 'FREE'
        else:
            bezier_point.handle_right_type = 'AUTO'
    
    bezierSpline.bezier_points[0].handle_right=handle
    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    
    return {"name":name, "points":control_points}

"""
创建瓶颈形状的curve，即底部为一个竖直线封闭的形状
"""
def create_bottleneck_curve(name, control_points=[], resolution=12):
    coords = copy.deepcopy(control_points)
        
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution
    
    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(coords) - 1) 

    for i, coord in enumerate(coords):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        if i<2:
            bezier_point.handle_left_type = 'VECTOR'
            bezier_point.handle_right_type = 'VECTOR'
        else:
            bezier_point.handle_left_type = 'AUTO'
            bezier_point.handle_right_type = 'AUTO'
    
    bezierSpline.bezier_points[2].handle_left_type="VECTOR"
    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    
    return {"name":name, "points":control_points}

"""
创建蘑菇形状的bevel
"""
def create_mushroom_curve(name, control_points=[], resolution=12):
    coords = copy.deepcopy(control_points)
        
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution
    
    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(3)
    bezierSpline.bezier_points[0].co = coords[0]
    bezierSpline.bezier_points[0].handle_right = coords[1]
    bezierSpline.bezier_points[1].handle_left = coords[2]
    bezierSpline.bezier_points[1].co = coords[3]
        
    
    for i, coord in enumerate(coords[4:]):
        i+=2
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        
        bezier_point.handle_left_type = 'VECTOR'
        bezier_point.handle_right_type = 'VECTOR'
    
    bezierSpline.bezier_points[1].handle_right_type="VECTOR"
    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    
    return {"name":name, "points":control_points}

"""
创建陶罐形状的bevel
"""
def create_pot_curve(name, control_points=[], resolution=12):
    coords = copy.deepcopy(control_points)
        
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution
    
    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(3)
    bezierSpline.bezier_points[0].co = coords[0]
    bezierSpline.bezier_points[0].handle_right_type="VECTOR"
    bezierSpline.bezier_points[1].handle_left_type="VECTOR"
    bezierSpline.bezier_points[1].co = coords[1]
    bezierSpline.bezier_points[1].handle_right=coords[2]
    bezierSpline.bezier_points[2].handle_left=coords[3]
    bezierSpline.bezier_points[2].co=coords[4]
    bezierSpline.bezier_points[2].handle_right_type="VECTOR"
    bezierSpline.bezier_points[3].handle_left_type="VECTOR"
    bezierSpline.bezier_points[3].co=coords[5]
    
    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    
    return {"name":name, "points":control_points}
"""
创建折线形状的bevel
"""
def create_polyline(name,control_points=[],resolution=12,sort_point=False):
    control_points=copy.deepcopy(control_points)
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(control_points) - 1)
    # bezierSpline.use_cyclic_u = True

    for i, coord in enumerate(control_points):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'VECTOR'
        bezier_point.handle_right_type = 'VECTOR'

    curveOB = bpy.data.objects.new(name, curveData)

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    points = np.array(control_points)
    if sort_point:
        points = np.array(sort_coordinates_endpoint(control_points))    

    return {"name":name,"points":np.array(points) - np.array(points[0])}

"""
创建封闭的曲线+直线bevel
"""
def create_closed_concat_curve(name, control_points=None, index=None,sort_point=False):
    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 12

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.use_cyclic_u = True  # 设置样条为封闭的

    if index == 1:
        bezierSpline.bezier_points.add(len(control_points) - 3)
        for i, coord in enumerate(control_points[:-2]):
            x, y, z = coord
            bezier_point = bezierSpline.bezier_points[i]

            bezier_point.co = (x, y, z)
            bezier_point.handle_left_type = 'VECTOR'
            bezier_point.handle_right_type = 'VECTOR'
            if i == 0:  # P1处理
                bezier_point.handle_left_type = 'FREE'  
                bezier_point.handle_left = control_points[-2]
            elif i == len(control_points) - 3:
                bezier_point.handle_right_type = 'FREE'  # 左手柄自由，但需调整其位置
                bezier_point.handle_right = control_points[-1]
    elif index ==2:
        bezierSpline.bezier_points.add(len(control_points) - 2)
        for i, coord in enumerate(control_points[:-1]):
            x, y, z = coord
            bezier_point = bezierSpline.bezier_points[i]

            bezier_point.co = (x, y, z)
            bezier_point.handle_left_type = 'VECTOR'
            bezier_point.handle_right_type = 'VECTOR'
            # 对于第一个和最后一个点，我们调整手柄来形成弧线
            if i == 0:
                bezier_point.handle_left_type = 'FREE'  # 右手柄自由，但需调整其位置
                bezier_point.handle_left = (x, y, z) 
            elif i == len(control_points) - 2:
                bezier_point.handle_right_type = 'FREE'  # 左手柄自由，但需调整其位置
                bezier_point.handle_right = control_points[-1]
    elif index ==3:
        bezierSpline.bezier_points.add(len(control_points) - 2)

        for i, coord in enumerate(control_points[:-1]):
            x, y, z = coord
            bezier_point = bezierSpline.bezier_points[i]

            bezier_point.co = (x, y, z)
            bezier_point.handle_left_type = 'VECTOR'
            bezier_point.handle_right_type = 'VECTOR'
            # 对于第一个和最后一个点，我们调整手柄来形成弧线
            if i == 0:
                bezier_point.handle_right_type = 'FREE'  # 右手柄自由，但需调整其位置
                # 让右手柄沿着P1-P2的方向
                bezier_point.handle_right = (x, y, z) 
            elif i == len(control_points) - 2:  # P3处理
                bezier_point.handle_left_type = 'FREE'  # 左手柄自由，但需调整其位置
                # 让左手柄沿着P3-P2的方向
                bezier_point.handle_left = control_points[-1]
    elif index ==4:
        bezierSpline.bezier_points.add(len(control_points) - 1)

        for i, coord in enumerate(control_points):
            x, y, z = coord
            bezier_point = bezierSpline.bezier_points[i]

            bezier_point.co = (x, y, z)
            bezier_point.handle_left_type = 'VECTOR'
            bezier_point.handle_right_type = 'VECTOR'
    elif index ==5:
        bezierSpline.bezier_points.add(len(control_points) - 3)

        for i, coord in enumerate(control_points[:-2]):
            x, y, z = coord
            bezier_point = bezierSpline.bezier_points[i]

            bezier_point.co = (x, y, z)
            bezier_point.handle_left_type = 'VECTOR'
            bezier_point.handle_right_type = 'VECTOR'
            # 对于第一个和最后一个点，我们调整手柄来形成弧线
            if i == len(control_points) - 3:  
                bezier_point.handle_left_type = 'FREE'  # 右手柄自由，但需调整其位置
                bezier_point.handle_left = control_points[-2]
            elif i == len(control_points) - 4:
                bezier_point.handle_right_type = 'FREE'  # 左手柄自由，但需调整其位置
                bezier_point.handle_right = control_points[-1]
    elif index ==6:
        bezierSpline.bezier_points.add(len(control_points) - 3)

        for i, coord in enumerate(control_points[:-2]):
            x, y, z = coord
            bezier_point = bezierSpline.bezier_points[i]

            bezier_point.co = (x, y, z)
            bezier_point.handle_left_type = 'VECTOR'
            bezier_point.handle_right_type = 'VECTOR'
            if i == len(control_points) - 3: 
                bezier_point.handle_left_type = 'FREE'  # 右手柄自由，但需调整其位置
                # 让右手柄沿着P1-P2的方向
                bezier_point.handle_left = control_points[-2]
            elif i == len(control_points) - 4:
                bezier_point.handle_right_type = 'FREE'  # 左手柄自由，但需调整其位置
                bezier_point.handle_right = control_points[-1]

    curveOB = bpy.data.objects.new(name, curveData)
    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    if sort_point:
        control_points = np.array(sort_coordinates_endpoint(control_points))    
    return {"name":name,"points":np.array(control_points) - np.array(control_points[0]),"index":index}

"""
Create a section shape
"""
def create_section_shape(type, param_dict):
    
    param = {}

    name = param_dict["name"]
    if type=="rectangle":
        if "center" in param_dict.keys():
            param=create_rectangle(name,param_dict["width"],param_dict["length"],param_dict["rotation_rad"],param_dict["center"],closed=True)
        else:
            param=create_rectangle(name,param_dict["width"],param_dict["length"],param_dict["rotation_rad"],closed=True)
    elif type=="rectangle_points":
        param = create_rectangle_by_points(name,param_dict['points'],type=type)

    elif type=="rectangle_open":
        param=create_rectangle(name,param_dict["width"],param_dict["length"], param_dict["rotation_rad"],closed=False)

    elif type=="rectangle_open_points":
        param = create_rectangle_by_points(name,param_dict['points'],type=type)

    elif type=="circle":
        if "center" in param_dict.keys():
            param=create_circle(name, param_dict["radius"], center=param_dict["center"])
        else:
            param=create_circle(name, param_dict["radius"])

    elif type=="circle_points":
        create_bezier_circle_by_3Dpoints(name,param_dict['points'])

    elif type=="arc" or type=="arc_closed":
        if "center" in param_dict.keys():
            param=create_arc_points(name, param_dict["radius"], param_dict["start_angle"], param_dict["end_angle"],param_dict["location"], param_dict["rotation"],param_dict["center"] )
        else:
            param=create_arc_points(name, param_dict["radius"], param_dict["start_angle"], param_dict["end_angle"],param_dict["location"], param_dict["rotation"] )
    
    elif type=="arc_points":
        create_bezier_arc_by_3Dpoints(name,param_dict['points'])

    elif type=="arc_closed_points":
        create_bezier_arc_by_3Dpoints(name,param_dict['points'],param_dict["center"],closed=True)

    elif type=="oval":
        if "center" in param_dict.keys():
            param = create_oval(name, param_dict['a'],param_dict['b'],param_dict['rotation_rad'],param_dict['center'])
        else:
            param = create_oval(name, param_dict['a'],param_dict['b'],param_dict['rotation_rad'])

    elif type=="quad" or type=="triangle" :
        if "center" in param_dict.keys():
            param=create_quad(name, param_dict["points"],param_dict["center"])
        else:
            param=create_quad(name, param_dict["points"])

    elif type=="square": 
        if "center" in param_dict.keys():
            param=create_square(name, param_dict["length"],param_dict['rotation_rad'],param_dict['center'])
        else:
            param=create_square(name, param_dict["length"],param_dict['rotation_rad'])

    elif type=="curve":
        param = create_curve(name, param_dict["points"])

    elif type=="closed_curve":
        if "center" in param_dict.keys():
            param = create_closed_curve(name, param_dict["points"],param_dict["center"])
        else:
            param = create_closed_curve(name, param_dict["points"])
    
    elif type=="linebowl_curve":
        param = create_linebowl_curve(name, param_dict["points"])

    elif type=="bowl_curve":
        param = create_bowl_curve(name, param_dict["points"])

    elif type=="bottleneck_curve":
        param = create_bottleneck_curve(name, param_dict["points"])
    
    elif type=="mushroom_curve":
        param = create_mushroom_curve(name, param_dict["points"])
    
    elif type=="pot_curve":
        param = create_pot_curve(name, param_dict["points"])

    elif type=="polyline":
        param = create_polyline(name, param_dict["points"])

    elif type=="closed_concat":
        param = create_closed_concat_curve(name, param_dict["points"],index=param_dict["index"])
    
    return param

    

"""
Creates a translational object of a line trajectory
"""
def create_line_translation(name, bevel_name, control_points, points_radius,thickness, resolution=24, use_smooth=False, fill_caps="none"):
    coords = control_points

    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    polyline = curveData.splines.new('POLY')
    polyline.points.add(len(coords) - 1) 

    for i, coord in enumerate(coords):
        x, y, z = coord
        polyline.points[i].co = (x, y, z, 1)
        polyline.points[i].radius = points_radius[i]

    curveOB = bpy.data.objects.new(name, curveData)
    curveData.bevel_mode = "OBJECT"
    if fill_caps=="both":
        curveData.use_fill_caps = True
    else:
        curveData.use_fill_caps = False
    curveData.splines[0].use_smooth = use_smooth
    curveData.bevel_object = bpy.data.objects[bevel_name]

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)
    
    #bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.data.objects[name].select_set(True)
    bpy.ops.object.convert(target='MESH')
    bpy.data.objects.remove(bpy.data.objects[bevel_name], do_unlink=True)
    #bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    if fill_caps in ["start","end"]:
        make_caps(name,fill_caps)
    solidify(name,thickness)
    
    return curveOB


"""
Create a translational object of an arc or circle trajectory by three control points
"""
def arc_translation_by_points(name, control_points=[],type='circle'):
    
    assert len(control_points) == 3
    
    ng = bpy.data.node_groups.new('nodeGroupTranslation', 'GeometryNodeTree')

    inNode = ng.nodes.new('NodeGroupInput')
    outNode = ng.nodes.new('NodeGroupOutput')
    
    c2mNode = ng.nodes.new('GeometryNodeCurveToMesh')
    
    if type=='circle':
        arcNode = ng.nodes.new('GeometryNodeCurvePrimitiveCircle')
    elif type=='arc':
        arcNode = ng.nodes.new('GeometryNodeCurveArc')
    realNode = ng.nodes.new('GeometryNodeRealizeInstances')

    ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    
    ng.links.new(arcNode.outputs['Curve'], c2mNode.inputs['Curve'])
    ng.links.new(inNode.outputs['Geometry'], c2mNode.inputs['Profile Curve'])
    ng.links.new(c2mNode.outputs['Mesh'], realNode.inputs['Geometry'])
    ng.links.new(realNode.outputs['Geometry'], outNode.inputs['Geometry'])

    arcNode.mode = 'POINTS'
    for i,point in enumerate(control_points):
        arcNode.inputs[i+1].default_value=point
    modifier = bpy.data.objects[name].modifiers.new('nodeTranslation', "NODES")
    modifier.node_group = ng
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.ops.object.convert(target='MESH')
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    #bpy.ops.object.mode_set(mode = 'EDIT')
    #make_caps(name,fill_caps)


"""
Three control points create a translational body of the arc trajectory
"""
def create_arc_translation(name,bevel_name, control_points,thickness,points_radius=[1,1],use_smooth=False, fill_caps="none"):
    create_bezier_arc_by_3Dpoints(name,control_points,points_radius)
    curveData = bpy.data.objects[name].data
    
    curveData.bevel_mode = "OBJECT"
    if fill_caps=="both":
        curveData.use_fill_caps = True
    else:
        curveData.use_fill_caps = False
    curveData.splines[0].use_smooth = use_smooth
    curveData.bevel_object = bpy.data.objects[bevel_name]
    
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.data.objects[name].select_set(True)
    bpy.ops.object.convert(target='MESH')
    bpy.data.objects.remove(bpy.data.objects[bevel_name], do_unlink=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    if fill_caps in ["start","end"]:
        make_caps(name,fill_caps)

    solidify(name,thickness)

"""
A translational body of a radius-controlled circular trajectory
"""
def create_circle_translation(name, bevel_name, r, location, rotation,thickness, points_radius=[1.,1.,1.,1.], resolution=24,use_smooth=False, fill_caps="none"):
    bpy.ops.curve.simple(align='WORLD', location=[0,0,0], rotation=[0,0,0], Simple_Type='Circle',shape='3D',Simple_sides=4,Simple_radius=r, outputType='BEZIER', use_cyclic_u=True,edit_mode=False)
    curve = bpy.context.object
    curve.name = name
    curve.location = location
    curve.rotation_mode = "QUATERNION"
    curve.rotation_quaternion = rotation
    
    curveData = bpy.data.objects[name].data
    curveData.resolution_u = resolution
    for i in range(4):
        curveData.splines[0].bezier_points[i].radius=points_radius[i]
    
    curveData.bevel_mode = "OBJECT"
    if fill_caps=="both":
        curveData.use_fill_caps = True
    else:
        curveData.use_fill_caps = False
    curveData.splines[0].use_smooth = use_smooth
    curveData.bevel_object = bpy.data.objects[bevel_name]
    
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.data.objects[name].select_set(True)
    bpy.ops.object.convert(target='MESH')
    bpy.data.objects.remove(bpy.data.objects[bevel_name], do_unlink=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    if fill_caps in ["start","end"]:
        make_caps(name,fill_caps)
    solidify(name,thickness)
    
    
"""
Creates a translational object of a Bezier trajectory
"""
def create_curve_translation(name, bevel_name, control_points, points_radius, thickness, resolution=24, use_smooth=False, fill_caps="none", flip_normals=False):
    coords = control_points

    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(len(coords) - 1) 

    for i, coord in enumerate(coords):
        x, y, z = coord
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.co = (x, y, z)
        bezier_point.handle_left_type = 'AUTO'
        bezier_point.handle_right_type = 'AUTO'
        bezier_point.radius = points_radius[i]

    curveOB = bpy.data.objects.new(name, curveData)
    curveData.bevel_mode = "OBJECT"
    if fill_caps=="both":
        curveData.use_fill_caps = True
    else:
        curveData.use_fill_caps = False
    curveData.splines[0].use_smooth = use_smooth
    curveData.bevel_object = bpy.data.objects[bevel_name]

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)

    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.data.objects[name].select_set(True)
    bpy.ops.object.convert(target='MESH')
    bpy.data.objects.remove(bpy.data.objects[bevel_name], do_unlink=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    
    if flip_normals:
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.mode_set(mode = 'EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode = 'OBJECT')
   
    if fill_caps in ["start","end"]:
        make_caps(name,fill_caps)
        
    if thickness>1e-10:
        solidify(name,thickness)
    
    return curveOB

"""
Creates a translational object in which the middle segment is a Bézier trajectory of a straight line
"""
def create_concatcurve_translation(name, bevel_name, control_points, concatcurve_id,points_radius, thickness, resolution=24, use_smooth=False, fill_caps="none"):
    points = control_points
    if concatcurve_id==0:
        control_points = points[:4]+[points[3]] + [points[4]] + points[4:] 
    elif concatcurve_id==1:
        control_points = [points[0]] + points[:2] + [points[1]] + points[2:4] + [points[4]]+points[4:] + [points[-1]]
    elif concatcurve_id==2:
        control_points = [points[0]] + points[:2] + [points[1]] + points[2:]
    elif concatcurve_id==3:
        control_points = points[:-1] + [points[-2]] + [points[-1]] + [points[-1]]
    
    control_points = [[0,0,0]]+control_points+[[0,0,0]]
    control_points = np.array(control_points)

    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = resolution

    
    bezierSpline = curveData.splines.new('BEZIER') 
    num_cycle = int(len(control_points)/3)
    bezierSpline.bezier_points.add(num_cycle - 1)
    #assert len(control_points) == 12
    
    for i in range(num_cycle):        
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.handle_left = control_points[i*3]
        bezier_point.co = control_points[i*3+1]
        bezier_point.handle_right = control_points[i*3+2]
        

    curveOB = bpy.data.objects.new(name, curveData)
    curveData.bevel_mode = "OBJECT"
    if fill_caps=="both":
        curveData.use_fill_caps = True
    else:
        curveData.use_fill_caps = False
    curveData.splines[0].use_smooth = use_smooth
    curveData.bevel_object = bpy.data.objects[bevel_name]

    scn = bpy.context.scene.collection
    scn.objects.link(curveOB)

    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.data.objects[name].select_set(True)
    bpy.ops.object.convert(target='MESH')
    bpy.data.objects.remove(bpy.data.objects[bevel_name], do_unlink=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    if fill_caps in ["start","end"]:
        make_caps(name,fill_caps)
    solidify(name,thickness)
    return curveOB

"""
Create a rotation object
"""
def bezier_rotation(name,bevel_name,location=[0,0,0], rotation=[0,0,0,0],thickness=0.01,use_smooth=False):
    r=1
    bpy.ops.curve.simple(align='WORLD', location=[0,0,0], rotation=[0,0,0], Simple_Type='Circle',shape='3D',Simple_sides=4,Simple_radius=r, outputType='BEZIER', use_cyclic_u=True,edit_mode=False)
    curve = bpy.context.object
    curve.name = name
    curveData = bpy.data.objects[name].data
    curveData.offset=r
    curve.location = location
    curve.rotation_mode = "QUATERNION"
    curve.rotation_quaternion = rotation
    
    curveData.bevel_mode = "OBJECT"
    curveData.splines[0].use_smooth = use_smooth
    curveData.bevel_object = bpy.data.objects[bevel_name]
    
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.data.objects[name].select_set(True)
    bpy.ops.object.convert(target='MESH')
    bpy.data.objects.remove(bpy.data.objects[bevel_name], do_unlink=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    solidify(name, thickness)
    
"""
Creates a translational object of a line trajectory
"""
def create_curve(name, profile_name=None,control_points=[],points_radius=[],handle_type=[],closed=False, center="POINT",thickness=None, fill_caps="none",flip_normals=False):
    type_dict={0:"AUTO", 1:"VECTOR", 2:"ALIGNED", 3:"FREE"}

    control_points = np.array(control_points).tolist()
    control_points_tmp = copy.deepcopy(control_points)
    #统计手柄坐标的数量与真实控制点的数量
    num_handle_co = handle_type.count(3)
    num_control_points = len(control_points) - num_handle_co

    curveData = bpy.data.curves.new(name, type='CURVE')
    curveData.dimensions = '3D'
    curveData.resolution_u = 24

    bezierSpline = curveData.splines.new('BEZIER')
    bezierSpline.bezier_points.add(num_control_points - 1) 
    bezierSpline.use_cyclic_u = closed

    for i in range(num_control_points):
        bezier_point = bezierSpline.bezier_points[i]
        bezier_point.handle_left_type = type_dict[handle_type[2*i]]
        if type_dict[handle_type[2*i]]=="FREE":
            bezier_point.handle_left = control_points.pop(0)
        bezier_point.co = control_points.pop(0)
        bezier_point.handle_right_type = type_dict[handle_type[2*i+1]]
        if type_dict[handle_type[2*i+1]]=="FREE":
            bezier_point.handle_right = control_points.pop(0)
        bezier_point.radius = points_radius[i] if len(points_radius)!=0 else 1.0
    
    assert len(control_points)==0, "cannot create curve"
    curveOB = bpy.data.objects.new(name, curveData)

    if profile_name != None:
        curveData.bevel_mode = "OBJECT"
        if fill_caps=="both":
            curveData.use_fill_caps = True
        else:
            curveData.use_fill_caps = False
        curveData.splines[0].use_smooth = False
        curveData.bevel_object = bpy.data.objects[profile_name]

        scn = bpy.context.scene.collection
        scn.objects.link(curveOB)

        bpy.context.view_layer.objects.active = bpy.data.objects[name]
        bpy.ops.object.mode_set(mode = 'OBJECT')
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.convert(target='MESH')
        bpy.data.objects.remove(bpy.data.objects[profile_name], do_unlink=True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
        
        if flip_normals:
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.mode_set(mode = 'EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode = 'OBJECT')
    
        if fill_caps in ["start","end"]:
            make_caps(name,fill_caps)
            
        if thickness>1e-10:
            solidify(name,thickness)

        return curveOB
    
    else:
        scn = bpy.context.scene.collection
        scn.objects.link(curveOB)
        if center=="MEDIAN":
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        points=np.array(control_points_tmp)
        return {"name":name, "points":points, "handle_type":handle_type, "closed":closed, "center":center}
def create_star(name, outer_radius, inner_radius, num_points, edge="smooth"):
    # Create a simple circle curve using the built-in operator
    bpy.ops.curve.simple(
        align='WORLD',
        location=[0, 0, 0],
        rotation=[0, 0, 0],
        Simple_Type='Circle',
        shape='3D',
        Simple_sides=num_points * 2,  # Each point alternates between outer and inner radius
        Simple_radius=outer_radius,
        outputType='BEZIER',
        use_cyclic_u=True,
        edit_mode=False
    )
    
    curve = bpy.context.object
    curve.name = name
    # if center == "POINT":
    #     curve.location = [outer_radius, 0, 0]
    #     bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    
    curve_data = curve.data
    curve_data.dimensions = '3D'
    
    # Adjust the control points to create a star shape
    angle_step = math.pi / num_points  # Angle step between each point
    spline = curve_data.splines[0]
    points = spline.bezier_points
    
    handle_type = "VECTOR" if edge == "sharp" else "AUTO"
    
    for i, point in enumerate(points):
        angle = i * angle_step
        # Alternate between outer and inner radius
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0
        point.co = (x, y, z)
        
        # Set handles to smooth transition VECTOR
        point.handle_left_type = handle_type
        point.handle_right_type = handle_type
    
    return {"name": name, "outer_radius": outer_radius, "inner_radius": inner_radius, "num_points": num_points}
def star_curve_rotation(name,bevel_name,num_points=16, inner_radius=1, location=[0,0,0], rotation=[0,0,0,0],thickness=0.002,use_smooth=False):
    create_star(name, 1, inner_radius, num_points, edge="sharp")
    curve = bpy.context.object
    curveData = bpy.data.objects[name].data
    curve.location = location
    curve.rotation_mode = "QUATERNION"
    curve.rotation_quaternion = rotation
    curveData.offset=inner_radius
    curveData.bevel_mode = "OBJECT"
    curveData.splines[0].use_smooth = use_smooth
    curveData.bevel_object = bpy.data.objects[bevel_name]
    
    bpy.context.view_layer.objects.active = bpy.data.objects[name]
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.data.objects[name].select_set(True)
    bpy.ops.object.convert(target='MESH')
    bpy.data.objects.remove(bpy.data.objects[bevel_name], do_unlink=True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    solidify(name, thickness)
    
def subsurf(name, levels, simple=False):
    """ 
    This function aims to add subdivide surface modifier and apply to the object
    TODO: There are some properties that haven't added to this function yet.
    """
    if levels > 0:
        bpy.data.objects[name].modifiers.new("Subsurf", "SUBSURF")
        bpy.data.objects[name].modifiers["Subsurf"].levels = levels
        bpy.data.objects[name].modifiers["Subsurf"].render_levels = levels
        bpy.data.objects[name].modifiers["Subsurf"].subdivision_type = "SIMPLE" if simple else "CATMULL_CLARK"
        bpy.context.view_layer.objects.active = bpy.data.objects[name]
        bpy.ops.object.modifier_apply(modifier="Subsurf")
    
from random import uniform
from random import choice
def create_vase(name):
    neck_scale = uniform(0.2, 0.8)
    top_scale = neck_scale * uniform(0.8, 1.2)
    z = uniform(0.25, 0.40)
    x = uniform(0.2, 0.4) * z
    neck_position = 0.5 * neck_scale + 0.5 + uniform(-0.05, 0.05)
    neck_middle_position = uniform(0.7, 0.95)
    shoulder_position = uniform(0.3, 0.7)
    shoulder_thickness = uniform(0.1, 0.25)
    foot_height = uniform(0.01, 0.1)
    foot_scale = uniform(0.4, 0.6)
    inner_radius = choice([1.0, uniform(0.8, 1.0)])
    map_range = shoulder_position * (neck_position - foot_height) + foot_height
    pos_body_top = map_range
    pos_body_top += (neck_position - foot_height) * shoulder_thickness
    pos_body_top = min(pos_body_top, neck_position)
    pos_body_bottom = map_range
    pos_body_bottom -= (neck_position - foot_height) * shoulder_thickness
    pos_body_bottom = max(pos_body_bottom, foot_height)
    pts = [
        [top_scale * x, z, 0],
        [(top_scale + neck_scale) / 2 * x, z * ((1 - neck_position) * neck_middle_position + neck_position), 0],
        [neck_scale * x, z * neck_position, 0],
        [x, pos_body_top * z, 0],
        [x, pos_body_bottom * z, 0],
        [x * foot_scale,foot_height * z, 0],
        [x * foot_scale, 0, 0],
        [0, 0, 0]
    ]
    handle_type = [
        0, 0, 0, 0, 0,0 ,0 ,0, 0,0 ,0, 0, 0, 1, 1, 1
    ]
    print(pts, inner_radius)
    create_curve("temp", None, pts, [], handle_type)
    star_curve_rotation(name, "temp", inner_radius=inner_radius)
    subsurf(name, 2)
#create_vase("revolute")

def create_spiral(name, profile_name=None,dif_z=0.5,radius=0.5,turns=5,closed=False, center="POINT",thickness=0, handle_type="AUTO",fill_caps="none",flip_normals=False, direction="COUNTER_CLOCKWISE", resolution=24 ,location=[0,0,0], rotation=[0,0,0,0]):
    bpy.ops.curve.spirals(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), spiral_type='ARCH', turns=turns, dif_z=dif_z, radius=radius, shape='3D', curve_type='BEZIER', use_cyclic_u=closed,edit_mode=False, spiral_direction=direction, handleType=handle_type)
    curve = bpy.context.object
    curve.name = name
    curveData = curve.data
    curveData.resolution_u = resolution
    #curveOB = bpy.data.objects.new(name, curveData)
    if thickness is None:
        thickness = 0
    if profile_name != None:
        curveData.bevel_mode = "OBJECT"
        if fill_caps=="both":
            curveData.use_fill_caps = True
        else:
            curveData.use_fill_caps = False
        curveData.splines[0].use_smooth = False
        curveData.bevel_object = bpy.data.objects[profile_name]

        #scn = bpy.context.scene.collection
        #scn.objects.link(curveOB)
        bpy.context.view_layer.objects.active = bpy.data.objects[name]
        bpy.ops.object.mode_set(mode = 'OBJECT')
        bpy.data.objects[name].select_set(True)
        bpy.ops.object.convert(target='MESH')
        bpy.data.objects.remove(bpy.data.objects[profile_name], do_unlink=True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
        
        if flip_normals:
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.mode_set(mode = 'EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode = 'OBJECT')
    
        if fill_caps in ["start","end"]:
            make_caps(name,fill_caps)
            
        if thickness>1e-10:
            solidify(name,thickness)
    
    else:
        if center=="MEDIAN":
            bpy.data.objects[name].select_set(True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    obj = bpy.data.objects[name]
    if rotation:
        obj.rotation_mode = 'QUATERNION'
        #print(rotation)
        obj.rotation_quaternion = rotation
    if location:
        obj.location = location
    bpy.ops.object.transform_apply(True)
    return {
        "name": name, "profile_name": profile_name,"dif_z": 0.5,"radius": 0.5,"turns": 5,"closed": False, "center": "POINT","thickness": None, "handle_type": "AUTO","fill_caps": "none","flip_normals": False, "direction": "COUNTER_CLOCKWISE", "resolution":24, "thickness": thickness
    }



from numpy.random import uniform
def log_uniform(low, high, size=None):
    return np.exp(uniform(np.log(low), np.log(high), size))
def create_fork(name, x_tip, thickness, n_cuts, x_anchors, y_anchors, z_anchors, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0,0.0), scale=(1, 1, 1)):
    has_cut =True
    print(n_cuts)
    n = 2 * (n_cuts + 1)
    obj = create_primitive(name=name, primitive_type="grid", x_subdivisions=len(x_anchors) - 1, y_subdivisions=n - 1)
    x = np.concatenate([x_anchors] * n)
    y = np.ravel(y_anchors[np.newaxis, :] * np.linspace(1, -1, n)[:, np.newaxis])
    z = np.concatenate([z_anchors] * n)
    arr = np.stack([x, y, z], -1)
    obj.data.vertices.foreach_set("co", arr.reshape(-1))

    #simply copy make cuts
    if has_cut:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(obj.data)
        front_verts = []
        for v in bm.verts:
            if abs(v.co[0] - x_tip) < 1e-3:
                front_verts.append(v)
        front_verts = sorted(front_verts, key=lambda v: v.co[1])
        geom = []
        for f in bm.faces:
            vs = list(v for v in f.verts if v in front_verts)
            if len(vs) == 2:
                if min(front_verts.index(vs[0]), front_verts.index(vs[1])) % 2 == 1:
                    geom.append(f)
        bmesh.ops.delete(bm, geom=geom, context="FACES")
        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
    solidify(name, thickness)
    subsurf(name, 1)
    subsurf(name, 1)
    arr = np.array([v.co for v in obj.data.vertices])
    center = np.array(((arr[:, 0].max() + arr[:, 0].min()) / 2, (arr[:, 1].max() + arr[:, 1].min()) / 2, (arr[:, 2].max() + arr[:, 2].min()) / 2))
    obj.location = -center
    bpy.ops.object.transform_apply(True)
    if scale is not None:
        obj.scale = scale
        bpy.ops.object.transform_apply(True)
    if rotation is not None:
        obj.rotation_mode = 'QUATERNION'
        #print(rotation)
        obj.rotation_quaternion = rotation
    bpy.ops.object.transform_apply(True)
    if location is not None:
        obj.location = location
    bpy.ops.object.transform_apply(True)
    return {"name": name, "x_tip": x_tip, "thickness": thickness,"n_cuts": n_cuts, "x_anchors": x_anchors, "y_anchors": y_anchors, "z_anchors": z_anchors, "location": location, "rotation": rotation, "scale": scale}
    
def make_fork():
    for i in range(100):
        x_length = log_uniform(0.4, 0.8)
        x_tip = uniform(0.15, 0.2)
        y_length = log_uniform(0.05, 0.08)
        x_end = 0.15
        z_depth = log_uniform(0.02, 0.04)
        z_offset = uniform(0.0, 0.05)
        thickness = log_uniform(0.008, 0.015)
        n_cuts = np.random.randint(1, 3) if uniform(0, 1) < 0.3 else 3
        x_anchors = np.array(
            [
                x_tip,
                uniform(-0.04, -0.02),
                -0.08,
                -0.12,
                -x_end,
                -x_end - x_length,
                -x_end - x_length * log_uniform(1.2, 1.4),
            ]
        )
        y_anchors = np.array(
            [
                y_length * log_uniform(0.8, 1.0),
                y_length * log_uniform(1.0, 1.2),
                y_length * log_uniform(0.6, 1.0),
                y_length * log_uniform(0.2, 0.4),
                log_uniform(0.01, 0.02),
                log_uniform(0.02, 0.05),
                log_uniform(0.01, 0.02),
            ]
        )
        z_anchors = np.array(
            [
                0,
                -z_depth,
                -z_depth,
                0,
                z_offset,
                z_offset + uniform(-0.02, 0.04),
                z_offset + uniform(-0.02, 0),
            ]
        )
        location = (random.random(), random.random(), random.random())
        rotation = (random.random(), random.random(), random.random(), random.random())
        rotation = (0.0, 0.0, 0.0,0.0)
        p = create_fork(f"fork{i}", x_tip, thickness, n_cuts, x_anchors, y_anchors, z_anchors, location, rotation)
        p = change_param_according_mesh(f"fork{i}", p)
        p["name"] = f"fork_{i}"
        print(p)
        create_fork(**p)

from numpy.random import uniform
def log_uniform(low, high, size=None):
    return np.exp(uniform(np.log(low), np.log(high), size))
def create_spoon(name, z_depth, thickness, x_anchors, y_anchors, z_anchors, location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0,0.0), scale=(1, 1, 1)):
    obj = create_primitive(name=name, primitive_type="grid", x_subdivisions=len(x_anchors) - 1, y_subdivisions=2)
    x = np.concatenate([x_anchors] * 3)
    y = np.concatenate([y_anchors, np.zeros_like(y_anchors), -y_anchors])
    z = np.concatenate([z_anchors] * 3)
    x[len(x_anchors)] += 0.02
    z[len(x_anchors) + 1] = -z_depth
    arr = np.stack([x, y, z], -1)
    obj.data.vertices.foreach_set("co", arr.reshape(-1))
    solidify(name, thickness)
    subsurf(name, 1)
    subsurf(name, 2)
    if scale is not None:
        obj.scale = scale
    bpy.ops.object.transform_apply(True)
    if location is not None:
        obj.location = location
    bpy.ops.object.transform_apply(True)
    if rotation is not None:
        obj.rotation_mode = 'QUATERNION'
        #print(rotation)
        obj.rotation_quaternion = rotation
    bpy.ops.object.transform_apply(True)
    return {"name": name, "z_depth": z_depth, "thickness": thickness,"x_anchors": x_anchors, "y_anchors": y_anchors, "z_anchors": z_anchors, "location": location, "rotation": rotation, "scale": scale}
import random
def make_spoon():
    for i in range(100):
        x_end = 0.15
        x_length = log_uniform(0.2, 0.8)
        y_length = log_uniform(0.06, 0.12)
        z_depth = log_uniform(0.08, 0.25)
        z_offset = uniform(0.0, 0.05)
        thickness = log_uniform(0.008, 0.015)
        x_anchors = np.array(
                [
                    log_uniform(0.07, 0.25),
                    0,
                    -0.08,
                    -0.12,
                    -x_end,
                    -x_end - x_length,
                    -x_end - x_length * log_uniform(1.2, 1.4),
                ]
            )
        y_anchors = np.array(
                [
                    y_length * log_uniform(0.1, 0.8),
                    y_length * log_uniform(1.0, 1.2),
                    y_length * log_uniform(0.6, 1.0),
                    y_length * log_uniform(0.2, 0.4),
                    log_uniform(0.01, 0.02),
                    log_uniform(0.02, 0.05),
                    log_uniform(0.01, 0.02),
                ]
            )
        z_anchors = np.array(
                [
                    0,
                    0,
                    0,
                    0,
                    z_offset,
                    z_offset + uniform(-0.02, 0.04),
                    z_offset + uniform(-0.02, 0),
                ]
            )
        location = (random.random(), random.random(), random.random())
        #rotation = (random.random(), random.random(), random.random(), random.random())
        rotation = (0.0, 0.0, 0.0,0.0)
        p = create_spoon(f"spoon_{random.random()}", z_depth, thickness, x_anchors, y_anchors, z_anchors, location, rotation)
        p = change_param_according_mesh(p["name"], p)
        print(p)
        create_spoon(**p)

make_fork()
from infinigen.core.util import blender as butil
butil.save_blend("fork.blend", autopack=True)
