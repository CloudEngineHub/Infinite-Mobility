import bpy
import addon_utils
addon_utils.enable("add_curve_extra_objects")
import numpy as np
import copy

def create_curve(name, profile_name=None,control_points=[],points_radius=[],handle_type=[],closed=False, center="POINT",thickness=None, fill_caps="none",flip_normals=False, resolution=24):
    if isinstance(name, str):
        type_dict={0:"AUTO", 1:"VECTOR", 2:"ALIGNED", 3:"FREE"}

        control_points = np.array(control_points).tolist()
        control_points_tmp = copy.deepcopy(control_points)
        #统计手柄坐标的数量与真实控制点的数量
        num_handle_co = handle_type.count(3)
        num_control_points = len(control_points) - num_handle_co

        curveData = bpy.data.curves.new(name, type='CURVE')
        curveData.dimensions = '3D'
        curveData.resolution_u = resolution

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

            return curveOB
        else:
            scn = bpy.context.scene.collection
            scn.objects.link(curveOB)
            if center=="MEDIAN":
                bpy.data.objects[name].select_set(True)
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
            points=np.array(control_points_tmp)
            return {"name":name, "points":points, "handle_type":handle_type, "closed":closed, "center":center}
    else:
        if profile_name==None:
            if isinstance(profile_name, str):
                profile_name = [profile_name]*len(name)
            if isinstance(points_radius[0], float) or isinstance(points_radius[0], int):
                points_radius = [points_radius]*len(name)
            if isinstance(handle_type[0], int):
                handle_type = [handle_type]*len(name)
            if isinstance(closed, bool):
                closed = [closed]*len(name)
            if isinstance(center, str):
                center = [center]*len(name)
            for i in range(len(name)):
                create_curve(name=name[i], control_points=control_points[i], points_radius=points_radius[i], handle_type=handle_type[i], closed=closed[i], center=center[i], thickness=thickness, fill_caps=fill_caps, flip_normals=flip_normals, resolution=resolution)
            points = np.array(copy.deepcopy(control_points))
            return {"name":name, "points":points, "handle_type":handle_type, "closed":closed, "center":center}