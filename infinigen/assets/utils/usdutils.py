from pxr import Usd, UsdGeom, UsdPhysics, Gf

from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
import open3d as o3d

max_try = 100

name2path = {}
stage = None
def init_usd_stage(path):
    """
    Initializes a new USD stage with a root Xform.

    Args:
        usd_file (str): Path to the USD file to be created.

    Returns:
        Usd.Stage: The new USD stage.
    """
    global stage
    if stage is not None:
        print("Stage already exists. Clearing existing stage.")
        stage.GetRootLayer().Clear()
    # Create a new USD stage
    stage = Usd.Stage.CreateNew(path)
    
    # Create the root Xform
    root_xform = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root_xform.GetPrim())


def traverse_stage(stage):
        for prim in stage.Traverse():
            traverse_prim_and_references(prim, stage)

def traverse_prim_and_references(prim, stage):
        if prim.GetTypeName() == "Mesh":
            UsdPhysics.RigidBodyAPI.Apply(prim)  # Enable rigid body physics
            UsdPhysics.CollisionAPI.Apply(prim)  

def add_mesh(obj_file, name, translate=(0, 0, 0)):
    """
    Adds a mesh to the USD stage.

    Args:
        obj_file (str): Path to the OBJ file to be imported.
        name (str): Name of the mesh.
        translate (tuple): Translation of the mesh.

    Returns:
        UsdGeom.Mesh: The imported mesh.
    """
    # Import the mesh
    global stage
    print(name, translate)
    stage1 = Usd.Stage.Open(obj_file)
    root = UsdGeom.Xform(stage1.GetPrimAtPath("/World"))
    root.ClearXformOpOrder()
    translate = translate[0], -translate[2] + 0.2, translate[1]
    root.AddTranslateOp().Set(Gf.Vec3d(translate))
    for prim in stage1.Traverse():
        if prim.GetTypeName() == "Mesh":
            UsdPhysics.RigidBodyAPI.Apply(prim)  # Enable rigid body physics
            UsdPhysics.CollisionAPI.Apply(prim)  # Enable collision for mesh1
            name2path[name] = prim.GetPath()
            prim.ClearXformOpOrder()
            
            #prim.AddTranslateOp().Set(Gf.Vec3d(translate))
    stage1.GetRootLayer().Export(obj_file)
    UsdGeom.Xform(stage.DefinePrim(f"/World/{name}", "Xform"))
    stage.GetPrimAtPath(f"/World/{name}").GetReferences().AddReference(obj_file)
    print(f"Mesh added: {name}")


def add_joint(name_joint, name_1, name_2, axis="Z", lower_limit=0.0, upper_limit=0.0, type="fixed", shift_from_body0=(0, 0, 0)):
    """
    Adds a joint to the USD stage.

    Args:
        name (str): Name of the joint.
        body0 (str): Path to the first body.
        body1 (str): Path to the second body.
        axis (str): Axis of the joint.
        lower_limit (float): Minimum relative translation.
        upper_limit (float): Maximum relative translation.
    """
    # Define a Prismatic Joint
    global stage
    return 
    if name_1 == "l_world" or name_2 == "l_world":
        print("Cannot add joint with l_world")
        return
    if type == "prismatic":
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/World/{name_joint}")
    elif type == "revolute":
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/{name_joint}")
    elif type == "fixed":
        joint = UsdPhysics.FixedJoint.Define(stage, f"/World/{name_joint}")
    else:
        print("Invalid joint type.")
        return
    path_1 = str(name2path[name_1])
    path_1 = '/' + '/'.join([path_1.split("/")[1], name_1] + path_1.split("/")[2:])
    path_2 = str(name2path[name_2])
    path_2 = '/' + '/'.join([path_2.split("/")[1], name_2] + path_2.split("/")[2:])

    #joint = UsdPhysics.PrismaticJoint.Define(stage, f"/Root/{name_joint}")
    body0 = path_1#f"/World/{name_1}"
    body1 = path_2#f"/World/{name_2}"
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    if type == "fixed":
        print(f"Joint added: {name_joint}")
        return
    joint.CreateAxisAttr(axis)  # Movement along X-axis
    #joint.CreateLocalPos0Attr(shift_from_body0)  # Position of the joint in body0's local frame
    joint.CreateLowerLimitAttr(lower_limit)  # Minimum relative translation
    joint.CreateUpperLimitAttr(upper_limit)  # Maximum relative translation
    print(f"Joint added: {name_joint}")


def save():
    """
    Saves the USD stage to a file.

    Args:
        path (str): Path to save the USD file.
    """
    global stage
    stage.GetRootLayer().Save()
    print(f"USD file saved")

    