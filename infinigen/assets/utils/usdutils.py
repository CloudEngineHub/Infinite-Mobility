from pxr import Usd, UsdGeom, UsdPhysics, Gf

from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
import open3d as o3d

max_try = 100


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
    root_xform = UsdGeom.Xform.Define(stage, "/Root")
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
    mesh1 = UsdGeom.Mesh.Define(stage, f"/Root/{name}")
    mesh1.GetPrim().GetReferences().AddReference(obj_file)
    mesh1.AddTranslateOp().Set(value=translate)  # Position mesh1 above the ground
    UsdPhysics.RigidBodyAPI.Apply(mesh1.GetPrim())  # Enable rigid body physics
    UsdPhysics.CollisionAPI.Apply(mesh1.GetPrim())  # Enable collision for mesh1
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
    if type == "prismatic":
        joint = UsdPhysics.PrismaticJoint.Define(stage, f"/Root/{name_joint}")
    elif type == "revolute":
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Root/{name_joint}")
    elif type == "fixed":
        joint = UsdPhysics.FixedJoint.Define(stage, f"/Root/{name_joint}")
    else:
        print("Invalid joint type.")
        return
    #joint = UsdPhysics.PrismaticJoint.Define(stage, f"/Root/{name_joint}")
    body0 = f"/Root/{name_1}"
    body1 = f"/Root/{name_2}"
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    if type == "fixed":
        print(f"Joint added: {name_joint}")
        return
    joint.CreateAxisAttr(axis)  # Movement along X-axis
    joint.CreateLocalPos0Attr(shift_from_body0)  # Position of the joint in body0's local frame
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

    