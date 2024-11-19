import bpy

def get_geometry_data(geometry):
    # 创建一个临时对象
    temp_mesh = bpy.data.meshes.new(name="TempMesh")
    temp_obj = bpy.data.objects.new(name="TempObject", object_data=temp_mesh)
    bpy.context.collection.objects.link(temp_obj)
    
    # 获取包含 geometry_node 的节点树 (NodeTree)
    node_tree = geometry.id_data  # geometry_node 是一个节点实例
    
    # 添加几何节点修改器并设置节点组为 node_tree
    modifier = temp_obj.modifiers.new(name="GeometryNodes", type='NODES')
    modifier.node_group = node_tree  # 赋值为节点树而不是单个节点
    
    # 评估依赖图获取计算后的对象
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = temp_obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()
    
    # 获取顶点和面数据
    vertices = [v.co.copy() for v in eval_mesh.vertices]
    faces = [list(p.vertices) for p in eval_mesh.polygons]
    
    # 清理临时对象
    eval_obj.to_mesh_clear()
    bpy.data.objects.remove(temp_obj, do_unlink=True)
    bpy.data.meshes.remove(temp_mesh)
    
    return vertices, faces
