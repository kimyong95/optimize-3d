import inspect
import trimesh
from trellis.representations.mesh.cube2mesh import MeshExtractResult
import numpy as np

class collect_calls:
    def __init__(self, target_method, arg_names=None):
        self.obj, self.func_name = target_method.__self__, target_method.__name__
        self.original_func = getattr(self.obj, self.func_name)
        self.param_names = list(inspect.signature(self.original_func).parameters)
        self.names_to_collect = set(arg_names or [])
        self.data = []

    def __enter__(self):
        def wrapper(*args, **kwargs):
            res = self.original_func(*args, **kwargs)
            all_args = {**dict(zip(self.param_names, args)), **kwargs}
            self.data.append({'args': {k: v for k, v in all_args.items() if k in self.names_to_collect}, 'return': res})
            return res
        setattr(self.obj, self.func_name, wrapper)
        return self.data

    def __exit__(self, *args):
        setattr(self.obj, self.func_name, self.original_func)

def to_trimesh(mesh_result: MeshExtractResult) -> trimesh.Trimesh:
    vertices = mesh_result.vertices.cpu().numpy()
    faces = mesh_result.faces.cpu().numpy()
    vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def post_process_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    - Rotate +90° around X, then +90° around Z (about AABB centroid).
    - Uniformly scale 3x larger (about AABB centroid).
    - Translate so:
        • y=0 at center of volume,
        • x=0 at 20% of x-span from x_min,
        • z-min = -0.318469 m.
    """
    if mesh is None:
        return None

    # 1) ROTATE
    c = mesh.bounding_box.centroid
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(90.0), [1,0,0], point=c))
    c = mesh.bounding_box.centroid
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(90.0), [0,0,-1], point=c))

    # 2) SCALE 3.5× about current AABB centroid
    c = mesh.bounding_box.centroid
    mesh.apply_transform(trimesh.transformations.scale_matrix(3.5, c))

    # 3) TRANSLATE
    m_bounds = mesh.bounds
    x_min, x_max = float(m_bounds[0,0]), float(m_bounds[1,0])
    z_min = float(m_bounds[0,2])
    x_20 = x_min + 0.2 * (x_max - x_min)

    dx = -x_20
    dy = -float(mesh.center_mass[1])
    dz = -0.318469 - z_min

    mesh.apply_transform(trimesh.transformations.translation_matrix([dx, dy, dz]))
    return mesh