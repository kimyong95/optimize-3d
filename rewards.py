# set -a; source .env; set +a
# docker run --name domino -d --runtime=nvidia \
#   --gpus \"device=0,1,2,3\" --shm-size 2g -p 8000:8000 -e NGC_API_KEY \
#   -t nvcr.io/nim/nvidia/domino-automotive-aero:2.0.0

import io, httpx, numpy
import tempfile
import functools
import time
import trimesh
from httpx import HTTPError
from shapely.geometry import Polygon
from concurrent.futures import ThreadPoolExecutor, as_completed
import shapely
import uuid
import pickle
import numpy as np
import torch

def retry(times, failed_return, exceptions, backoff_factor=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs, retry_attempt=attempt)
                except exceptions as e:
                    print(
                        f"Exception [{type(e)}:{e}] thrown when attempting to run {func}, attempt {attempt} of {times}"
                    )
                    time.sleep(backoff_factor * 2**attempt)
                    attempt += 1
            return failed_return
        return wrapper
    return decorator

OBJECTIVE_SHORT_NAMES = {
    "drag-force": "df",
    "drag-coefficient": "dc",
    "scaled-drag-force": "sdf",
    "lift-force": "lf",
    "lift-coefficient": "lc",
    "scaled-lift-force": "slf",
    "lift-to-drag-ratio": "ld",
}
class ObjectiveEvaluator:
    def __init__(self, domain_name="localhost", port="8000", objectives=["drag-coefficient"]):
        self.url = f"http://{domain_name}:{port}/v1/infer"
        self.data = {
            "stream_velocity": "30.0", 
            "stencil_size": "1",
            "point_cloud_size": "500000",
        }

        self.objectives = objectives

        # --- Constants for Cd calculation (no new user parameters) ---
        self._RHO_AIR = 1.225  # kg/m^3 (ISA sea-level)
        self._GRID_RES = 512   # internal resolution for frontal-area estimate

    @property
    def num_objectives(self):
        return len(self.objectives)

    @property
    def objective_short_names(self):
        return [OBJECTIVE_SHORT_NAMES[obj] for obj in self.objectives]

    @staticmethod
    def _silhouette_area(mesh: trimesh.Trimesh, axis: str = 'x', n1=4096, n2=4096) -> float:
        """
        Silhouette area projected onto the plane perpendicular to `axis`, by ray casting.
            axis='x' -> frontal area  (YZ-plane, rays along +X)
            axis='z' -> planform area (XY-plane, rays along +Z, top-down)
        """
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        o0, o1 = [i for i in range(3) if i != axis_idx]  # the two in-plane axes

        bmin, bmax = mesh.bounds
        a = np.linspace(bmin[o0], bmax[o0], n1, dtype=np.float64)
        b = np.linspace(bmin[o1], bmax[o1], n2, dtype=np.float64)
        AA, BB = np.meshgrid(a, b, indexing='xy')

        eps = 1e-6 * float((bmax - bmin).sum() + 1.0)
        origins = np.zeros((AA.size, 3), dtype=np.float64)
        origins[:, axis_idx] = bmin[axis_idx] - eps
        origins[:, o0] = AA.ravel()
        origins[:, o1] = BB.ravel()

        direction = np.zeros(3); direction[axis_idx] = 1.0
        directions = np.tile(direction[None, :], (origins.shape[0], 1))

        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh) \
                    if trimesh.ray.has_embree \
                    else trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        covered = np.zeros(origins.shape[0], dtype=bool)
        CHUNK = 200_000
        for s in range(0, origins.shape[0], CHUNK):
            covered[s:s+CHUNK] = intersector.intersects_any(origins[s:s+CHUNK], directions[s:s+CHUNK])

        cell_area = ((bmax[o0] - bmin[o0]) / (n1 - 1)) * ((bmax[o1] - bmin[o1]) / (n2 - 1))
        return float(covered.sum() * cell_area)

    @staticmethod
    def frontal_area(mesh, nx=4096, ny=4096):
        return ObjectiveEvaluator._silhouette_area(mesh, axis='x', n1=nx, n2=ny)

    @staticmethod
    def planform_area(mesh, nx=4096, ny=4096):
        return ObjectiveEvaluator._silhouette_area(mesh, axis='z', n1=nx, n2=ny)

    def force_to_coefficient(self, force_newtons, area_m2):
        """Standard area-based coefficient: F / (0.5 rho V^2 A_ref)."""
        V = float(self.data["stream_velocity"])  # m/s
        q = 0.5 * self._RHO_AIR * (V ** 2)       # dynamic pressure [Pa]
        return force_newtons / (q * area_m2)
 
    # return [objective_value] lower is better
    @retry(times=10, failed_return=None, exceptions=(HTTPError), backoff_factor=2)
    def evaluate_one(self, mesh, retry_attempt):

        if mesh.body_count != 1 or not mesh.is_watertight:
            return torch.full((self.num_objectives,), float('inf'))

        with tempfile.NamedTemporaryFile(mode='wb+', delete=True, suffix='.stl') as f:
            mesh.export(f.name)
            f.seek(0)
            files = {"design_stl": (f.name, f)}
            r = httpx.post(self.url, files=files, data=self.data, timeout=120.0)

        r.raise_for_status()

        with numpy.load(io.BytesIO(r.content)) as output_data:
            output_dict = {key: output_data[key] for key in output_data.keys()}
        objective_value = torch.zeros(self.num_objectives)
        for i, objective in enumerate(self.objectives):
            if objective == "drag-coefficient":
                drag_force = output_dict["drag_force"]
                frontal_area = self.frontal_area(mesh)
                drag_coefficient = self.force_to_coefficient(drag_force, frontal_area)
                objective_value[i] = drag_coefficient.item()
            elif objective == "lift-coefficient":
                lift_force = output_dict["lift_force"]
                lift_coefficient_frontal = self.force_to_coefficient(lift_force, self.frontal_area(mesh))
                lift_coefficient_planform = self.force_to_coefficient(lift_force, self.planform_area(mesh))
                lift_coefficient = (lift_coefficient_frontal + lift_coefficient_planform) / 2
                objective_value[i] = lift_coefficient.item()
            elif objective == "lift-to-drag-ratio":
                frontal_area = self.frontal_area(mesh)
                planform_area = self.planform_area(mesh)
                cl = self.force_to_coefficient(output_dict["lift_force"], planform_area).item()
                cd = self.force_to_coefficient(output_dict["drag_force"], frontal_area).item()
                objective_value[i] = cl / cd
            elif objective == "drag-force":
                objective_value[i] = output_dict["drag_force"].item()
            elif objective == "lift-force":
                objective_value[i] = output_dict["lift_force"].item()
            elif objective == "scaled-drag-force":
                drag_force = output_dict["drag_force"].item()  # range ~ [0, 200]
                objective_value[i] = (drag_force - 200) / 200  # normalize to [-1, 0]
            elif objective == "scaled-lift-force":
                lift_force = output_dict["lift_force"].item()  # range ~ [-100, 100]
                objective_value[i] = (lift_force - 100) / 200  # normalize to [-1, 0]
            else:
                raise ValueError(f"Unknown objective: {objective}")
        
        return objective_value
    
    def evaluate_batch(self, meshes, max_workers=8):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            objective_values = list(executor.map(self.evaluate_one, meshes))
        return objective_values

    def __call__(self, meshes):
        objective_values = self.evaluate_batch(meshes)
        objective_values = torch.stack(objective_values)
        return objective_values