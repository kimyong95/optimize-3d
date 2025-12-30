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
    "drag-coefficient": "dc",
    "drag-force": "df",
    "lift-force": "lf",
    "scaled-drag-force": "sdf",
    "scaled-lift-force": "slf",
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
    def frontal_area(mesh: trimesh.Trimesh,
                            nx=4096, ny=4096,
                            axis='x') -> float:
        """
        Robust silhouette area onto the YZ-plane (axis='x' flow).
        Works best for closed/watertight meshes but tolerates many non-manifold quirks.
        Resolution (nx, ny) controls accuracy.
        """
        # Project to YZ plane, so rays go along +X and sample over (Y,Z).
        assert axis == 'x'
        # Mesh bounds to define sampling window in Y,Z
        (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
        y = np.linspace(ymin, ymax, nx, dtype=np.float64)
        z = np.linspace(zmin, zmax, ny, dtype=np.float64)
        YY, ZZ = np.meshgrid(y, z, indexing='xy')

        # Ray origins at xmin - eps to be in front of geometry
        eps = 1e-6 * (xmax - xmin + ymax - ymin + zmax - zmin + 1.0)
        origins = np.column_stack([np.full(YY.size, xmin - eps),
                                YY.ravel(), ZZ.ravel()])
        directions = np.tile(np.array([[1.0, 0.0, 0.0]]), (origins.shape[0], 1))

        # Fast intersector (uses Embree if available)
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh) \
                    if trimesh.ray.has_embree \
                    else trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        # Query any hit; returns hit locations per ray (ragged)
        # For speed/memory, do in chunks
        covered = np.zeros(origins.shape[0], dtype=bool)
        CHUNK = 200_000
        for s in range(0, origins.shape[0], CHUNK):
            o = origins[s:s+CHUNK]
            d = directions[s:s+CHUNK]
            hits = intersector.intersects_any(o, d)   # boolean per ray
            covered[s:s+CHUNK] = hits

        # Pixel cell area in YZ
        cell_area = ((ymax - ymin) / (nx - 1)) * ((zmax - zmin) / (ny - 1))
        return float(covered.sum() * cell_area)

    # Convert drag force -> drag coefficient using the drag equation with per-mesh frontal area
    def force_to_coefficient(self, drag_force_newtons: numpy.ndarray, frontal_area_m2: numpy.ndarray) -> numpy.ndarray:
        V = float(self.data["stream_velocity"])  # m/s
        q = 0.5 * self._RHO_AIR * (V ** 2)       # dynamic pressure [Pa = N/m^2]
        # Guard against zeros
        frontal_area_m2 = numpy.clip(numpy.asarray(frontal_area_m2, dtype=numpy.float64), 1e-9, None)
        Cd = drag_force_newtons / (q * frontal_area_m2)
        return Cd

    # return [objective_value] lower is better
    @retry(times=10, failed_return=None, exceptions=(HTTPError), backoff_factor=2)
    def evaluate_one(self, mesh, retry_attempt):

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
            elif objective == "drag-force":
                objective_value[i] = output_dict["drag_force"].item()
            elif objective == "lift-force":
                objective_value[i] = output_dict["lift_force"].item()
            elif objective == "scaled-drag-force":
                drag_force = output_dict["drag_force"].item() # range ~ [0, 200]
                objective_value[i] = (drag_force - 200) / 200  # normalize to [-1, 0]
            elif objective == "scaled-lift-force":
                lift_force = output_dict["lift_force"].item() # range ~ [-100, 100]
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