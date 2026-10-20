"""
Helpers for placing an imported STL inside the LBM duct and previewing it.

The model is loaded centered at the origin of the test volume and positioned
interactively (drag/rotate gizmo) via place_model_interactively(); the result is
saved to model_transformations.json and reused by get_placed_mesh(). Run this file
directly to reposition.
"""

from pathlib import Path
import json
import os
import sys
import numpy as np

try:
    import trimesh
except Exception:
    trimesh = None
try:
    import plotly.graph_objects as go
except Exception:
    go = None


# User-editable placement parameters.
DEFAULT_STL_PATH = ""
MODEL_SCALE = 1.0                   # STL is already in meters (use 0.001 for mm)
CENTER_MODEL_AT_ORIGIN = True
AUTO_FIT_TO_DOMAIN = False
AUTO_FIT_MARGIN = 0.85

# Last gizmo placement, shared by this preview, the LBM solver, and openfoam_case.py.
PLACEMENT_FILE = Path(__file__).parent / "model_transformations.json"


def prompt_for_stl_path(default_path=DEFAULT_STL_PATH):
    """Return a mesh path selected by GUI/text prompt, or None if skipped."""
    candidate = Path(default_path).expanduser()
    if candidate.is_file():
        return candidate

    if trimesh is None:
        return None

    response = ""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        filetypes = [("STL files", "*.stl *.STL"), ("All files", "*")]
        response = filedialog.askopenfilename(
            title="Select STL file to import", filetypes=filetypes
        )
        root.destroy()
    except Exception:
        # Never block on input() without a terminal (e.g. a server job).
        if sys.stdin is not None and sys.stdin.isatty():
            try:
                response = input("Enter path to STL file to import (or press Enter to skip): ").strip()
            except Exception:
                response = ""
        else:
            print("No STL path given and no interactive terminal available.")

    if not response:
        return None

    selected = Path(os.path.expanduser(response))
    return selected if selected.is_file() else None


def load_transformed_mesh(
    stl_path,
    domain_lengths,
    scale=MODEL_SCALE,
    center_at_origin=CENTER_MODEL_AT_ORIGIN,
    auto_fit=AUTO_FIT_TO_DOMAIN,
    auto_fit_margin=AUTO_FIT_MARGIN,
):
    """Load an STL, scaled and centered at the test-volume origin (no rotation)."""
    if trimesh is None:
        raise RuntimeError("trimesh is not available")

    mesh = trimesh.load(Path(stl_path), force="mesh")
    if mesh.is_empty:
        raise ValueError("Loaded mesh is empty")

    mesh = mesh.copy()
    if center_at_origin:
        mesh.apply_translation(-mesh.bounding_box.centroid)

    if auto_fit:
        extents = np.maximum(mesh.extents, 1e-12)
        fit_scale = auto_fit_margin * np.min(np.asarray(domain_lengths) / extents)
        mesh.apply_scale(fit_scale)

    mesh.apply_scale(scale)
    return mesh


def build_mesh_mask(mesh, nx, ny, nz, dx):
    """Voxelize the transformed mesh by testing lattice cell centers."""
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dx
    zs = (np.arange(nz) + 0.5) * dx
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])

    try:
        inside = mesh.contains(points)
    except Exception:
        inside = trimesh.proximity.signed_distance(mesh, points) >= 0.0

    return inside.reshape((nx, ny, nz))


def apply_model_to_wall_mask(wall, dx, domain_lengths, stl_path=None):
    """
    Load, transform, and rasterize the configured STL into the solid mask.

    Returns
    -------
    wall_out, mesh
    """
    if trimesh is None:
        print("(trimesh not available - model import disabled. Install 'trimesh' to enable.)")
        return wall, None

    resolved_path = Path(stl_path) if stl_path else prompt_for_stl_path()
    if resolved_path is None:
        return wall, None

    if not resolved_path.is_file():
        print(f"STL file not found: {resolved_path}. Skipping import.")
        return wall, None

    try:
        mesh = get_placed_mesh(resolved_path, domain_lengths)
        mesh_mask = build_mesh_mask(mesh, *wall.shape, dx)
    except Exception as exc:
        print(f"Error loading STL - skipping import: {exc}")
        return wall, None

    wall_out = np.logical_or(wall, mesh_mask)
    print(f"Imported STL: marked {int(mesh_mask.sum())} cells as solid from '{resolved_path}'")
    return wall_out, mesh


def preview_model_in_domain(mesh, domain_lengths):
    """Preview the transformed mesh and duct using Plotly.

    Domain lengths are in meters. Plotly is required; if not installed a
    RuntimeError is raised.
    """
    if go is None:
        raise RuntimeError("Plotly is required for preview_model_in_domain (install plotly)")

    lx, ly, lz = domain_lengths
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=int)

    # Mesh3d expects i,j,k arrays for triangle indices
    if faces.size == 0:
        raise RuntimeError("Mesh contains no faces to render")
    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    mesh_trace = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=i, j=j, k=k,
        opacity=0.6,
        color='orange',
        flatshading=True,
        name='STL mesh',
        showscale=False,
    )

    # Build duct edges as a single line trace with None separators
    corners = np.array([
        [0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0],
        [0, 0, lz], [lx, 0, lz], [lx, ly, lz], [0, ly, lz],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs, ys, zs = [], [], []
    for e in edges:
        p0, p1 = corners[e[0]], corners[e[1]]
        xs += [p0[0], p1[0], None]
        ys += [p0[1], p1[1], None]
        zs += [p0[2], p1[2], None]

    edge_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color='black', width=2),
        name='duct edges',
    )

    fig = go.Figure(data=[mesh_trace, edge_trace])
    fig.update_layout(
        title='Transformed Model Inside Fluid Volume',
        scene=dict(
            xaxis_title='x [m]', yaxis_title='y [m]', zaxis_title='z [m]',
            xaxis=dict(range=[0, lx]),
            yaxis=dict(range=[0, ly]),
            zaxis=dict(range=[0, lz]),
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show(renderer="browser")


def load_saved_placement():
    """Return the saved 4x4 placement matrix (applies to any STL), or None."""
    try:
        data = json.loads(PLACEMENT_FILE.read_text())
        return np.array(data["matrix"], dtype=float).reshape(4, 4)
    except Exception:
        return None


def save_placement(matrix):
    PLACEMENT_FILE.write_text(json.dumps({"matrix": np.asarray(matrix).tolist()}, indent=2))


def place_model_interactively(mesh, domain_lengths):
    """Drag/rotate the mesh in a PyVista window; return it with the placement applied.

    Starts from the saved placement (if any) and saves the result on close.
    """
    import pyvista as pv

    saved = load_saved_placement()
    start = np.eye(4) if saved is None else saved
    mesh = mesh.copy()
    mesh.apply_transform(start)

    lx, ly, lz = domain_lengths
    faces = np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices), faces)

    pl = pv.Plotter(title="Drag arrows to move, rings to rotate - close window when done")
    actor = pl.add_mesh(poly, color="orange", opacity=0.8)
    pl.add_mesh(pv.Box(bounds=(0, lx, 0, ly, 0, lz)), color="lightblue", opacity=0.08)
    pl.add_axes()
    pl.add_affine_transform_widget(actor)
    # Must come after the widget: adding actors resets the grid to scene bounds.
    pl.show_bounds(
        bounds=(0, lx, 0, ly, 0, lz), grid="back", location="outer", fmt="%.3f",
        n_xlabels=5, n_ylabels=3, n_zlabels=3,
        xtitle="X [m]", ytitle="Y [m]", ztitle="Z [m]",
    )
    pl.show()

    delta = np.asarray(actor.user_matrix, dtype=float)
    mesh.apply_transform(delta)
    save_placement(delta @ start)
    print(f"Saved placement -> {PLACEMENT_FILE}")
    return mesh


def get_placed_mesh(stl_path, domain_lengths, reposition=False):
    """Mesh placed per the saved placement; opens the gizmo if none or reposition=True.

    The placement is STL-agnostic: a new model inherits the last pose, so only
    adjust it (reposition=True) if it needs to move.
    """
    mesh = load_transformed_mesh(stl_path, domain_lengths)
    saved = None if reposition else load_saved_placement()
    if saved is None:
        return place_model_interactively(mesh, domain_lengths)
    mesh.apply_transform(saved)
    return mesh


if __name__ == "__main__":
    if trimesh is None:
        raise SystemExit("trimesh is not available")

    DOMAIN = (0.10, 0.05, 0.05)
    path = prompt_for_stl_path()
    if path is None:
        raise SystemExit("No STL selected.")

    placed_mesh = get_placed_mesh(path, DOMAIN, reposition=True)
    print("Final bounds [m]:")
    print(np.round(placed_mesh.bounds, 4))
