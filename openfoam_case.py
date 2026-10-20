"""
OpenFOAM case generator for the wind tunnel.

Reuses fluid_volume_model.py for geometry placement (saved in
model_transformations.json; pass --reposition to reopen the gizmo).
Run this script to prepare an OpenFOAM case directory, then call:

    blockMesh -case case
    snappyHexMesh -case case -overwrite
"""

from pathlib import Path
import os
import subprocess
import textwrap
import numpy as np

from fluid_volume_model import get_placed_mesh, prompt_for_stl_path

# ── Domain (must match windtunnel.py) ─────────────────────────────────────────
Lx, Ly, Lz = 0.10, 0.05, 0.05   # duct dimensions [m]
DOMAIN = (Lx, Ly, Lz)

# Background cell size for blockMesh.  snappyHexMesh refines further near the
# model surface, so this can be coarse (5 mm gives 20×10×10 = 2 000 hex cells).
BACKGROUND_CELL_SIZE = 0.005  # m

# Refinement levels applied at the model surface by snappyHexMesh.
# level N → cell size = BACKGROUND_CELL_SIZE / 2^N at that surface.
REFINEMENT_LEVEL_MIN = 2
REFINEMENT_LEVEL_MAX = 3

CASE_DIR = Path(__file__).parent / "case"
STL_OUT  = CASE_DIR / "constant" / "triSurface" / "model.stl"
SYS_DIR  = CASE_DIR / "system"


# ── OpenFOAM file header ──────────────────────────────────────────────────────
def _foam_header(foam_class, object_name):
    return textwrap.dedent(f"""\
        FoamFile
        {{
            version     2.0;
            format      ascii;
            class       {foam_class};
            object      {object_name};
        }}
        // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
        """)


# ── Step 1: export STL ────────────────────────────────────────────────────────
def export_stl(stl_path=None, reposition=False):
    if stl_path is None:
        stl_path = os.environ.get("STL_PATH") or prompt_for_stl_path()
    if stl_path is None:
        raise SystemExit("No STL provided. Pass --stl PATH or set STL_PATH.")
    stl_path = Path(stl_path)

    mesh = get_placed_mesh(stl_path, DOMAIN, reposition=reposition)

    STL_OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(STL_OUT))

    bounds = np.array(mesh.bounds)
    print(f"Exported STL  →  {STL_OUT}")
    print(f"  bounds x: [{bounds[0,0]:.4f}, {bounds[1,0]:.4f}] m")
    print(f"  bounds y: [{bounds[0,1]:.4f}, {bounds[1,1]:.4f}] m")
    print(f"  bounds z: [{bounds[0,2]:.4f}, {bounds[1,2]:.4f}] m")
    print(f"  domain   : {Lx} × {Ly} × {Lz} m")

    outside = (
        bounds[0].min() < 0
        or bounds[1, 0] > Lx
        or bounds[1, 1] > Ly
        or bounds[1, 2] > Lz
    )
    if outside:
        print("  WARNING: mesh extends outside the duct — re-run with --reposition")
    else:
        print("  Mesh is fully inside the duct domain.")

    return mesh


def find_location_in_mesh(mesh):
    """Pick a point in the fluid: inside the duct, outside the model, max clearance.

    snappyHexMesh keeps the region containing this point, so it must not be inside
    the model or sit on a background cell face. Candidates are offset from the
    blockMesh cell faces; clearance is the distance to the nearest wall or model
    surface. Ties go to the lowest x (upstream).
    """
    import trimesh

    n = [int(round(L / BACKGROUND_CELL_SIZE)) for L in DOMAIN]
    axes = [(np.arange(k) + 0.46) * BACKGROUND_CELL_SIZE for k in n]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    pts = pts[~mesh.contains(pts)]
    if pts.size == 0:
        raise SystemExit(
            "No fluid candidate points — the model fills the duct. "
            "Re-run with --reposition or shrink the model."
        )

    wall_clear = np.minimum(pts, np.array(DOMAIN) - pts).min(axis=1)
    _, surf_clear, _ = trimesh.proximity.closest_point(mesh, pts)
    clearance = np.minimum(wall_clear, surf_clear)

    best = np.flatnonzero(clearance >= clearance.max() - 1e-9)
    point = pts[best[np.argmin(pts[best, 0])]]
    if clearance.max() < BACKGROUND_CELL_SIZE:
        raise SystemExit(
            f"No fluid point with >= {BACKGROUND_CELL_SIZE} m clearance: "
            "the model fills the duct. Re-run with --reposition."
        )
    print(f"locationInMesh = ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})  "
          f"(clearance {clearance.max() * 1000:.1f} mm)")
    return tuple(float(v) for v in point)


# ── Step 2: write OpenFOAM dictionaries ───────────────────────────────────────
def write_dicts(location_in_mesh):
    SYS_DIR.mkdir(parents=True, exist_ok=True)
    _write_control_dict()
    _write_block_mesh_dict()
    _write_snappy_hex_mesh_dict(location_in_mesh)
    _write_mesh_quality_dict()
    print(f"Wrote dictionaries  →  {SYS_DIR}")
    print("  controlDict, blockMeshDict, snappyHexMeshDict, meshQualityDict")


def _write_control_dict():
    content = _foam_header("dictionary", "controlDict") + textwrap.dedent("""\

        application     snappyHexMesh;

        startFrom       startTime;
        startTime       0;
        stopAt          endTime;
        endTime         1;
        deltaT          1;

        writeControl    timeStep;
        writeInterval   1;
        purgeWrite      0;
        writeFormat     ascii;
        writePrecision  6;
        writeCompression off;
        timeFormat      general;
        timePrecision   6;
        runTimeModifiable true;
        """)
    (SYS_DIR / "controlDict").write_text(content)


def _write_block_mesh_dict():
    nx = max(1, round(Lx / BACKGROUND_CELL_SIZE))
    ny = max(1, round(Ly / BACKGROUND_CELL_SIZE))
    nz = max(1, round(Lz / BACKGROUND_CELL_SIZE))

    # Vertices of the duct box.
    # Index layout:
    #   0 (0,0,0)   1 (Lx,0,0)   2 (Lx,Ly,0)  3 (0,Ly,0)
    #   4 (0,0,Lz)  5 (Lx,0,Lz)  6 (Lx,Ly,Lz) 7 (0,Ly,Lz)
    #
    # Face orderings use (v1-v0)×(v2-v0) to give outward normal:
    #   inlet  x=0  → (0 4 7 3)   normal: -x
    #   outlet x=Lx → (1 2 6 5)   normal: +x
    #   bottom y=0  → (0 1 5 4)   normal: -y  (wall)
    #   top    y=Ly → (3 7 6 2)   normal: +y  (wall)
    #   front  z=0  → (0 3 2 1)   normal: -z  (wall)
    #   back   z=Lz → (4 5 6 7)   normal: +z  (wall)

    content = _foam_header("dictionary", "blockMeshDict") + textwrap.dedent(f"""\

        scale 1;

        vertices
        (
            ( 0    0    0   )   // 0
            ( {Lx}  0    0   )   // 1
            ( {Lx}  {Ly}  0   )   // 2
            ( 0    {Ly}  0   )   // 3
            ( 0    0    {Lz} )   // 4
            ( {Lx}  0    {Lz} )   // 5
            ( {Lx}  {Ly}  {Lz} )   // 6
            ( 0    {Ly}  {Lz} )   // 7
        );

        blocks
        (
            hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
        );

        boundary
        (
            inlet
            {{
                type patch;
                faces ( (0 4 7 3) );
            }}
            outlet
            {{
                type patch;
                faces ( (1 2 6 5) );
            }}
            walls
            {{
                type wall;
                faces
                (
                    (0 1 5 4)   // bottom y = 0
                    (3 7 6 2)   // top    y = {Ly}
                    (0 3 2 1)   // front  z = 0
                    (4 5 6 7)   // back   z = {Lz}
                );
            }}
        );
        """)
    (SYS_DIR / "blockMeshDict").write_text(content)


def _write_snappy_hex_mesh_dict(location_in_mesh):
    lx, ly, lz = location_in_mesh
    rmin = REFINEMENT_LEVEL_MIN
    rmax = REFINEMENT_LEVEL_MAX

    content = _foam_header("dictionary", "snappyHexMeshDict") + textwrap.dedent(f"""\

        castellatedMesh true;
        snap            true;
        addLayers       false;

        geometry
        {{
            model
            {{
                type triSurfaceMesh;
                file "model.stl";
            }}
        }}

        castellatedMeshControls
        {{
            maxLocalCells       1000000;
            maxGlobalCells      2000000;
            minRefinementCells  10;
            maxLoadUnbalance    0.10;
            nCellsBetweenLevels 3;

            features ();

            refinementSurfaces
            {{
                model
                {{
                    level ({rmin} {rmax});
                }}
            }}

            refinementRegions {{}}

            locationInMesh ({lx} {ly} {lz});

            allowFreeStandingZoneFaces true;
        }}

        snapControls
        {{
            nSmoothPatch      3;
            tolerance         2.0;
            nSolveIter        30;
            nRelaxIter        5;
            nFeatureSnapIter  10;
            implicitFeatureSnap false;
            explicitFeatureSnap false;
            multiRegionFeatureSnap false;
        }}

        addLayersControls
        {{
            relativeSizes           true;
            layers {{}}
            expansionRatio          1.0;
            finalLayerThickness     0.3;
            minThickness            0.1;
            nGrow                   0;
            featureAngle            60;
            nRelaxIter              3;
            nSmoothSurfaceNormals   1;
            nSmoothNormals          3;
            nSmoothThickness        10;
            maxFaceThicknessRatio   0.5;
            maxThicknessToMedialRatio 0.3;
            minMedialAxisAngle      90;
            nBufferCellsNoExtrude   0;
            nLayerIter              50;
        }}

        #include "meshQualityDict"

        writeFlags
        (
            scalarLevels
            meshQuality
            layerInfo
        );

        mergeTolerance 1e-6;
        """)
    (SYS_DIR / "snappyHexMeshDict").write_text(content)


def _write_mesh_quality_dict():
    content = _foam_header("dictionary", "meshQualityDict") + textwrap.dedent("""\

        maxNonOrtho             65;
        maxBoundarySkewness     20;
        maxInternalSkewness     4;
        maxConcave              80;
        minFlatness             0.5;
        minVol                  1e-13;
        minTetQuality           1e-15;
        minArea                 -1;
        minTwist                0.02;
        minDeterminant          0.001;
        minFaceWeight           0.02;
        minVolRatio             0.01;
        minTriangleTwist        -1;
        nSmoothScale            4;
        errorReduction          0.75;
        """)
    (SYS_DIR / "meshQualityDict").write_text(content)


# ── Step 3: run blockMesh + snappyHexMesh (blueCFD-Core) ──────────────────────
def run_meshing():
    """
    Call blockMesh then snappyHexMesh using blueCFD-Core's native Windows
    executables, which are expected to be on PATH after installing blueCFD-Core.
    """
    case = str(CASE_DIR)

    print("Running blockMesh …")
    subprocess.run(["blockMesh", "-case", case], check=True)

    print("Running snappyHexMesh …")
    subprocess.run(["snappyHexMesh", "-case", case, "-overwrite"], check=True)

    poly_mesh = CASE_DIR / "constant" / "polyMesh"
    print(f"Meshing complete  →  {poly_mesh}")


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    stl = argv[argv.index("--stl") + 1] if "--stl" in argv else None
    mesh = export_stl(stl_path=stl, reposition="--reposition" in argv)
    write_dicts(find_location_in_mesh(mesh))
    run_meshing()
