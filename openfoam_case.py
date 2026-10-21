"""
OpenFOAM case generator for the wind tunnel.

Reuses fluid_volume_model.py for geometry placement (saved in
model_transformations.json; pass --reposition to reopen the gizmo).

Generates a complete incompressible RANS case (simpleFoam, k-omega SST,
air at INLET_VELOCITY m/s) targeting OpenFOAM 11 (Foundation line), then
optionally meshes and solves:

    python openfoam_case.py --stl model.stl   # write case, then mesh
    python openfoam_case.py --dicts-only      # write the case only
    python openfoam_case.py --solve           # also run simpleFoam after meshing

Meshing/solving call blockMesh, snappyHexMesh and simpleFoam from PATH
(blueCFD-Core on Windows, or the pinned container image in CI).
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

# ── Flow conditions (incompressible RANS: simpleFoam + k-omega SST) ───────────
INLET_VELOCITY = 10.0   # m/s, in +x
NU = 1.5e-5             # kinematic viscosity of air (~20 °C) [m²/s]
RHO_INF = 1.225         # scales reported forces only; the solve is incompressible [kg/m³]
END_TIME = 1500         # SIMPLE iterations
WRITE_INTERVAL = 250    # field write cadence [iterations]

# Inlet turbulence from standard duct correlations:
#   I = 0.16 Re_Dh^(-1/8)          (fully-developed duct flow)
#   k = 1.5 (U I)^2
#   omega = sqrt(k) / (Cmu^0.25 * l),  l = 0.07 Dh
HYDRAULIC_DIAMETER = 2 * Ly * Lz / (Ly + Lz)   # 0.05 m for the default duct
TURBULENT_INTENSITY = 0.16 * (INLET_VELOCITY * HYDRAULIC_DIAMETER / NU) ** -0.125
K_INLET = 1.5 * (INLET_VELOCITY * TURBULENT_INTENSITY) ** 2
OMEGA_INLET = K_INLET ** 0.5 / (0.09 ** 0.25 * 0.07 * HYDRAULIC_DIAMETER)

CASE_DIR = Path(__file__).parent / "case"
STL_OUT  = CASE_DIR / "constant" / "triSurface" / "model.stl"
SYS_DIR  = CASE_DIR / "system"
CONST_DIR   = CASE_DIR / "constant"
FIELD_DIR   = CASE_DIR / "0"


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
def write_dicts(location_in_mesh, mesh=None):
    SYS_DIR.mkdir(parents=True, exist_ok=True)
    CONST_DIR.mkdir(parents=True, exist_ok=True)
    FIELD_DIR.mkdir(parents=True, exist_ok=True)
    ref = _reference_quantities(mesh)
    _write_control_dict(ref)
    _write_block_mesh_dict()
    _write_snappy_hex_mesh_dict(location_in_mesh)
    _write_mesh_quality_dict()
    _write_fv_schemes()
    _write_fv_solution()
    _write_transport_properties()
    _write_turbulence_properties()
    _write_initial_fields()
    print(f"Wrote dictionaries  →  {CASE_DIR}")
    print("  system:   controlDict, blockMeshDict, snappyHexMeshDict,"
          " meshQualityDict, fvSchemes, fvSolution")
    print("  constant: transportProperties, turbulenceProperties")
    print("  0/:       U, p, k, omega, nut")
    print(f"  inlet: U = {INLET_VELOCITY} m/s, I = {TURBULENT_INTENSITY:.3f},"
          f" k = {K_INLET:.4g} m²/s², omega = {OMEGA_INLET:.4g} 1/s")
    print(f"  reference: lRef = {ref[0]:.4g} m, Aref = {ref[1]:.4g} m²")


def _reference_quantities(mesh):
    """lRef, Aref, CofR for the forceCoeffs function object.

    lRef = streamwise (x) extent, Aref = x-extent × z-extent, CofR = bounding-box
    centre. With mesh=None, falls back to the bounds of the shipped model.stl.
    """
    if mesh is None:
        lo = np.array([0.03279, 0.00692, 0.02189])   # shipped model.stl bounds [m]
        hi = np.array([0.06738, 0.04308, 0.02695])
    else:
        lo, hi = np.asarray(mesh.bounds, dtype=float)
    chord = hi[0] - lo[0]
    span = hi[2] - lo[2]
    return chord, chord * span, tuple(0.5 * (lo + hi))


def _write_control_dict(ref):
    l_ref, a_ref, (cx, cy, cz) = ref
    content = _foam_header("dictionary", "controlDict") + textwrap.dedent(f"""\

        application     simpleFoam;

        startFrom       startTime;
        startTime       0;
        stopAt          endTime;
        endTime         {END_TIME};
        deltaT          1;

        writeControl    timeStep;
        writeInterval   {WRITE_INTERVAL};
        purgeWrite      0;
        writeFormat     ascii;
        writePrecision  6;
        writeCompression off;
        timeFormat      general;
        timePrecision   6;
        runTimeModifiable true;

        functions
        {{
            forces
            {{
                type            forces;
                libs            ("libforces.so");
                writeControl    timeStep;
                writeInterval   50;
                patches         (model);
                rho             rhoInf;
                rhoInf          {RHO_INF};
                CofR            ({cx:.6g} {cy:.6g} {cz:.6g});
            }}

            coefficients
            {{
                type            forceCoeffs;
                libs            ("libforces.so");
                writeControl    timeStep;
                writeInterval   50;
                patches         (model);
                rho             rhoInf;
                rhoInf          {RHO_INF};
                liftDir         (0 1 0);
                dragDir         (1 0 0);
                pitchAxis       (0 0 1);
                magUInf         {INLET_VELOCITY};
                lRef            {l_ref:.6g};
                Aref            {a_ref:.6g};
                CofR            ({cx:.6g} {cy:.6g} {cz:.6g});
            }}
        }}
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

            resolveFeatureAngle 30;

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

        meshQualityControls
        {{
            #include "meshQualityDict"
        }}

        writeFlags
        (
            scalarLevels
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


# Solver-side dictionaries.  Note for ESI-line images (v2212+): their
# snappyHexMesh also requires fvSchemes/fvSolution to exist before meshing, so
# emitting them here keeps the case runnable on both OpenFOAM lines.
def _write_fv_schemes():
    content = _foam_header("dictionary", "fvSchemes") + textwrap.dedent("""\

        ddtSchemes
        {
            default         steadyState;
        }

        gradSchemes
        {
            default         Gauss linear;
            grad(U)         cellLimited Gauss linear 1;
        }

        divSchemes
        {
            default         none;
            div(phi,U)      bounded Gauss linearUpwind grad(U);
            div(phi,k)      bounded Gauss upwind;
            div(phi,omega)  bounded Gauss upwind;
            div((nuEff*dev2(T(grad(U))))) Gauss linear;
        }

        laplacianSchemes
        {
            default         Gauss linear corrected;
        }

        interpolationSchemes
        {
            default         linear;
        }

        snGradSchemes
        {
            default         corrected;
        }

        wallDist
        {
            method          meshWave;
        }
        """)
    (SYS_DIR / "fvSchemes").write_text(content)


def _write_fv_solution():
    content = _foam_header("dictionary", "fvSolution") + textwrap.dedent("""\

        solvers
        {
            p
            {
                solver          GAMG;
                tolerance       1e-7;
                relTol          0.01;
                smoother        GaussSeidel;
            }

            "(U|k|omega)"
            {
                solver          smoothSolver;
                smoother        symGaussSeidel;
                tolerance       1e-8;
                relTol          0.1;
            }
        }

        SIMPLE
        {
            nNonOrthogonalCorrectors 1;
            consistent      yes;

            residualControl
            {
                p               1e-4;
                U               1e-5;
                "(k|omega)"     1e-5;
            }
        }

        relaxationFactors
        {
            equations
            {
                U               0.7;
                k               0.7;
                omega           0.7;
            }
        }
        """)
    (SYS_DIR / "fvSolution").write_text(content)


def _write_transport_properties():
    content = _foam_header("dictionary", "transportProperties") + textwrap.dedent(f"""\

        transportModel  Newtonian;
        nu              [0 2 -1 0 0 0 0] {NU};
        """)
    (CONST_DIR / "transportProperties").write_text(content)


def _write_turbulence_properties():
    # OpenFOAM 11 name/format; OpenFOAM 12 renamed this file to
    # constant/momentumTransport with keyword "model" instead of "RASModel".
    content = _foam_header("dictionary", "turbulenceProperties") + textwrap.dedent("""\

        simulationType  RAS;

        RAS
        {
            RASModel        kOmegaSST;
            turbulence      on;
            printCoeffs     on;
        }
        """)
    (CONST_DIR / "turbulenceProperties").write_text(content)


# ── 0/ initial fields ─────────────────────────────────────────────────────────
def _write_field(name, foam_class, dims, internal, boundaries):
    """Write a 0/ field file. boundaries: [(patch, [body lines]), ...]."""
    body = [f"dimensions      [{dims}];", f"internalField   {internal};", "",
            "boundaryField", "{"]
    for patch, spec in boundaries:
        body.append(f"    {patch}")
        body.append("    {")
        body.extend(f"        {line}" for line in spec)
        body.append("    }")
    body.append("}")
    (FIELD_DIR / name).write_text(
        _foam_header(foam_class, name) + "\n".join(body) + "\n")


def _write_initial_fields():
    u = f"uniform ({INLET_VELOCITY:.6g} 0 0)"
    k = f"uniform {K_INLET:.6g}"
    w = f"uniform {OMEGA_INLET:.6g}"

    _write_field("U", "volVectorField", "0 1 -1 0 0 0 0", u, [
        ("inlet",  ["type fixedValue;", f"value {u};"]),
        ("outlet", ["type inletOutlet;", "inletValue uniform (0 0 0);", f"value {u};"]),
        ("walls",  ["type noSlip;"]),
        ("model",  ["type noSlip;"]),
    ])
    # Kinematic pressure (m²/s²), as used by OpenFOAM 11 incompressible solvers.
    _write_field("p", "volScalarField", "0 2 -2 0 0 0 0", "uniform 0", [
        ("inlet",  ["type zeroGradient;"]),
        ("outlet", ["type fixedValue;", "value uniform 0;"]),
        ("walls",  ["type zeroGradient;"]),
        ("model",  ["type zeroGradient;"]),
    ])
    _write_field("k", "volScalarField", "0 2 -2 0 0 0 0", k, [
        ("inlet",  ["type fixedValue;", f"value {k};"]),
        ("outlet", ["type inletOutlet;", f"inletValue {k};", f"value {k};"]),
        ("walls",  ["type kqRWallFunction;", f"value {k};"]),
        ("model",  ["type kqRWallFunction;", f"value {k};"]),
    ])
    _write_field("omega", "volScalarField", "0 0 -1 0 0 0 0", w, [
        ("inlet",  ["type fixedValue;", f"value {w};"]),
        ("outlet", ["type inletOutlet;", f"inletValue {w};", f"value {w};"]),
        ("walls",  ["type omegaWallFunction;", f"value {w};"]),
        ("model",  ["type omegaWallFunction;", f"value {w};"]),
    ])
    _write_field("nut", "volScalarField", "0 2 -1 0 0 0 0", "uniform 0", [
        ("inlet",  ["type calculated;", "value uniform 0;"]),
        ("outlet", ["type calculated;", "value uniform 0;"]),
        ("walls",  ["type nutkWallFunction;", "value uniform 0;"]),
        ("model",  ["type nutkWallFunction;", "value uniform 0;"]),
    ])


# ── Step 3: run blockMesh + snappyHexMesh ─────────────────────────────────────
def run_meshing():
    """
    Call blockMesh then snappyHexMesh; the executables must be on PATH
    (blueCFD-Core on Windows, or the pinned OpenFOAM container image).
    """
    case = str(CASE_DIR)

    print("Running blockMesh …")
    subprocess.run(["blockMesh", "-case", case], check=True)

    print("Running snappyHexMesh …")
    subprocess.run(["snappyHexMesh", "-case", case, "-overwrite"], check=True)

    poly_mesh = CASE_DIR / "constant" / "polyMesh"
    print(f"Meshing complete  →  {poly_mesh}")


# ── Step 4: run simpleFoam ────────────────────────────────────────────────────
def run_solve():
    """Steady incompressible RANS; residuals go to the log, forces to postProcessing/."""
    print("Running simpleFoam …")
    subprocess.run(["simpleFoam", "-case", str(CASE_DIR)], check=True)
    print(f"Solving complete  →  {CASE_DIR / 'postProcessing'}")


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    stl = argv[argv.index("--stl") + 1] if "--stl" in argv else None
    mesh = export_stl(stl_path=stl, reposition="--reposition" in argv)
    write_dicts(find_location_in_mesh(mesh), mesh)
    if "--dicts-only" in argv:
        raise SystemExit(0)
    run_meshing()
    if "--solve" in argv:
        run_solve()
