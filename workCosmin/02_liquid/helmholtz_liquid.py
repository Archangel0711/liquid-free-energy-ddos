#!/usr/bin/env python
"""
Absolute Helmholtz free energy of LIQUID W -- Route 1 (KineticHessian), self-contained.

Computes F(N, V, T) per atom of the liquid configuration in
structures/W-liquid.data, with NO comparison to any other phase and NO external
reference: the KineticHessian isosurface samples the true SNAP potential by NVE
(capturing the full anharmonic / liquid behaviour) while measuring the isosurface
in the Hessian mode basis of the input configuration. Its records therefore
restore to an *analytic* harmonic reference (HarmonicReferenceFreeEnergy,
requires_fit == False), so the free energy is absolute and needs no fit_S_alpha.

Because KineticHessian does not stack a kinetic column (unlike the pure Kinetic
displacer), its descriptor is the plain 55-dim SNAP vector and the model
parameters are simply Thetas = theta_SNAP -- identical bookkeeping to a Hessian
sampling.

    F_abs(T) = E0 + score_free_energy(T, theta_SNAP)['F']        [eV/atom]

with E0 = theta_SNAP . D_0 the static energy of the liquid snapshot, and the
absolute (classical, momentum-included) harmonic scale carried by the reference.

Run (samples on first call, then analyses; re-runs reuse the cached sampling):

    python helmholtz_liquid.py
    python helmholtz_liquid.py --calls 2000 --nalpha 20 --Tmax 6000
    python helmholtz_liquid.py --analyse-only          # just re-analyse existing data

NB: KineticHessian re-targets the energy every draw (~2N NVE steps) so it is slow
(~5-7 s/call for this cell). It supports mpirun, but only with a launcher that
matches the venv's mpi4py build (Intel MPI here, NOT the box's OpenMPI mpirun).
"""

import os
import sys
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(SCRIPT_DIR)

import yaml
from DescriptorDOS import DDOSManager, DDOSAnalyzer
from DescriptorDOS.workers.LAMMPSWorker import LAMMPSWorker


def is_running_with_mpi():
    for v in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMI_RANK",
              "I_MPI_JOB_SIZE", "SLURM_NTASKS"):
        if v in os.environ and os.environ[v] not in ("", "1"):
            return True
    return False


ELEMENT, STRUCTURE, MASS, MODEL = "W", "liquid", 183.84, "W"
ISO = "KineticHessian"

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--calls", type=int, default=200,
                    help="TOTAL NVE force calls per isosurface value (split over ranks)")
parser.add_argument("--nalpha", type=int, default=15)
parser.add_argument("--amin", type=float, default=0.5, help="min isosurface (~T/1000 K)")
parser.add_argument("--amax", type=float, default=6.0, help="max isosurface (~T/1000 K)")
parser.add_argument("--Tmin", type=float, default=300.0)
parser.add_argument("--Tmax", type=float, default=5000.0)
parser.add_argument("--nT", type=int, default=24)
parser.add_argument("--match-order", type=int, default=7)
parser.add_argument("--match-type", default="chebyshev")
parser.add_argument("--analyse-only", action="store_true",
                    help="skip sampling, analyse existing pkl")
parser.add_argument("--resample", action="store_true",
                    help="force re-sampling even if the pkl already exists")
parser.add_argument("--append", action="store_true")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

if is_running_with_mpi():
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
else:
    world = None
    rank = 0

input_configuration = f"structures/{ELEMENT}-{STRUCTURE}.data"
assert os.path.exists(input_configuration), f"{input_configuration} not found!"
dump_file = f"DDOSData-{ELEMENT}-{STRUCTURE}-{ISO}.pkl"
pkl_file = os.path.join("output", dump_file)

# ---------------------------------------------------------------------------
# 1) SAMPLE the liquid on the KineticHessian isosurface (NVE + Hessian basis)
# ---------------------------------------------------------------------------
need_run = args.resample or (not os.path.exists(pkl_file) and not args.analyse_only)
if need_run:
    extra_params = yaml.safe_load(open(f"models/{MODEL}/params.yaml"))
    iso_list = list(np.linspace(args.amin, args.amax, args.nalpha))
    if rank == 0:
        print(f"\n\t\tliquid W Helmholtz free energy (Route 1, {ISO})")
        print(f"\t\t{args.nalpha} alpha in [{args.amin},{args.amax}], "
              f"{args.calls} calls/alpha, config={input_configuration}\n")
    mgr = DDOSManager(
        world=world,
        worker=LAMMPSWorker,
        isosurface=ISO,
        element=ELEMENT,
        mass=MASS,
        default_calls=args.calls,
        lammps_input_configuration=input_configuration,
        dump_path="output",
        dump_file=dump_file,
        hessian_path="output",
        hessian_data=f"hessian-{ELEMENT}-{STRUCTURE}-kh.pkl",
        input_parameters={"IsoSurface": iso_list, "Strain": [0.0]},
        reconstruction_error=1.0e-7,
        append_data=args.append,
        verbose=args.verbose,
        **extra_params,
    )
    mgr.run()
    mgr.close()

# only rank 0 analyses
if world is not None:
    world.Barrier()
if rank != 0:
    from mpi4py import MPI
    MPI.Finalize()
    sys.exit(0)

# ---------------------------------------------------------------------------
# 2) ANALYSE -> absolute Helmholtz free energy (no fit_S_alpha, no other phase)
# ---------------------------------------------------------------------------
assert os.path.exists(pkl_file), f"{pkl_file} not found -- run without --analyse-only first!"
theta = np.loadtxt(f"models/{ELEMENT}/X_{ELEMENT}.snapcoeff", skiprows=6)

liq = DDOSAnalyzer(pkl_file, mass=MASS, verbose=True)
# KineticHessian carries an analytic harmonic reference -> no external fit needed
assert not liq.reference_free_energy.requires_fit, (
    "unexpected: this data needs an external reference (not KineticHessian?)")
assert theta.size == liq.D_0.size, (
    f"Theta dim {theta.size} != descriptor dim {liq.D_0.size}")

E0 = liq.cohesive_energy(Thetas=theta.reshape(1, -1))[0]      # static, eV/atom

Temperatures = np.linspace(args.Tmin, args.Tmax, args.nT)
liq.match_score(poly_order=args.match_order, poly_type=args.match_type)
n_a = len(liq.alphas)
order = min(5, max(1, n_a - 1))
F_vib = np.array([
    liq.score_free_energy(T, Thetas=theta.reshape(1, -1), alpha_poly_order=order)["F"][0]
    for T in Temperatures])
F_abs = E0 + F_vib                                            # ABSOLUTE Helmholtz F
F_harm = E0 + liq.reference_free_energy.free_energy(Temperatures)  # harmonic reference

print(f"\n\t\tLIQUID W  V/atom = {liq.volume_per_atom:.4f} A^3, N = {liq.N}")
print(f"\t\tstatic E0 (snapshot) = {1000*E0:.2f} meV/atom\n")
print("\t\t   T [K]   F_abs [meV/at]   (harmonic ref [meV/at])")
for T, fa, fh in zip(Temperatures, F_abs, F_harm):
    print(f"\t\t  {T:7.1f}   {1000*fa:12.2f}     ({1000*fh:11.2f})")

out = f"output/helmholtz_W_liquid.dat"
np.savetxt(out, np.column_stack([Temperatures, F_abs, F_harm]),
           header="Absolute Helmholtz free energy of liquid W (Route 1, KineticHessian)\n"
                  "T[K]  F_absolute[eV/atom]  F_harmonic_reference[eV/atom]")
print(f"\n\t\tWrote {out}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(Temperatures, 1000 * F_abs, "-o", ms=3, label="liquid (KineticHessian)")
    ax.plot(Temperatures, 1000 * F_harm, "--", color="gray", label="harmonic reference")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"$F$ [meV/atom]")
    ax.set_title(f"Absolute Helmholtz free energy of liquid W (V/atom={liq.volume_per_atom:.2f} $\\AA^3$)")
    ax.legend()
    fig.tight_layout()
    png = "output/helmholtz_W_liquid.png"
    fig.savefig(png)
    print(f"\t\tWrote {png}\n")
except ImportError:
    print("\t\t(matplotlib not available -- skipped plot)\n")
