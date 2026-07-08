#!/usr/bin/env python
"""
DescriptorDOS sampling for bcc Tungsten -- NEW-style workflow.

This is the modern re-write of examples/run_bcc_test.py from the old repo.
It samples the descriptor density-of-states (DDOS) of bcc W on the *Hessian*
isosurface over a grid of isosurface values ("alpha", ~ temperature/1000 K).

What happens on the first run
-----------------------------
1. The HessianDisplacer builds the Hessian (via LAMMPS `dynamical_matrix`),
   diagonalises it and caches the modes in  <hessian_path>/<hessian_data>.
   -> this is the "perform some Hessian for W" step; it is automatic.
2. For every requested isosurface value it draws `default_calls` samples on the
   harmonic hypersphere and score-matches the descriptor DOS, saving the
   compressed histograms in  <dump_path>/<dump_file>.

Later runs re-use the cached Hessian and (with --append) add statistics.

Run
---
    # serial (default)
    python run_bcc.py

    # parallel: total calls per alpha are split across ranks
    mpirun -np 8 python run_bcc.py --calls 2000

    # kinetic isosurface instead of Hessian (needs an external S(alpha) fit
    # in the analysis step -- see analyse_bcc.py)
    python run_bcc.py --isosurface Kinetic
"""

import os
import sys
import argparse
import numpy as np

# --- locate the DescriptorDOS package (this dir lives at <repo>/workCosmin/01_bcc) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
# run everything relative to this directory so the model paths resolve
os.chdir(SCRIPT_DIR)

import yaml
from DescriptorDOS import DDOSManager
from DescriptorDOS.workers.LAMMPSWorker import LAMMPSWorker


def is_running_with_mpi():
    """True if launched under mpirun/srun (so we should use MPI)."""
    for v in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMI_RANK",
        "I_MPI_JOB_SIZE",
        "SLURM_NTASKS",
    ):
        if v in os.environ and os.environ[v] not in ("", "1"):
            return True
    return False


# ----------------------------------------------------------------------------
#  Physical / numerical configuration for bcc W
# ----------------------------------------------------------------------------
ELEMENT = "W"
STRUCTURE = "bcc"
MASS = 183.84  # amu
MODEL = "W"  # -> models/W/params.yaml
# reference volume per atom for bcc W (see old run_bcc_test.py: V_target/atom)
V_TARGET_PER_ATOM = 15.52976221327011

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--isosurface", default="Hessian",
                    choices=["Hessian", "Kinetic", "DynamicalHessian", "KineticHessian"],
                    help="isosurface / reference used for sampling")
parser.add_argument("--calls", type=int, default=400,
                    help="TOTAL force calls per isosurface value (split over MPI ranks)")
parser.add_argument("--nalpha", type=int, default=15,
                    help="number of isosurface (alpha) grid points")
parser.add_argument("--amin", type=float, default=0.5, help="min isosurface value")
parser.add_argument("--amax", type=float, default=4.0, help="max isosurface value")
parser.add_argument("--append", action="store_true",
                    help="append to / combine with existing data")
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

# ----------------------------------------------------------------------------
#  MPI (optional).  Serial by default -- no mpi4py needed.
# ----------------------------------------------------------------------------
if is_running_with_mpi():
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    rank = world.Get_rank()
else:
    world = None
    rank = 0

# isosurface (alpha) grid -- the "sample at given alpha" of the old script.
# bcc W needs a wider range than a15 (old script used linspace(0.5, 4.0, 35)).
iso_list = list(np.linspace(args.amin, args.amax, args.nalpha))
strain_list = [0.0]  # sample at the reference volume only

# model configuration (lammps_input_script + pair_coeff), paths relative to here
extra_params = yaml.safe_load(open(f"models/{MODEL}/params.yaml"))

input_configuration = f"structures/{ELEMENT}-{STRUCTURE}2.data"
assert os.path.exists(input_configuration), f"{input_configuration} not found!"

dump_path = "output"
dump_file = f"DDOSData-{ELEMENT}-{STRUCTURE}-{args.isosurface}.pkl"
hessian_data = f"hessian-{ELEMENT}-{STRUCTURE}.pkl"

if rank == 0:
    print(f"\n\t\tbcc W DDOS: isosurface={args.isosurface}, "
          f"{args.nalpha} alpha in [{args.amin},{args.amax}], "
          f"{args.calls} calls/alpha\n")

DDOS_Manager = DDOSManager(
    world=world,
    worker=LAMMPSWorker,
    isosurface=args.isosurface,
    element=ELEMENT,
    mass=MASS,
    default_calls=args.calls,
    lammps_input_configuration=input_configuration,
    dump_path=dump_path,
    dump_file=dump_file,
    hessian_path=dump_path,
    hessian_data=hessian_data,
    input_parameters={"IsoSurface": iso_list, "Strain": strain_list},
    reconstruction_error=1.0e-7,
    append_data=args.append,
    verbose=args.verbose,
    **extra_params,
)
DDOS_Manager.run()
DDOS_Manager.close()

if world is not None:
    world.Barrier()
    MPI.Finalize()
