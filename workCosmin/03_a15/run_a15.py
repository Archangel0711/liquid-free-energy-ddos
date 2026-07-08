#!/usr/bin/env python
"""
Multi-strain DescriptorDOS sampling for the bcc <-> a15 phase competition in W.

Samples BOTH phases (bcc, 128 atoms; a15, 64 atoms) on the Hessian isosurface
over a grid of

  * isosurface values  alpha   (~ T / 1000 K), and
  * volumetric strains         (each strain = a different volume).

Sampling several strains lets the analysis minimise the free energy over volume
at every temperature (the P = 0 equilibrium), so each phase is compared at its
own thermal-equilibrium volume -- essential because bcc and a15 have different
equilibrium volumes (V/atom ~ 16.16 vs 16.45 A^3).

A separate Hessian is built and cached per (structure, strain); this is
automatic on the first pass.

Outputs (one pkl per phase, records span strain x alpha):
    output/DDOSData-W-bcc-Hessian.pkl
    output/DDOSData-W-a15-Hessian.pkl

Run
---
    python run_a15.py                              # both phases, serial
    mpirun -np 8 python run_a15.py --calls 2400
    python run_a15.py --structures a15             # just one phase
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
from DescriptorDOS import DDOSManager
from DescriptorDOS.workers.LAMMPSWorker import LAMMPSWorker


def is_running_with_mpi():
    for v in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMI_RANK",
              "I_MPI_JOB_SIZE", "SLURM_NTASKS"):
        if v in os.environ and os.environ[v] not in ("", "1"):
            return True
    return False


ELEMENT, MASS, MODEL = "W", 183.84, "W"
# reference volumes/atom (for information only; strain is applied to the .data cell)
V_TARGET = {"bcc": 16.16228768, "a15": 16.45160002}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--structures", default="bcc,a15",
                    help="comma list of phases to sample")
parser.add_argument("--calls", type=int, default=200,
                    help="TOTAL force calls per (strain, alpha) point")
parser.add_argument("--nalpha", type=int, default=10)
parser.add_argument("--amin", type=float, default=0.5)
parser.add_argument("--amax", type=float, default=4.0)
parser.add_argument("--strains", default="-0.02,-0.01,0.0,0.01,0.02",
                    help="comma list of volumetric (linear) strains")
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

structures = [s.strip() for s in args.structures.split(",") if s.strip()]
strain_list = sorted(float(s) for s in args.strains.split(","))
iso_list = list(np.linspace(args.amin, args.amax, args.nalpha))
extra_params = yaml.safe_load(open(f"models/{MODEL}/params.yaml"))

for structure in structures:
    input_configuration = f"structures/{ELEMENT}-{structure}.data"
    assert os.path.exists(input_configuration), f"{input_configuration} not found!"
    if rank == 0:
        print(f"\n\t\t################ {ELEMENT} {structure} "
              f"(V_ref/atom={V_TARGET[structure]}) ################")
        print(f"\t\t{len(strain_list)} strains x {args.nalpha} alpha, "
              f"{args.calls} calls each\n")
    mgr = DDOSManager(
        world=world,
        worker=LAMMPSWorker,
        isosurface="Hessian",
        element=ELEMENT,
        mass=MASS,
        default_calls=args.calls,
        lammps_input_configuration=input_configuration,
        dump_path="output",
        dump_file=f"DDOSData-{ELEMENT}-{structure}-Hessian.pkl",
        hessian_path="output",
        hessian_data=f"hessian-{ELEMENT}-{structure}.pkl",
        input_parameters={"IsoSurface": iso_list, "Strain": strain_list},
        reconstruction_error=1.0e-7,
        append_data=args.append,
        verbose=args.verbose,
        **extra_params,
    )
    mgr.run()
    mgr.close()

if world is not None:
    world.Barrier()
    MPI.Finalize()
