#!/usr/bin/env python
"""
DescriptorDOS sampling for bcc W on the *Kinetic* (NVE) isosurface.

Unlike the Hessian isosurface (01_bcc), the Kinetic displacer samples the actual
SNAP potential with short NVE bursts targeted at each isosurface energy. Its
reference entropy S(alpha) is therefore NOT analytic: the analysis step must fit
it from an external harmonic free energy (see analyse_kinetic.py, fit_S_alpha).

To keep this example self-contained, this script samples TWO data sets for bcc W
over the SAME alpha grid:

  * a fast **Hessian** pass   -> output/DDOSData-W-bcc-Hessian.pkl
        used as the analytic harmonic *reference* F(beta) for the kinetic fit,
        and as an independent anharmonic cross-check.
  * the **Kinetic** pass      -> output/DDOSData-W-bcc-Kinetic.pkl
        the NVE sampling whose free energy we actually want.

Run
---
    python run_kinetic.py                       # both passes, serial
    mpirun -np 8 python run_kinetic.py --calls 800
    python run_kinetic.py --skip-hessian        # kinetic only (reuse existing ref)

NB: Kinetic sampling runs short MD and is ~50x slower per call than Hessian.
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


ELEMENT, STRUCTURE, MASS, MODEL = "W", "bcc", 183.84, "W"

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--calls", type=int, default=200,
                    help="TOTAL Kinetic (NVE) force calls per alpha (split over ranks)")
parser.add_argument("--href-calls", type=int, default=200,
                    help="TOTAL Hessian force calls per alpha for the reference pass")
parser.add_argument("--nalpha", type=int, default=12)
parser.add_argument("--amin", type=float, default=0.3, help="min isosurface (~T/1000 K)")
parser.add_argument("--amax", type=float, default=2.5, help="max isosurface (~T/1000 K)")
parser.add_argument("--skip-hessian", action="store_true",
                    help="do not (re)generate the Hessian reference pass")
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

iso_list = list(np.linspace(args.amin, args.amax, args.nalpha))
extra_params = yaml.safe_load(open(f"models/{MODEL}/params.yaml"))
input_configuration = f"structures/{ELEMENT}-{STRUCTURE}.data"
assert os.path.exists(input_configuration)


def sample(isosurface, calls):
    if rank == 0:
        print(f"\n\t\t### {isosurface} pass: {args.nalpha} alpha in "
              f"[{args.amin},{args.amax}], {calls} calls/alpha ###\n")
    mgr = DDOSManager(
        world=world,
        worker=LAMMPSWorker,
        isosurface=isosurface,
        element=ELEMENT,
        mass=MASS,
        default_calls=calls,
        lammps_input_configuration=input_configuration,
        dump_path="output",
        dump_file=f"DDOSData-{ELEMENT}-{STRUCTURE}-{isosurface}.pkl",
        hessian_path="output",
        hessian_data=f"hessian-{ELEMENT}-{STRUCTURE}.pkl",
        input_parameters={"IsoSurface": iso_list, "Strain": [0.0]},
        reconstruction_error=1.0e-7,
        append_data=args.append,
        verbose=args.verbose,
        **extra_params,
    )
    mgr.run()
    mgr.close()


# 1) harmonic reference + cross-check (fast)
if not args.skip_hessian:
    sample("Hessian", args.href_calls)

# 2) kinetic sampling (the target)
sample("Kinetic", args.calls)

if world is not None:
    world.Barrier()
    MPI.Finalize()
