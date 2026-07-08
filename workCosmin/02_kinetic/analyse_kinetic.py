#!/usr/bin/env python
"""
Free energy of bcc W from the *Kinetic* (NVE) DescriptorDOS sampling.

Kinetic samplings have no analytic reference entropy S(alpha): it is fitted from
an external harmonic free energy F(beta) via DDOSAnalyzer.fit_S_alpha, using the
Hessian reference produced alongside by run_kinetic.py. We then score the free
energy of the SNAP potential and cross-check it against the (independent) Hessian
anharmonic free energy.

Kinetic descriptor layout (LAMMPS SNAP): the analyzer stacks the 55 bispectrum
components with 1 appended (kinetic) column -> 56. The model parameters are
therefore Thetas = [ theta_SNAP(55), 1.0 ], the trailing 1.0 selecting the
kinetic column (cf. testing/integration/test_kinetic.py). The static cohesive
energy uses only the 55 SNAP coefficients.

Run
---
    python analyse_kinetic.py
"""

import os
import sys
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(SCRIPT_DIR)

from DescriptorDOS import DDOSAnalyzer
from DescriptorDOS.Constants import Boltzmann, eV

kB = Boltzmann / eV  # eV/K

ELEMENT, STRUCTURE, MASS = "W", "bcc", 183.84

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--Tmin", type=float, default=300.0)
parser.add_argument("--Tmax", type=float, default=2500.0)
parser.add_argument("--nT", type=int, default=12)
parser.add_argument("--match-order", type=int, default=4)
parser.add_argument("--match-type", default="monomial")
parser.add_argument("--theta-kinetic", type=float, default=1.0,
                    help="coefficient on the appended kinetic column of Thetas "
                         "(1.0 = include K -> total energy, per test_kinetic; "
                         "0.0 = potential energy only)")
args = parser.parse_args()

kin_pkl = f"output/DDOSData-{ELEMENT}-{STRUCTURE}-Kinetic.pkl"
hess_pkl = f"output/DDOSData-{ELEMENT}-{STRUCTURE}-Hessian.pkl"
assert os.path.exists(kin_pkl), f"{kin_pkl} not found -- run run_kinetic.py first!"
assert os.path.exists(hess_pkl), (
    f"{hess_pkl} not found -- run run_kinetic.py (Hessian reference pass) first!")

# SNAP coefficients (55), the potential theta
theta_snap = np.loadtxt(f"models/{ELEMENT}/X_{ELEMENT}.snapcoeff", skiprows=6)

# ---- Hessian analyzer: harmonic reference + anharmonic cross-check ----
hess = DDOSAnalyzer(hess_pkl, mass=MASS, verbose=False)
E0 = hess.cohesive_energy(Thetas=theta_snap.reshape(1, -1))[0]  # static, eV/atom

Temperatures = np.linspace(args.Tmin, args.Tmax, args.nT)

# harmonic reference free energy F(beta), per atom, -> 0 as T -> 0 (vibrational)
Beta = 1.0 / (kB * Temperatures)
F_harm = hess.reference_free_energy.free_energy(Temperatures)

# Hessian anharmonic free energy (independent method, for comparison)
hess.match_score(poly_order=7, poly_type="chebyshev")
n_a_h = len(hess.alphas)
F_hess = np.array([
    hess.score_free_energy(T, Thetas=theta_snap.reshape(1, -1),
                           alpha_poly_order=min(5, max(1, n_a_h - 1)))["F"][0]
    for T in Temperatures])

# ---- Kinetic analyzer: fit S(alpha) from the harmonic reference ----
kin = DDOSAnalyzer(kin_pkl, mass=MASS, verbose=True)
assert kin.flavor == "kinetic"
kin.match_score(poly_order=args.match_order, poly_type=args.match_type)
# Kinetic requires a reference S(alpha) before scoring free energies:
kin.fit_S_alpha(Beta, F_harm)

# Thetas for the stacked (56-dim) kinetic descriptor: SNAP coeffs + kinetic column
Thetas_kin = np.concatenate([theta_snap, [args.theta_kinetic]]).reshape(1, -1)
assert Thetas_kin.shape[1] == kin.Rotations[0].shape[0], (
    f"Theta dim {Thetas_kin.shape[1]} != {kin.Rotations[0].shape[0]}")

n_a_k = len(kin.alphas)
F_kin = np.array([
    kin.score_free_energy(T, Thetas=Thetas_kin,
                          alpha_poly_order=min(4, max(1, n_a_k - 1)))["F"][0]
    for T in Temperatures])

# ---- report (total = static E0 + vibrational F) ----
print(f"\n\t\tbcc W  V/atom = {kin.volume_per_atom:.4f} A^3, "
      f"static E0 = {1000*E0:.2f} meV/atom\n")
print("\t\t   T [K]    F_harmonic    F_Hessian     F_Kinetic   dF(K-H)  [meV/atom]")
for T, fh, fH, fk in zip(Temperatures, F_harm, F_hess, F_kin):
    print(f"\t\t  {T:7.1f}   {1000*(E0+fh):10.2f}   {1000*(E0+fH):10.2f}   "
          f"{1000*(E0+fk):10.2f}   {1000*(fk-fH):7.1f}")

# Honesty check: Hessian and Kinetic sample the SAME potential, so their
# anharmonic free energies should coincide. In practice the *absolute* Kinetic
# free energy is delicate -- it inherits (i) the normalization of the reference
# entropy fit and (ii) the convergence of the short NVE bursts and their
# rank-compressed descriptor histograms. With modest sampling a smooth,
# T-growing (entropy-like) offset from the Hessian result is expected; treat the
# Hessian method (01_bcc) as the well-conditioned absolute reference.
dS = np.polyfit(Temperatures, 1000 * (F_kin - F_hess), 1)[0] / (1000 * kB)  # in kB
print(f"\n\t\tNOTE: Kinetic-Hessian gap ~ linear in T; slope ~ {dS:.2f} k_B/atom.")
print("\t\t      The Hessian result is the well-conditioned absolute free energy;")
print("\t\t      the Kinetic run here demonstrates the full NVE pipeline")
print("\t\t      (sample -> match_score -> fit_S_alpha -> score_free_energy).")

# ---- save + plot ----
out = f"output/free_energy_{ELEMENT}_{STRUCTURE}_kinetic.dat"
np.savetxt(out,
           np.column_stack([Temperatures, E0 + F_harm, E0 + F_hess, E0 + F_kin]),
           header="T[K]  F_harmonic  F_Hessian  F_Kinetic   [eV/atom, total]")
print(f"\n\t\tWrote {out}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(Temperatures, 1000 * (E0 + F_harm), "--", c="gray", label="harmonic ref.")
    ax.plot(Temperatures, 1000 * (E0 + F_hess), "-o", ms=3, label="Hessian (anharm.)")
    ax.plot(Temperatures, 1000 * (E0 + F_kin), "-s", ms=3,
            label="Kinetic/NVE (pipeline demo)")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"$F$ [meV/atom]")
    ax.set_title("bcc W free energy: Kinetic (NVE) pipeline vs Hessian (SNAP)")
    ax.legend()
    fig.tight_layout()
    png = f"output/free_energy_{ELEMENT}_{STRUCTURE}_kinetic.png"
    fig.savefig(png)
    print(f"\t\tWrote {png}\n")
except ImportError:
    print("\t\t(matplotlib not available -- skipped plot)\n")
