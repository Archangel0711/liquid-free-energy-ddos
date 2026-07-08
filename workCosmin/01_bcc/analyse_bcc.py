#!/usr/bin/env python
"""
Free energy of bcc Tungsten from DescriptorDOS Hessian sampling -- NEW-style.

Reads the DDOS pickle produced by run_bcc.py and turns it into the free energy
per atom F(T) of the SNAP W potential, using its own linear coefficients
(X_W.snapcoeff) as the model parameters `Thetas`.

Pipeline (see analysis/Inverse-NVT-BCC-FCC.ipynb for the reference version):

    Thetas  = coefficients of X_W.snapcoeff        (the potential to evaluate)
    ddos    = DDOSAnalyzer(<pkl>, mass=..)          (the sampled descriptor DOS)
    E0      = ddos.cohesive_energy(Thetas)          (static T=0 energy / atom)
    ddos.match_score(...)                           (fit the score model S(D|alpha))
    F_vib   = ddos.score_free_energy(T, Thetas)['F'](anharmonic vibrational F / atom)
    F_total = E0 + F_vib

Run
---
    python analyse_bcc.py
    python analyse_bcc.py --isosurface Hessian --Tmin 100 --Tmax 3000 --nT 30
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

ELEMENT = "W"
STRUCTURE = "bcc"
MASS = 183.84  # amu

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--isosurface", default="Hessian",
                    help="which sampled data set to analyse (matches run_bcc.py)")
parser.add_argument("--Tmin", type=float, default=100.0)
parser.add_argument("--Tmax", type=float, default=4000.0)
parser.add_argument("--nT", type=int, default=30)
parser.add_argument("--match-order", type=int, default=7)
parser.add_argument("--match-type", default="chebyshev")
parser.add_argument("--volume-per-atom", type=float, default=None,
                    help="select this volume (default: the single sampled volume)")
args = parser.parse_args()

pkl_file = f"output/DDOSData-{ELEMENT}-{STRUCTURE}-{args.isosurface}.pkl"
assert os.path.exists(pkl_file), (
    f"{pkl_file} not found -- run run_bcc.py --isosurface {args.isosurface} first!")

# ---- potential coefficients (the model whose free energy we evaluate) ----
# skiprows=6 drops the 3 comment lines, the "1 <ncoeff>" line, the element line
# and the constant beta_0 -> leaves the bispectrum coefficients matching the
# descriptor produced by `compute sna/atom ... 8 ...` (55 for twojmax=8).
snapcoeff = f"models/{ELEMENT}/X_{ELEMENT}.snapcoeff"
Thetas = np.loadtxt(snapcoeff, skiprows=6).reshape((1, -1))

# ---- analyser ----
ddos = DDOSAnalyzer(pkl_file, volume_per_atom=args.volume_per_atom,
                    verbose=True, mass=MASS)

assert Thetas.shape[1] == ddos.D_0.size, (
    f"Theta dim {Thetas.shape[1]} != descriptor dim {ddos.D_0.size}; "
    f"check skiprows for {snapcoeff}")

# Kinetic samplings sample the *potential itself*: their reference entropy
# S(alpha) is not analytic and must be fit from an external harmonic free
# energy first. Hessian samplings carry an analytic harmonic reference and
# need no fit -- this is the self-contained path.
if ddos.reference_free_energy.requires_fit:
    sys.exit(
        "This data set uses a reference that needs an external S(alpha) fit "
        "(e.g. Kinetic sampling). Use --isosurface Hessian for the self-contained "
        "free energy, or supply reference F via ddos.fit_S_alpha(Beta, F).")

E0 = ddos.cohesive_energy(Thetas=Thetas)[0]  # eV/atom, static
print(f"\n\t\tbcc W  V/atom = {ddos.volume_per_atom:.4f} A^3,  N = {ddos.N}")
print(f"\t\tStatic cohesive energy E0 = {E0:.5f} eV/atom "
      f"({1000*E0:.2f} meV/atom)")

# fit the score model, then evaluate F over a temperature grid
n_alpha = len(ddos.alphas)
alpha_order = min(5, max(1, n_alpha - 1))
ddos.match_score(poly_order=args.match_order, poly_type=args.match_type)

Temperatures = np.linspace(args.Tmin, args.Tmax, args.nT)
F_vib = np.array([
    ddos.score_free_energy(Temperature=T, Thetas=Thetas,
                           alpha_poly_order=alpha_order)["F"][0]
    for T in Temperatures
])
F_total = E0 + F_vib

# analytic (quasi-)harmonic reference free energy, for comparison
try:
    F_harm = E0 + ddos.reference_free_energy.free_energy(Temperatures)
except Exception:
    F_harm = np.full_like(Temperatures, np.nan)

# ---- report ----
print("\n\t\t   T [K]   F_vib [meV/at]   F_total [meV/at]   F_harmonic [meV/at]")
for T, fv, ft, fh in zip(Temperatures, F_vib, F_total, F_harm):
    print(f"\t\t  {T:7.1f}   {1000*fv:12.3f}   {1000*ft:14.3f}   {1000*fh:14.3f}")

# ---- save table ----
out_dat = f"output/free_energy_{ELEMENT}_{STRUCTURE}.dat"
header = ("Free energy of bcc W (SNAP), DDOS %s sampling\n"
          "T[K]  F_vib[eV/atom]  F_total[eV/atom]  F_harmonic[eV/atom]"
          % args.isosurface)
np.savetxt(out_dat, np.column_stack([Temperatures, F_vib, F_total, F_harm]),
           header=header)
print(f"\n\t\tWrote {out_dat}")

# ---- plot ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(Temperatures, 1000 * F_total, "-o", ms=3, label="DDOS (anharmonic)")
    ax.plot(Temperatures, 1000 * F_harm, "--", color="gray",
            label="harmonic reference")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"$F$ [meV/atom]")
    ax.set_title(f"bcc W free energy (SNAP), DDOS {args.isosurface}")
    ax.legend()
    fig.tight_layout()
    out_png = f"output/free_energy_{ELEMENT}_{STRUCTURE}.png"
    fig.savefig(out_png)
    print(f"\t\tWrote {out_png}\n")
except ImportError:
    print("\t\t(matplotlib not available -- skipped plot)\n")
