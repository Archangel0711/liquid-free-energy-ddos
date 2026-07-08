#!/usr/bin/env python
"""
bcc <-> a15 free energy difference of W from multi-strain DescriptorDOS data.

For each phase and each sampled volume (strain) we build a DDOSAnalyzer, score
the free energy of the SNAP potential and add the static cohesive energy:

    F_total(V, T) = E0(V) + F_vib(V, T)          [eV/atom]

The P = 0 equilibrium free energy at temperature T is the minimum over volume,

    F_phase(T) = min_V F_total(V, T)             (quadratic fit in V)

and the phase difference is

    dF(T) = F_a15(T) - F_bcc(T).

dF > 0 means bcc is the stable phase at that temperature; a sign change locates
a (free-energy) transition.

Run
---
    python analyse_a15.py
    python analyse_a15.py --Tmin 100 --Tmax 3000 --nT 30
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

ELEMENT, MASS = "W", 183.84
PHASES = ("bcc", "a15")

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--Tmin", type=float, default=300.0)
parser.add_argument("--Tmax", type=float, default=3000.0)
parser.add_argument("--nT", type=int, default=28)
parser.add_argument("--match-order", type=int, default=7)
parser.add_argument("--match-type", default="chebyshev")
args = parser.parse_args()

theta = np.loadtxt(f"models/{ELEMENT}/X_{ELEMENT}.snapcoeff", skiprows=6).reshape(1, -1)
Temperatures = np.linspace(args.Tmin, args.Tmax, args.nT)


def phase_free_energy(structure):
    """Return (F_equilibrium(T), V_equilibrium(T)) minimised over sampled volumes."""
    pkl = f"output/DDOSData-{ELEMENT}-{structure}-Hessian.pkl"
    assert os.path.exists(pkl), f"{pkl} not found -- run run_a15.py first!"

    volumes = DDOSAnalyzer(pkl, mass=MASS, testing=True).volumes_per_atom
    print(f"\n\t\t{ELEMENT} {structure}: {len(volumes)} volumes/atom = "
          f"{np.round(volumes, 3)}")

    # F_total(V, T) for every sampled volume
    F_of_V = np.zeros((volumes.size, Temperatures.size))
    for iv, V in enumerate(volumes):
        A = DDOSAnalyzer(pkl, volume_per_atom=V, mass=MASS, verbose=False)
        E0 = A.cohesive_energy(Thetas=theta)[0]
        A.match_score(poly_order=args.match_order, poly_type=args.match_type)
        n_a = len(A.alphas)
        order = min(5, max(1, n_a - 1))
        F_of_V[iv] = E0 + np.array([
            A.score_free_energy(T, Thetas=theta, alpha_poly_order=order)["F"][0]
            for T in Temperatures])

    # minimise over volume at each T (quadratic fit -> vertex, clamped to range)
    F_eq = np.empty(Temperatures.size)
    V_eq = np.empty(Temperatures.size)
    for it in range(Temperatures.size):
        if volumes.size >= 3:
            c2, c1, c0 = np.polyfit(volumes, F_of_V[:, it], 2)
            if c2 > 0:
                Vstar = np.clip(-c1 / (2 * c2), volumes.min(), volumes.max())
                F_eq[it] = np.polyval([c2, c1, c0], Vstar)
                V_eq[it] = Vstar
                continue
        iv = F_of_V[:, it].argmin()
        F_eq[it], V_eq[it] = F_of_V[iv, it], volumes[iv]
    return F_eq, V_eq


F, Veq = {}, {}
for ph in PHASES:
    F[ph], Veq[ph] = phase_free_energy(ph)

dF = F["a15"] - F["bcc"]  # >0 => bcc stable

# locate sign change (transition) if present
T_c = None
sign = np.sign(dF)
idx = np.where(np.diff(sign) != 0)[0]
if idx.size:
    i = idx[0]
    T_c = Temperatures[i] - dF[i] * (Temperatures[i + 1] - Temperatures[i]) / (
        dF[i + 1] - dF[i])

print("\n\t\t   T [K]   V_bcc   V_a15    F_bcc      F_a15     dF=a15-bcc  stable")
for k in range(Temperatures.size):
    stable = "bcc" if dF[k] > 0 else "a15"
    print(f"\t\t  {Temperatures[k]:7.1f}  {Veq['bcc'][k]:5.2f}  {Veq['a15'][k]:5.2f}  "
          f"{1000*F['bcc'][k]:9.2f}  {1000*F['a15'][k]:9.2f}  "
          f"{1000*dF[k]:9.2f}   {stable}")
if T_c is not None:
    print(f"\n\t\t>>> bcc<->a15 free-energy crossing near T = {T_c:.0f} K")
else:
    stable = "bcc" if dF.mean() > 0 else "a15"
    print(f"\n\t\t>>> no crossing in [{args.Tmin:.0f},{args.Tmax:.0f}] K; "
          f"{stable} stable throughout (test potential)")

# save + plot
out = f"output/phase_difference_{ELEMENT}_bcc_a15.dat"
np.savetxt(out, np.column_stack([Temperatures, F["bcc"], F["a15"], dF]),
           header="T[K]  F_bcc[eV/at]  F_a15[eV/at]  dF=a15-bcc[eV/at]")
print(f"\n\t\tWrote {out}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    ax1.plot(Temperatures, 1000 * F["bcc"], "-o", ms=3, label="bcc")
    ax1.plot(Temperatures, 1000 * F["a15"], "-s", ms=3, label="a15")
    ax1.set_xlabel("Temperature [K]"); ax1.set_ylabel(r"$F$ [meV/atom]")
    ax1.set_title("W free energy (P=0, quasi-harmonic)"); ax1.legend()

    ax2.axhline(0, c="k", lw=0.8)
    ax2.plot(Temperatures, 1000 * dF, "-o", ms=3, color="tab:red")
    if T_c is not None:
        ax2.axvline(T_c, ls="--", c="gray", label=f"$T_c$≈{T_c:.0f} K")
        ax2.legend()
    ax2.set_xlabel("Temperature [K]")
    ax2.set_ylabel(r"$\Delta F = F_{a15}-F_{bcc}$ [meV/atom]")
    ax2.set_title("bcc <-> a15 phase difference")
    fig.tight_layout()
    png = f"output/phase_difference_{ELEMENT}_bcc_a15.png"
    fig.savefig(png)
    print(f"\t\tWrote {png}\n")
except ImportError:
    print("\t\t(matplotlib not available -- skipped plot)\n")
