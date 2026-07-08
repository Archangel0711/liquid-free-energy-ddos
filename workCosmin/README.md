# workCosmin — DescriptorDOS free-energy workflows for W

Worked examples porting the old-repo W free-energy scripts
(`old_DescriptorDOS-Development.git/examples/run_bcc_test.py`, `work/run.py`) to
the current DescriptorDOS API. Each subdirectory is a self-contained study.

| dir | isosurface | what it computes |
|-----|-----------|------------------|
| [`01_bcc/`](01_bcc/)     | Hessian          | free energy F(T) of bcc W (self-contained, analytic harmonic reference) |
| [`02_kinetic/`](02_kinetic/) | Kinetic (NVE)    | F(T) of bcc W by NVE sampling + `fit_S_alpha`; cross-checked against the Hessian result |
| [`03_a15/`](03_a15/)     | Hessian, multi-strain | bcc ↔ a15 phase difference ΔF(T) at each phase's P=0 equilibrium volume |

Read each subdir's `README.md` for details. Start with `01_bcc`.

## Environment (important)

Run everything with the interpreter that has LAMMPS (built with **ML-SNAP** and
**PHONON**) **and** the analysis stack — in this checkout that is the old repo's
venv, to which `scipy`, `scikit-learn` and `tqdm` were added:

```bash
VENV=/home/marinica/GitHub/old_DescriptorDOS-Development.git/.venv/bin/python
```

The miniforge `base` python cannot be used (its `lammps` fails to load,
`libpython3.12.so.1.0` missing). The scripts add the new-repo root to
`sys.path` themselves, so no install of the package is required.

## Common shape of every example

```
<example>/
├── models/W/        X_W.snapcoeff, D.snapparam, in.lammps, params.yaml
├── structures/      W-<phase>.data
├── run_*.py         sampling (DDOSManager, new API)
├── analyse_*.py     free energy via DDOSAnalyzer
└── output/          generated pkl / dat / png / logs
```

New-API conventions used throughout (vs the old scripts):
- model config in `models/W/params.yaml` (`lammps_input_script` + `pair_coeff`),
  merged into `DDOSManager(**extra_params)`;
- structure passed as a single `lammps_input_configuration=…W-<phase>.data`
  (no more `input_path`/`input_data`);
- Hessian modes auto-generated & cached on first run (no `generate_hessian.sh`);
- serial by default (`world=None`, no mpi4py needed); `mpirun -np N` splits the
  per-alpha calls across ranks;
- analysis via `DDOSAnalyzer.score_free_energy` with `Thetas` = the SNAP
  coefficients of `X_W.snapcoeff` (`skiprows=6` → 55 bispectrum components).

## Caveat on the bundled potential

`X_W.snapcoeff` is a **test** potential (`# Test values for W`): cohesive energy
≈ −4.25 eV/atom, not the physical −8.9. Numbers are for demonstrating the
pipeline; swap in a production W SNAP potential (same `twojmax=8`) for physics.
