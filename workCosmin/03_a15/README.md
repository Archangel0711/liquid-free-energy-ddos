# 03_a15 — bcc ↔ a15 phase competition in W (multi-strain)

Free-energy *difference* between the **bcc** and **a15** phases of W, from
multi-strain DescriptorDOS Hessian sampling. This is the phase-stability
extension of `01_bcc`.

## Idea

Both phases are sampled on the Hessian isosurface over a grid of

- isosurface values `alpha` (~ T/1000 K), **and**
- volumetric strains (→ several volumes per phase).

Sampling multiple strains is what makes this a *phase* calculation: bcc and a15
have different equilibrium volumes (V/atom ≈ 16.16 vs 16.45 Å³), and both expand
with temperature, so each phase must be compared at its **own** P = 0 equilibrium
volume. For every temperature the analysis minimises

```
F_total(V, T) = E0(V) + F_vib(V, T)     over V   (quadratic fit → vertex)
```

giving `F_bcc(T)`, `F_a15(T)`, and the difference

```
dF(T) = F_a15(T) − F_bcc(T)      (>0 ⇒ bcc stable; sign change ⇒ transition)
```

## Run

```bash
VENV=/home/marinica/GitHub/old_DescriptorDOS-Development.git/.venv/bin/python
cd workCosmin/03_a15

# NOTE the '=' form: leading-'-' strain values must use --strains=... , not --strains ...
$VENV run_a15.py --strains="-0.02,-0.01,0.0,0.01,0.02"
$VENV analyse_a15.py
```

`run_a15.py` loops over both phases and over the strain grid; a separate Hessian
is built and cached per (structure, strain). Use `mpirun -np N` to parallelise.

## Outputs

- `output/DDOSData-W-bcc-Hessian.pkl`, `output/DDOSData-W-a15-Hessian.pkl`
  (records span strain × alpha)
- `output/hessian-W-{bcc,a15}.pkl` — cached Hessian modes, keyed by strain
- `output/phase_difference_W_bcc_a15.{dat,png}` — F_bcc, F_a15 and dF(T); the
  right panel marks any bcc↔a15 crossing temperature.

## Caveats

- The bundled `X_W.snapcoeff` is a **test** potential, so the numbers (and any
  crossing temperature) are *not* physical — the point is the workflow. Swap in a
  production W SNAP potential (same `twojmax`) for physical phase stability.
- a15 is a Frank–Kasper phase; check the `Zero Modes = k/3N` lines in the run log
  — anything above the 3 rigid translations flags soft/unstable modes at that
  strain (those modes are dropped, making the harmonic reference approximate).
- Widen `--strains` if the quadratic F(V) minimum lands on the edge of the grid
  at high T (thermal expansion pushes V_eq up).
