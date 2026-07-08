# 02_kinetic — bcc W free energy from the Kinetic (NVE) isosurface

Companion to `01_bcc`. Here the descriptor density-of-states is sampled on the
**Kinetic** isosurface: instead of drawing analytic harmonic displacements, the
`KineticDisplacer` runs short **NVE** bursts of the real SNAP potential targeted
at each isosurface energy. This is the sampling used when no Hessian is wanted
(or the reference is strongly anharmonic).

## The extra step: `fit_S_alpha`

The Hessian family carries an *analytic* reference entropy `S(alpha)`. The
Kinetic displacer does not — it samples the potential itself, so its reference
free energy must be supplied and fitted before any free energy can be scored:

```python
kin.match_score(...)
kin.fit_S_alpha(Beta, F_reference)      # <-- required for kinetic
kin.score_free_energy(T, Thetas=...)
```

`F_reference(Beta)` is the **harmonic free energy** of the same bcc W, taken from
the Hessian reference that `run_kinetic.py` produces alongside
(`hess.reference_free_energy.free_energy(T)`; per atom, vibrational, → 0 as
T → 0). `Beta = 1/(k_B T)`.

## Descriptor / Thetas layout (LAMMPS SNAP kinetic)

The analyzer stacks the **55** SNAP bispectrum components with **1** appended
kinetic column → 56. The model parameters are therefore

```python
Thetas = [ theta_SNAP(55) , 1.0 ]        # trailing 1.0 = kinetic column
```

(cf. `testing/integration/test_kinetic.py`). The static cohesive energy uses only
the 55 SNAP coefficients (via the Hessian analyzer, whose `D_0` is 55-dim).

## Run

```bash
VENV=/home/marinica/GitHub/old_DescriptorDOS-Development.git/.venv/bin/python
cd workCosmin/02_kinetic

$VENV run_kinetic.py          # Hessian reference pass (fast) + Kinetic pass (slow NVE)
$VENV analyse_kinetic.py      # fit_S_alpha, score F(T), cross-check vs Hessian
```

Kinetic sampling runs real MD and is ~50× slower per call than Hessian; use
`mpirun -np N` to parallelise (total calls per alpha are split across ranks).

## Outputs

- `output/DDOSData-W-bcc-Kinetic.pkl` — NVE descriptor DOS
- `output/DDOSData-W-bcc-Hessian.pkl` — harmonic reference + cross-check
- `output/free_energy_W_bcc_kinetic.{dat,png}` — F(T): harmonic vs Hessian vs Kinetic

## What to look for (and an honest caveat)

The **Kinetic** and **Hessian** free energies come from independent samplings of
the *same* potential, so in principle their anharmonic free energies coincide —
and the Hessian anharmonic correction here is only −6…−40 meV/atom (tiny, as it
should be for this near-harmonic test potential).

In practice, with the modest sampling in this example, the **absolute** Kinetic
free energy shows a smooth, temperature-growing offset from the Hessian result
(order ~1–2 k_B/atom in slope; `analyse_kinetic.py` prints it). This offset is
*not* physical anharmonicity: the absolute Kinetic free energy is sensitive to
(i) the normalization of the reference-entropy fit fed to `fit_S_alpha`, and
(ii) the convergence of the short NVE bursts and their rank-compressed
descriptor histograms. It shrinks with far more NVE calls and a carefully matched
reference.

**Takeaway:** use the **Hessian** method (`01_bcc`) for well-conditioned
*absolute* free energies. This example exists to exercise the full Kinetic
pipeline (sample → `match_score` → `fit_S_alpha` → `score_free_energy`) end-to-end
on a real LAMMPS/SNAP system. The Kinetic sampler is at its best for free-energy
*differences* relative to a reference potential whose absolute F is supplied.
`--theta-kinetic {1.0|0.0}` toggles whether the appended kinetic column is
included (total energy) or dropped (potential energy).
