# 02_liquid — Free energy of LIQUID Tungsten (Kinetic / NVE DescriptorDOS)

Free energy of **liquid W** from DescriptorDOS, and its difference to the bcc
solid at the same density. A liquid has no Hessian / harmonic reference, so it is
sampled on the **Kinetic (NVE)** isosurface — the momentum-dependent formalism
that (per the DDOS paper) "only requires the ability to perform NVE dynamics with
the reference potential".

## Two-step workflow (what to run)

```bash
# interpreter with LAMMPS (ML-SNAP+PHONON) + analysis stack:
VENV=/home/marinica/GitHub/old_DescriptorDOS-Development.git/.venv/bin/python
cd workCosmin/02_liquid

# 1) sample the liquid on the Kinetic isosurface (NVE bursts of the SNAP potential)
$VENV run_liquid.py                         # defaults: Kinetic, 20 alpha in [0.5,6.0], 400 calls/alpha
#    parallel:  mpirun -np 8 $VENV run_liquid.py --calls 4800

# 2) free energy F(T) of the liquid + liquid-vs-solid difference
$VENV analyse_liquid.py                      # builds the bcc reference the first time, then analyses
```

That's it — the two commands the workflow needs. `analyse_liquid.py` builds the
reference it needs on the first call and caches it in `output/`.

## Why step 2 needs a reference (and which one)

The Kinetic displacer samples the *actual* potential, so its reference entropy
`S(alpha)` is **not analytic**: `DDOSAnalyzer.fit_S_alpha(Beta, F_ref)` must be
given a reference free energy `F_ref(beta)` first. (A stock `01_bcc`-style
analysis script simply `sys.exit`s at this point — that is the trap this example
resolves.)

**Reference used here:** a **bcc-W harmonic free energy at the liquid volume**.
The liquid snapshot has V/atom ≈ 17.06 Å³; `analyse_liquid.py` strains the bcc
cell by the matching amount (≈ +1.8 %) to that same volume, builds its Hessian,
and uses its harmonic free energy `F_bcc_harm(beta)` (vibrational, → 0 as T → 0)
as the reference. Anchoring the liquid to the bcc solid at the **same density**
makes the output directly the liquid−solid free-energy difference

```
dF(T) = F_liquid(T) − F_bcc(T)          (per atom)
```

whose sign change is a (constant-volume) melting-like temperature.

### Pipeline inside analyse_liquid.py
```
F_bcc_harm(beta)  <-  bcc Hessian strained to the liquid volume     (reference)
liquid.match_score(...)
liquid.fit_S_alpha(Beta, F_bcc_harm)                                (required, Kinetic)
Thetas   = [ theta_SNAP(55) , 1.0 ]        # 56-dim kinetic descriptor (last col = K)
F_liquid(T) = E0_liquid + liquid.score_free_energy(T, Thetas)['F']
```

Useful flags: `--Tmin/--Tmax/--nT` (temperature grid), `--ref-calls` (bcc
reference sampling), `--regenerate-ref` (rebuild the reference),
`--theta-kinetic {1.0|0.0}` (include the kinetic column → total energy, or drop
it → potential only).

## Outputs

- `output/DDOSData-W-liquid-Kinetic.pkl` — liquid NVE descriptor DOS
- `output/DDOSData-W-bccref-Hessian.pkl`, `output/hessian-W-bccref.pkl` — cached bcc reference at the liquid volume
- `output/free_energy_W_liquid.dat` — `T, F_liquid, F_bcc, dF` (eV/atom)
- `output/free_energy_W_liquid.png` — F_liquid vs F_bcc, and dF(T) with the crossing

With the bundled test potential the crossing lands around ~2700 K (bcc lower
below, liquid lower above) — the qualitative shape of melting.

## Files

```
02_liquid/
├── models/W/            X_W.snapcoeff, D.snapparam, in.lammps, params.yaml
├── structures/
│   ├── W-liquid.data    128-atom liquid snapshot (V/atom ≈ 17.06 Å³)
│   └── W-bcc.data       bcc cell, strained to the liquid volume for the reference
├── run_liquid.py        Kinetic (NVE) sampling of the liquid
├── analyse_liquid.py    reference + fit_S_alpha + free energy + dF
└── output/
```

## Caveats (read before trusting numbers)

- **Test potential.** `X_W.snapcoeff` is `# Test values for W`; absolute energies
  and the crossing temperature are *not* physical. Swap in a production W SNAP
  (same `twojmax=8`) for real numbers.
- **Kinetic absolute F is delicate.** As documented in `02_kinetic`, the absolute
  Kinetic free energy is sensitive to (i) the reference-entropy normalization and
  (ii) NVE convergence, and can carry a smooth ~1–2 k_B/atom offset with modest
  sampling. Use many more NVE calls (`--calls`) for production, and treat `dF` as
  more robust than either absolute F.
- **Single-snapshot E0.** `E0_liquid` is the static energy of the one liquid
  snapshot in `structures/W-liquid.data`, not an ensemble average, so the liquid
  absolute F depends on that configuration. Average over several liquid snapshots
  (rerun with different `structures/W-liquid*.data`, `--append`) to reduce this.
- **Reference stability.** bcc under +1.8 % tension should stay stable (3 zero
  modes); check the `Zero Modes = k/3N` line printed while the reference builds.
- **Constant volume.** Both phases are compared at the fixed liquid volume, so
  `dF` is an NVT / constant-V difference, not the full P=0 melting free energy.
