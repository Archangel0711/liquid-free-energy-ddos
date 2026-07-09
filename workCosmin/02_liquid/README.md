# 01_bcc — Free energy of bcc Tungsten (DescriptorDOS, new-style)

Free-energy calculation for **bcc W** with the SNAP potential, ported from the
old-repo example `examples/run_bcc_test.py` to the current DescriptorDOS API
(`DescriptorDOS-Development.git`).

The physics is unchanged from the old workflow; only the driver API changed.

## What it does

1. **Hessian** — on the first `run_bcc.py` the `HessianDisplacer` builds the
   Hessian of bcc W with LAMMPS `dynamical_matrix`, diagonalises it, and caches
   the modes in `output/hessian-W-bcc.pkl`. This is the "perform a Hessian for W"
   step; it is automatic and reused on subsequent runs.
2. **Sample at given alpha** — for each isosurface value `alpha` (≈ T/1000 K)
   it draws samples on the harmonic hypersphere and score-matches the descriptor
   density-of-states (DDOS), saving compressed histograms in
   `output/DDOSData-W-bcc-Hessian.pkl`.
3. **Free energy** — `analyse_bcc.py` feeds that DDOS and the SNAP coefficients
   (`X_W.snapcoeff`) to `DDOSAnalyzer` and produces `F(T)` per atom.

## Layout

```
01_bcc/
├── models/W/
│   ├── X_W.snapcoeff     # SNAP linear coefficients (the potential θ)
│   ├── D.snapparam       # SNAP descriptor hyper-params (rcut 4.7, twojmax 8)
│   ├── in.lammps         # LAMMPS script; read_data/mass/pair_coeff overwritten from python
│   └── params.yaml       # new-style model config: lammps_input_script + pair_coeff
├── structures/
│   └── W-bcc.data        # 128-atom bcc W supercell (V/atom ≈ 16.16 Å³)
├── run_bcc.py            # sampling driver (new API)
├── analyse_bcc.py        # free-energy analysis + plot
└── output/               # generated data (pkl, dat, png, logs)
```

## How to run

Use the interpreter that has LAMMPS (ML-SNAP + PHONON) **and** the analysis
stack (numpy/scipy/scikit-learn/tqdm/mpi4py). In this checkout that is the
old-repo venv:

```bash
VENV=/home/marinica/GitHub/old_DescriptorDOS-Development.git/.venv/bin/python

cd workCosmin/01_bcc

# 1. sample the descriptor DOS (serial)
$VENV run_bcc.py                                   # defaults: 15 alpha in [0.5,4.0], 400 calls/alpha
# ... or parallel (total calls per alpha are split across ranks):
mpirun -np 8 $VENV run_bcc.py --calls 4800

# 2. free energy F(T) + plot
$VENV analyse_bcc.py
```

Outputs:
- `output/DDOSData-W-bcc-Hessian.pkl` — sampled DDOS histograms
- `output/hessian-W-bcc.pkl` — cached Hessian modes (per strain)
- `output/free_energy_W_bcc.dat` — `T, F_vib, F_total, F_harmonic` (eV/atom)
- `output/free_energy_W_bcc.png` — F(T) curve (anharmonic vs harmonic reference)

## Old → new API mapping

| old `run_bcc_test.py`                     | new `run_bcc.py`                                  |
|-------------------------------------------|---------------------------------------------------|
| `DDOSManager(world=..., worker=...)`       | same, but serial by default (`world=None`)        |
| `yaml_file="configuration-fine.yaml"`      | model YAML `models/W/params.yaml` merged via `**` |
| `lammps_input_script=.../in.lammps`        | supplied by `params.yaml` (`lammps_input_script`) |
| `input_path` + `input_data` (`.dat`)       | single `lammps_input_configuration=.../W-bcc.data`|
| `pair_coeff=` built by hand                | supplied by `params.yaml` (`pair_coeff`)          |
| `V_target=` (used to derive strain)        | not needed; strain grid is explicit               |
| Hessian pre-generated (`generate_hessian.sh`) | auto-generated & cached on first Hessian run   |
| `hessian_path`/`hessian_data`              | same kwargs (default under `dump_path`)           |
| analysis in `AnalyseSimulations/*.ipynb`   | `DDOSAnalyzer.score_free_energy` (`analyse_bcc.py`) |

## Notes

- **Descriptor dimension = 55** (`sna/atom … twojmax=8`). `X_W.snapcoeff` read
  with `skiprows=6` drops the 3 comments, the `1 56` line, the element line and
  the constant `β₀`, leaving the 55 bispectrum coefficients that match `D_0`.
- **Hessian vs Kinetic** — the Hessian isosurface carries an *analytic* harmonic
  reference, so `F(T)` is self-contained. The Kinetic isosurface samples the
  potential itself and needs an external `S(alpha)` fit
  (`DDOSAnalyzer.fit_S_alpha(Beta, F)`) before a free energy can be scored.
- The bundled `X_W.snapcoeff` is a **test** potential (`# Test values for W`):
  its static cohesive energy is ≈ −4.25 eV/atom, not the physical −8.9 eV/atom.
  Swap in a production W SNAP potential (same `twojmax`) to get physical numbers.
