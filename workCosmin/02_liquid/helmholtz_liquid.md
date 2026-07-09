# Absolute Helmholtz free energy of a liquid with D-DOS (Route 1)

Theory and numerical recipe behind `helmholtz_liquid.py`. Goal: the absolute
Helmholtz free energy `F(N,V,T)` per atom of the liquid W configuration stored in
`structures/W-liquid.data`, with **no** reference to any other phase and **no**
external thermodynamic-integration data.

---

## 1. The problem with a liquid

For a classical system the Helmholtz free energy is

$$
F(N,V,T) = -k_BT\,\ln Z,\qquad
Z = \frac{1}{N!\,\Lambda^{3N}}\int_V e^{-\beta U(\mathbf{x})}\,\mathrm{d}\mathbf{x},
\qquad \Lambda=\frac{h}{\sqrt{2\pi m k_BT}} .
$$

`F` splits into an ideal (momentum + combinatorial) part and a configurational
part. The momentum part is analytic; the hard part is the configurational
integral. For a **solid** there is a natural anchor — the harmonic crystal, whose
free energy is analytic — so one only has to add anharmonic corrections. A
**liquid** has *no* `T\to0` harmonic limit (it freezes), so there is no built-in
analytic anchor. Any absolute liquid free energy must be tied to a reference
whose absolute value is known.

D-DOS does not remove this requirement — it provides the sampling and a
score-matched estimator **relative to a reference**. Route 1 supplies that
reference *self-consistently from the configuration itself*.

---

## 2. D-DOS in one paragraph

Write the potential as a linear model of a descriptor,
$U(\mathbf{x}) = \boldsymbol\theta\cdot\mathbf{D}(\mathbf{x})$
(here `D` is the 55-component SNAP bispectrum, `θ` the coefficients in
`X_W.snapcoeff`). The **descriptor density of states** (D-DOS) is the
distribution of `D` on an isosurface labelled by `α` (a per-atom energy scale,
`α ≈ T/1000 K`). Its logarithm is the **descriptor entropy** `S(D\,|\,α)`.
Score-matching learns `∂S/∂D` from the sampled `D`, and the free energy follows
by a Legendre-type transform without any numerical integration:

$$
\beta F(\boldsymbol\theta,T)=\min_{\alpha}\Big[\;
\min_{D}\big(\beta\,\boldsymbol\theta\!\cdot\!D-S(D\,|\,\alpha)\big)
\;-\;S_\alpha(\alpha,T)\;\Big].
$$

`S_α(α,T)` is the **reference entropy**: the free energy of the reference system
that defines the isosurface. Everything hinges on how `S_α` is obtained.

---

## 3. Route 1 = the `KineticHessian` isosurface

Two ingredients define an isosurface displacer: **how it samples** and **what
reference it carries**.

| displacer        | sampling            | reference `S_α`              | needs external `F(β)`? |
|------------------|---------------------|------------------------------|------------------------|
| `Hessian`        | harmonic hypersphere| analytic harmonic            | no (solids only)       |
| `Kinetic`        | NVE of true potential| *unknown* → must be fitted   | **yes** (`fit_S_alpha`)|
| `KineticHessian` | **NVE of true potential** | **analytic harmonic**   | **no**                 |

`KineticHessian` is the hybrid we need:

1. **Sampling is real NVE** of the actual SNAP potential (`run_nve` after
   `target_energy`), so it explores the true anharmonic — liquid — configuration
   space, exactly like the pure `Kinetic` displacer.
2. **The isosurface is measured in a fixed Hessian mode basis**: the code builds
   the Hessian `H` of the input configuration once, keeps its stable modes
   `\{ \nu_\ell>0 \}`, and reports the harmonic energy
   `E = \tfrac12\langle \nu_\ell\, u_\ell^2\rangle\,\bar\omega` of each sampled
   displacement (`get_harmonic_energy`).
3. Because the *reference* is that harmonic basis, the record restores to a
   `HarmonicReferenceFreeEnergy` (`requires_fit == False`). The reference entropy
   is the analytic

$$
S_\alpha(\alpha,T)= -\tfrac{3}{2}\Big[\ln\!\big(3\,\beta\,\hbar^2 \bar\omega_{\text{eff}}/m\big)
+\langle\ln\nu_\ell\rangle-1-\alpha\Big],
$$

   i.e. the entropy of the `3N`-dimensional harmonic shell, with the absolute
   (momentum-included, correct `ħ`, `m`) normalization. **No `fit_S_alpha`,
   no second phase.**

The NVE sampling + score-matching then corrects this harmonic reference to the
true liquid, so the resulting `F` is **absolute**.

### Descriptor bookkeeping (why `Thetas = θ_SNAP`)
Only `isosurface=="Kinetic"` stacks the kinetic energy as an extra descriptor
column (`Manager.compress_data`). `KineticHessian` does **not**, so its descriptor
is the plain 55-dim SNAP vector and the model parameters are simply
`Thetas = θ_SNAP` — identical to a Hessian analysis. Hence

$$
F_{\text{abs}}(T)=E_0+\underbrace{\texttt{score\_free\_energy}(T,\theta_{\rm SNAP})[\text{'F'}]}_{\text{anharmonic, relative to harmonic ref.}},
\qquad E_0=\theta_{\rm SNAP}\!\cdot\!D_0 ,
$$

with `E_0` the static energy of the liquid snapshot.

---

## 4. Numerical steps (what `helmholtz_liquid.py` does)

1. **Input.** Read `structures/W-liquid.data` (128-atom liquid snapshot,
   V/atom ≈ 17.06 Å³). It is used *as-is* — do **not** relax it, or you collapse
   the liquid onto a nearby inherent structure.
2. **Build the reference basis (once).** `KineticHessian` calls LAMMPS
   `dynamical_matrix` at the snapshot, diagonalises `H`, keeps the stable modes
   `\nu_\ell>0`, and caches them in `output/hessian-W-liquid-kh.pkl`. The run log
   prints `Zero Modes = k/3N`: `k` counts the non-stable directions of the liquid
   snapshot (expected to be sizeable — that is the liquid).
3. **Sample the D-DOS.** For each isosurface value
   `α ∈ linspace(amin,amax,nalpha)` (`~T/1000 K`), draw `calls` NVE samples:
   `target_energy(2 U_0 α)` → `run_nve` → record `D` (SNAP) and the harmonic
   energy. Compress to histograms → `output/DDOSData-W-liquid-KineticHessian.pkl`.
4. **Score-match.** `DDOSAnalyzer.match_score` fits a polynomial score model
   `S(D|α)` per isosurface.
5. **Evaluate F.** For each `T`,
   `F_abs = E0 + score_free_energy(T, θ_SNAP)['F']`. The harmonic reference curve
   `E0 + reference_free_energy.free_energy(T)` is printed alongside so you can see
   how much of `F` is anharmonic (liquid) correction.
6. **Output.**
   - `output/helmholtz_W_liquid.dat` — `T, F_absolute, F_harmonic_reference` (eV/atom)
   - `output/helmholtz_W_liquid.png` — `F(T)` and the harmonic reference

### Commands
```bash
VENV=/home/marinica/GitHub/old_DescriptorDOS-Development.git/.venv/bin/python
cd workCosmin/02_liquid

# run + analyse in one shot (samples on first call, caches, then analyses)
$VENV helmholtz_liquid.py                       # defaults: 15 alpha in [0.5,6.0], 200 calls/alpha
$VENV helmholtz_liquid.py --calls 2000 --nalpha 20 --Tmax 6000   # production
$VENV helmholtz_liquid.py --analyse-only        # re-analyse cached sampling
```

### Cost and MPI
`KineticHessian` re-targets the total energy on essentially every draw (its
`target_energy` runs ~`2N` NVE steps), so it is the most expensive isosurface
here — order ~5–7 s per call for this 128-atom cell. Budget accordingly and
raise `--calls` only as far as wall time allows. The Hessian of the snapshot is
built once and cached in `output/hessian-W-liquid-kh.pkl`, so re-runs skip it.

`helmholtz_liquid.py` supports `mpirun -np N` (per-alpha calls split across
ranks), **but only with a launcher matching the venv's mpi4py build**. In this
checkout mpi4py is built against *Intel MPI*; the machine's default `mpirun`
(OpenMPI) silently gives every rank `size == 1` (independent jobs writing the
same file). Use Intel MPI's `mpiexec`/`mpirun`, or run serial.

---

## 5. Assumptions, validity and caveats

- **Single snapshot.** `E0` and the mode basis come from the one configuration in
  `W-liquid.data`; the absolute `F` therefore carries that snapshot's static
  energy. For an ensemble liquid free energy, repeat over several independent
  snapshots (`structures/W-liquid*.data`, `--append`) and average.
- **Harmonic basis of a liquid.** The reference uses the *stable* modes of an
  instantaneous (force ≠ 0) liquid configuration; many modes are unstable and are
  dropped. The reference is thus an *effective* harmonic anchor, and the NVE
  score-matching must bridge a larger gap than in a solid. Check the
  `Zero Modes = k/3N` line: the larger `k`, the more the result leans on the
  score model rather than the analytic reference.
- **Score-model expressivity.** For strongly anharmonic / entropically stabilised
  systems the conditional descriptor entropies can be multi-modal; the simple
  low-rank score model used here may under-resolve them (the D-DOS paper flags
  low-rank-tensor / neural-network score models as the fix, and treats liquid
  free energies as a *forthcoming* study). Treat absolute numbers as approximate.
- **Convergence.** NVE sampling is ~50× costlier per call than harmonic sampling;
  use large `--calls` for production and check stability of `F(T)` against
  `--calls`, `--nalpha`, `--match-order`.
- **Test potential.** `X_W.snapcoeff` is `# Test values for W`; energies are not
  physical. Swap a production W SNAP (same `twojmax=8`) for real numbers.
- **Constant volume.** This is `F(N,V,T)` at the liquid snapshot's volume — a
  Helmholtz (NVT) free energy, not a `P=0` Gibbs free energy.
