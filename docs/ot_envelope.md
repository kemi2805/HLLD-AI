# Measured interface envelope of the Orszag–Tang vortex

Companion to [`rotor_envelope.md`](rotor_envelope.md), same methodology,
second problem. Purpose: check whether a network trained on the rotor
envelope would actually cover OT too, or extrapolate on it the same way the
shock-tube-trained network extrapolated on the rotor — measured, not
assumed, per this project's own rule after that mistake.

Source: `results/ot_128_hlld_env` — a plain-HLLD 128² Orszag–Tang run to
t = 1.0 (1317 SSP-RK3 steps, 3401 s, γ = 4/3, CFL 0.25, MC limiter,
Gardiner–Stone constrained transport, periodic BCs). n=128 and t_end=1.0 are
this repo's own OT convention (`PROBLEMS["orszag_tang"]` in
`scripts/run_2d.py`), not invented for this measurement. Raw samples:
`data/ot_envelope_128.npz` (792,000 sampled interface states, recorded by
`src/physics/envelope.py` — same recorder, same `every=30`/`samples=3000`
settings as the rotor run, for a fair comparison).

Reproduce with:

```bash
python scripts/run_2d.py --problem orszag_tang --n 128 --tend 1.0 --solver hlld \
    --out results/ot_128_hlld_env \
    --envelope data/ot_envelope_128.npz --envelope-every 30 --envelope-samples 3000
```

These are the states the Riemann solver is actually handed, in the solver's
own normal/transverse decomposition — the same decomposition the network's
features use.

## The envelope

| field | min | 0.1% | median | 99.9% | max |
|---|---|---|---|---|---|
| `log10 rho_L` | −1.801 | −1.700 | −0.673 | 0.563 | 1.237 |
| `log10 p_L`   | −2.405 | −2.270 | −0.866 | 1.087 | 1.794 |
| `W_L`         | 1.000 | 1.000 | 1.230 | 6.212 | 6.986 |
| `\|B_t\|_L`   | 0.000 | 0.000 | 0.180 | 1.515 | 3.534 |
| `log10 rho_R` | −1.801 | −1.700 | −0.673 | 0.564 | 1.237 |
| `log10 p_R`   | −2.405 | −2.270 | −0.866 | 1.080 | 1.795 |
| `W_R`         | 1.000 | 1.000 | 1.230 | 6.206 | 6.985 |
| `\|B_t\|_R`   | 0.000 | 0.000 | 0.180 | 1.524 | 3.529 |
| `\|B_n\|`     | 0.000 | 0.000 | 0.182 | 1.562 | 5.096 |
| `cos∠(v_t,B_t)` | −1.000 | −1.000 | 1.000 | 1.000 | 1.000 |
| `sigma = b²/(ρh)` | 0.000 | 0.000 | 0.072 | 0.913 | 1.631 |

## OT vs the rotor, dimension by dimension

Both measured the same way, so this is a direct comparison, not a guess —
and it says a rotor-trained network is **not** automatically safe on OT:

| quantity | rotor max (99.9%) | OT max (99.9%) | which is wider |
|---|---|---|---|
| `log10 rho` | 1.180 (1.031) | 1.237 (0.563/0.564) | similar max; **OT reaches much lower density** (min −1.80 vs −0.54) |
| `log10 p`   | 0.915 (0.658) | 1.794/1.795 (1.087/1.080) | **OT reaches much higher pressure** — 99.9th pct alone is 0.4 dex above the rotor's *max* |
| `W`         | 8.364 (3.192/3.196) | 6.986 (6.212/6.206) | rotor wider at the extreme, but **OT's 99.9th-percentile tail is ~2× fatter** (6.21 vs 3.19) |
| `\|B_t\|`   | 3.972/3.979 | 3.534/3.529 | rotor slightly wider |
| `\|B_n\|`   | 5.093 | 5.096 | essentially identical |
| `sigma`     | 7.847 (4.315) | 1.631 (0.913) | **rotor is far more magnetized** — almost 5× the max, 4.7× the 99.9th percentile |

So the two problems stress different corners: **OT pushes pressure and the
high-`W` tail harder than the rotor does**, while **the rotor pushes
magnetization (`sigma`) far harder than OT does**. A network trained only on
the rotor envelope would very likely extrapolate on OT's low-density/
high-pressure states; a network trained only on OT would badly under-cover
the rotor's high-`sigma` regime. If both problems matter, the training range
should be the *union*, not either envelope alone — this table is what that
union should be built from, not a guess.

## Run health, for reference

| | |
|---|---|
| div B | ~3.1e−12 – 3.4e−12 for the whole run (machine precision; CT working; note this is *smaller* than the rotor's 1.0e−11, plausibly because OT has no rotor-style discontinuous initial jump to seed round-off) |
| c2p non-convergences | effectively 0 for most of the run; isolated single-cell blips (`c2p_bad` 1–3) appeared only late, around t≈0.92–0.98 |
| rho range | 0.221 initial-ish → up to ~17.3 transient peak (~t=0.38) → settling toward ~0.05–1.9 by t=1.0 |

No π-rotation symmetry check here — that diagnostic is rotor-specific
(`scripts/run_rotor.py::symmetry_error` exploits the rotor's exact π
rotational symmetry, which OT does not share).
