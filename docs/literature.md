# What is new here: the literature (checked 2026-10-13)

What this work does that the literature has not, and what it shares with
earlier work, claim by claim.

**How it was checked.** Five parallel web searches, one per claim, each
listing only papers whose abstract or full text was opened. The quotes that
carry a verdict were then re-checked at the source. This is not a
systematic review:
- no Scopus or Web of Science;
- some full texts were blocked (ScienceDirect, some Cambridge pages);
- theses, unindexed reports and practice that lives only in code are not
  covered.

Re-check before claiming priority in a paper ("Before a paper", below).

## In one table

| Our result | Closest prior work | Verdict |
|---|---|---|
| An exact seven-wave RMHD Riemann solver as the production flux of 2D CT runs (rotor, Orszag–Tang, four-quadrant), millions of verified solves per run | Pons, Martí & Müller 2000: exact relativistic **hydro** flux at every interface of a 2D code. Quirk 1994: exact Euler flux in 2D, HLLE near shocks. In RMHD only approximate solvers with an iterative closure: HLLD, Kim–Balsara HLLC, Antón et al.'s full-wave Roe | **New for RMHD.** The literature calls it infeasible in multi-D (§1) |
| A per-face, located census of where and why the exact solve fails | HLLD's fallback "usually less than 0.1%" of zones, no locations (Mignone, Ugliano & Bodo 2009); Musolino 2025, "occasional failures" | **The census is new.** The failure *classes* are known from theory (§2) |
| Rescue stages: planar five-wave, four-unknown, continuation | Takahashi & Yamada 2014 (strength parametrisation, frozen rotation angles); homotopy continuation after an ML seed in power flow (Okhuegbe et al. 2024) | **Modest.** The ingredients are known; their use as fallback stages for an RMHD solver, and the measured gain (94% → 97–99%), are ours |
| The weak-jump gate with a linearised flux below it | Toro 1991: a linearised solver over >99% of the flow, exact only for strong jumps. Wen, Panaitescu & Laguna 1997 | **Not new as a design.** Our measurement (25–120× closer than HLLD) is |
| Neural-network seeds for the Newton iteration, which still runs to a verified residual | Musolino 2025: networks *replace* the root-find (1D relativistic hydro). Mudimadugula et al. 2025: a network *seeds* the c2p root-finder in 3D neutron-star runs, marginal gain. Ma et al. 2023: lookup seeds for an exact bimaterial solver in 2D | **New for any MHD or RMHD Riemann solver.** Every neural Riemann solver found replaces the solve |
| The seed ceiling: better seeds stop helping; retries from other seeds do | Giacomazzo & Rezzolla 2006 already found the basin narrow. Sharony et al. 2024: diverse multiple seeds beat one learned seed (optimal control) | **New for Riemann or c2p solvers.** The principle has analogues elsewhere |
| Canonical orientation over the 32-element symmetry orbit | Musolino 2025: reflects to p_L ≥ p_R and reflects the flux back (two elements, to simplify learning). Torrilhon 2002: B_n ≥ 0 as a convention. Zhang et al. 2026: mirror symmetry by *averaging*. ML: Kaba et al. 2023 (canonicalisation); Dym et al. 2024 (canonicalisation must jump at ties) | **New as a symmetry technique.** There are precedents for parts of it (§4) |
| An iterative sub-solver breaks mirror symmetry through whether and how it converges | Symmetry fixes so far reorder floating-point operations inside fixed formulas: Aloy et al. 1999; Fleischmann et al. 2019; Wakimura et al. 2021; Stone et al. 2020 | **Not reported before**, as far as found |
| Exact flux against resolution: 0.1–7% of the gap to 512² at 64² | Approximate solvers only, mostly "HLL needs ~2× the zones of HLLD" (MUB 2009, B&S 2011, Antón 2010, Mattia & Mignone 2022). High order shrinks the solver's effect in smooth flows (Balsara 2004, 2025) | **New.** The end point every approximate flux tends to (§5) |
| Test problems | Fast Orszag–Tang: van der Holst et al. 2008 (0.99c, W ≈ 7, γ = 4/3). Beckwith & Stone 2011 (v = 0.5, γ = 5/3). Rotor: Del Zanna et al. 2003. Four-quadrant: hydro only (Del Zanna & Bucciantini 2002) | Our Orszag–Tang is a **colder variant** of van der Holst's. **No magnetised RMHD four-quadrant problem was found** (§6) |

## 1. An exact RMHD Riemann solver as the production flux

**What exists.**
- **Exact solvers for RMHD are test references.**
  - Romero et al. 2005 (JFM 544, 323) cover only the purely tangential field.
  - Giacomazzo & Rezzolla 2006 (JFM 562, 223) give the general seven-wave
    solution, by Newton from an HLLE seed, with code "available from the
    authors upon request". The stated purpose is to test codes.
- **Their own production code does not use it.** Giacomazzo & Rezzolla's
  WhiskyMHD (2007, CQG 24, S235) uses HLLE because the exact solver is
  "still computationally too expensive to be used in ordinary
  multidimensional codes".
- **Every later RMHD code paper concludes the same**, each choosing
  something cheaper:
  - van der Holst et al. 2008 (CPC 179, 617) choose TVDLF;
  - Mignone, Ugliano & Bodo 2009 (HLLD, MNRAS 393, 1141) call the exact
    flux prohibitive;
  - Mattia & Mignone 2022 (MNRAS 510, 481) say it "is not a feasible
    option to solve a multidimensional problem".
- **Athena refuses it in code.** Its exact relativistic solver
  (`src/rsolvers/exact_sr.c`) is hydro only and contains
  `#error : The exact flux for MHD has not been implemented.`
- **Hydro is different.** Exact or iterative solvers have been production
  fluxes for decades:
  - Martí & Müller's relativistic PPM, in 1D;
  - Pons, Martí & Müller 2000 (JFM 422, 125), at every interface of a 2D
    Godunov code;
  - the two-shock solver of Mignone, Plewa & Bodo 2005 (ApJS 160, 199), in
    1D–3D;
  - Musolino 2025, in 1D.
- **Non-relativistic MHD.** The nonlinear MHD solvers used as fluxes are
  approximate: Dai & Woodward 1994 treat every wave as a discontinuity.
  Ryu & Jones 1995 and Torrilhon 2003 use their exact solvers as
  references, and Takahashi & Yamada 2014 only propose Godunov use for the
  future.
- **Closest in RMHD.** Approximate solvers with an iterative closure and a
  per-face HLL fallback: HLLD; Kim & Balsara's HLLC (JCP 270, 634, 2014);
  and Antón et al. 2010 (ApJS 188, 1), a full-wave Roe solver at 1.5–3×
  HLL's cost.

**Hybrid designs that ours repeats.**
- Toro 1991 (Cranfield report 9107) uses a linearised solver over more than
  99% of the flow and the exact one only for strong jumps. That is our
  weak-jump gate.
- Quirk 1994 (IJNMF 18, 555) runs an exact Euler flux in 2D and substitutes
  HLLE face by face near shocks. That is our per-face HLLD fallback.
- Komissarov 1999 (MNRAS 303, 343) falls back from his linearised RMHD
  solver to HLLE.

**Verdict.** Nothing found runs an exact RMHD Riemann solver as the
production flux of multi-D runs. Four groups and one public code say it
cannot be done affordably. The designs around it (adaptive solver,
fallback) are known.

## 2. Where it fails, and why

**Every failure class we found is known in theory. None has been measured
in a real flow.**

- **Compound waves** (a field reversal fused to a shock, outside the
  regular seven-wave family):
  - Brio & Wu 1988 introduced them.
  - Barmin, Kulikovskiy & Pogorelov 1996 (JCP 126) show any shock-capturing
    scheme produces them on coplanar data.
  - Torrilhon 2003 (JCP 192, 73) shows coplanar data have two exact
    solutions, and that Roe, HLLE and central schemes "pseudo-converge" to
    the compound one until the grid is very fine. This is the precedent
    for our finding that the dissipation, not the flux formula, chooses
    between Balsara test 1's two answers.
  - Takahashi & Yamada 2013 (J. Plasma Phys. 79) find states with no
    regular solution and others with uncountably many.
  - Giacomazzo & Rezzolla 2006 exclude compound waves by construction.
- **Coplanar degeneracy of the rotation angles.** Takahashi & Yamada 2014
  (J. Plasma Phys. 80, 255): the rotational-discontinuity map is
  non-injective, so the Jacobian is singular, and they freeze the angles.
  This is non-relativistic, but it is our coplanar degeneracy and our fix
  (the planar rung).
- **Field nearly normal to the face:**
  - Antón et al. 2010 define the Type II degeneracy (an Alfvén wave
    coincides with the fast or slow wave) and show it is frame-dependent
    in RMHD.
  - Giacomazzo & Rezzolla 2006 (Appendix B) leave the rarefaction ODEs
    unregularised where a fast or slow speed equals the Alfvén speed.
  - Zhang, Lani & Poedts 2025 (arXiv:2507.10420) report HLLD in trouble
    "when the normal component of the magnetic field is dominant".
  - Taylor & Cargill 2001 describe the fast↔slow exchange at the umbilic
    point.
  - Our account adds the cause in the solver: tangential field is the
    wrong slow-wave coordinate there.
- **Magnetic null sheets.** Indirect precedent only:
  - the Type I degeneracy (B_n → 0);
  - Snow et al. 2021, who place intermediate shocks at the high-current
    regions of the (non-relativistic) Orszag–Tang vortex.
- **Failure rates in the literature:**
  - HLLD: "usually less than 0.1%" of zones fall back, with no locations;
  - Mattia & Mignone 2022: HLLD and HLLC need the HLL failsafe at strong
    magnetisation, with no rates;
  - the public port of Torrilhon's solver (github.com/torrilhon/
    ExactMHDRiemannApp) reports "NoConvergence" in a few percent of
    problems near its tangential-field thresholds.

**Verdict.** The failure classes are textbook. A measured, per-face census
inside real multi-D flows was not found, for an exact solver or for an
approximate one's fallbacks. That census includes:
- how often the solver fails;
- where, located on the grid;
- what each fallback stage recovers;
- that the remaining 1–3% are structural, not near-misses.

## 3. Machine-learned seeds

- **Neural Riemann solvers replace the solve.** Musolino 2025
  (arXiv:2505.18914; relativistic hydro, 1D; networks replace the root-find,
  HLLE fallback) is the closest in physics. The others are hydro or shallow
  water:
  - Magiera et al. 2020 (JCP 409);
  - Ruggeri et al. 2022 (Phys. Fluids 34);
  - Peyvan et al. 2024 (RiemannONets);
  - Thakur & Zahr 2025;
  - Zhang et al. 2026 (arXiv:2603.30007).

  The one MHD flux network found (Kim, Ha & Kang 2024, arXiv:2404.16015)
  is non-relativistic and also replaces the flux.
- **Seeding a verified solve:**
  - Mudimadugula et al. 2025 (arXiv:2503.08289) use a network as the
    initial guess of the c2p root-finder in 3D neutron-star runs. The
    speed-up was "subdominant" (−5% to +5%), consistent with our ceiling.
  - Ma et al. 2023 (JCP 493) seed an exact bimaterial Riemann solver in 2D
    from a nearest-neighbour lookup of earlier solves. That is data-driven
    but not learned, and its gain is case-dependent.
  - Learned Newton seeds elsewhere: Jin et al. 2025 (implicit time
    stepping); Aghili et al. 2024 (elliptic PDEs).
- **The ceiling:**
  - Giacomazzo & Rezzolla 2006 already report that Newton needs a guess
    "sufficiently close to the exact solution".
  - Sharony et al. 2024 (arXiv:2411.02158) show diverse multiple seeds
    beating one learned seed in optimal control.
  - Okhuegbe et al. 2024 (IEEE Trans. Ind. Appl.) put homotopy
    continuation after an ML seed in power flow.

**Verdict.** Neural-network seeds for an MHD or RMHD Riemann solver: not
found, so new. The seed ceiling and multistart finding is new for Riemann
and c2p solvers but has analogues in other fields, so present it as the
quantified RMHD case of a known principle.

## 4. Canonical orientation

- **How symmetry is preserved today.** Shock-capturing codes fix the
  operation order inside fixed formulas:
  - Aloy et al. 1999 (GENESIS): symmetrised flux expressions and cyclic
    sums;
  - Fleischmann, Adami & Adams 2019 (Comput. Fluids 189);
  - Wakimura, Takagi & Xiao 2021 (arXiv:2106.01872): bracketed sums and
    HLLC's s* = 0 branch averaged;
  - Stone et al. 2020 (Athena++): no FMA, no reassociation.

  Pakmor, Bauer & Springel 2011 report AREPO losing Orszag–Tang symmetry
  through the flux-update order. None of these can help when the iteration
  count and the solve-or-fail outcome depend on the data, which is our
  case.
- **Precedents for parts of the technique:**
  - Musolino 2025 reflects each face so that p_L ≥ p_R and reflects the
    flux back, to reduce the wave patterns a network must learn. That is a
    two-element canonicalisation with a strict comparison and no tie
    tolerance.
  - Torrilhon 2002 (ETH SAM report 2002-06) assumes B_n ≥ 0 throughout,
    as a convention.
  - Zhang et al. 2026 enforce mirror symmetry by averaging the network
    over the two-element group, and Galilean and scaling invariance by
    normalising inputs. Unconstrained, their implosion depends on the face
    orientation.
- **The ML analogues:**
  - Kaba et al. 2023 (ICML): a canonicalising map in front of a
    non-equivariant model. This is exactly our structure.
  - Puny et al. 2022 (frame averaging); Ma et al. 2024 (frames and
    canonical forms).
  - Dym, Lawrence & Siegel 2024 (ICML) prove that a canonical choice must
    be discontinuous at orbit ties. That is why the rule has a tolerance:
    the 1e-7 key moves the discontinuity away from rounding noise but
    cannot remove it.

**Verdict.** Not found before:
- canonicalising the inputs of an iterative or exact Riemann solver over
  its full symmetry orbit (reflection, signed exchange of the tangential
  axes, B → −B) with a tolerance-based key;
- the observation that the solver's *outcome* seeds the asymmetry.

The technique applies to any iterative or learned flux. It is the most
transferable result here.

## 5. How much the flux matters

- **Approximate solvers against resolution:**
  - Mignone, Ugliano & Bodo 2009: HLL needs "almost twice the number of
    grid zones" of HLLD on the 3D rotor; 1D L1 ratios HLL : HLLC : HLLD ≈
    1 : 0.6 : 0.27.
  - Beckwith & Stone 2011: HLLE is "more diffusive" on Orszag–Tang
    (qualitative); HLLD has about 2× HLLE's effective spectral resolution
    in 3D Kelvin–Helmholtz.
  - Antón et al. 2010: the full-wave solver is sharper than HLL "even with
    half the resolution", and ≈ HLLC on the rotor.
  - Mattia & Mignone 2022: L1 errors only for the circularly polarised
    Alfvén wave; on Kelvin–Helmholtz, HLLD at 256×512 matches HLL at
    512×1024.

  None measures L1(ρ) on the rotor, Orszag–Tang or a 2D Riemann problem
  against a high-resolution reference, and none reaches the exact end
  point.
- **Reconstruction against the solver:**
  - Balsara 2004 (astro-ph/0308249) and Balsara et al. 2025
    (arXiv:2506.11181): at higher order the Riemann solver "becomes less
    important" (smooth flows). This agrees with our WENO5-Z result.
  - Kiuchi et al. 2022 (arXiv:2205.04487) qualify it: HLLE's smearing of
    tangential discontinuities is not cured by higher order.
  - Mignone & Del Zanna 2021 (JCP 424) show that the CT electric-field
    averaging moves the Orszag–Tang centre about as much as doubling the
    resolution. Ours is Gardiner & Stone's upwinded corner EMF built from
    the solver's own face fluxes (`src/physics/ct.py`); a paper must say
    so.
- **Resolution in relativistic Orszag–Tang:**
  - Kayanikhoo et al. 2024 (MNRAS 527, 10151; 0.99c, γ = 4/3): the
    reconnection rate converges only at ≥ 2048², with plasmoids at 4096².
    This agrees with our unconverged filaments at 512².
  - Balsara & Kim 2016 (arXiv:1602.00633): the rotor reaches its nominal
    W = 10 only at ≥ 1200². Our 64²–512² rotors are below it (W 8.4 at
    256², measured on our 256² run).

**Verdict.** Measuring the exact flux against resolution is new. It is the
upper bound on what any approximate flux can gain, and it answers the
"HLL at 2N" literature from the other end.

## 6. The test problems

- **Orszag–Tang.** The fast regime is van der Holst, Keppens & Meliani 2008
  (CPC 179, 617; [0, 2π]², ρ = 1, p = 10, peak 0.99, W ≈ 7, γ = 4/3, TVDLF,
  effective 2560²). It is also run by BHAC (Porth et al. 2017), Wu & Shu
  2021 and Kayanikhoo et al. 2024. Beckwith & Stone 2011 run v = 0.5 per
  component with γ = 5/3, as do Zhao & Tang 2017 and Duan & Tang 2020.
  Ideal RMHD is unchanged under ρ, p, B² → λ(ρ, p, B²). In van der Holst's
  units (B0 = 1), ours is therefore ρ = 25/9, p = 5/3:
  - a colder gas, p/ρ = 0.6 against 10;
  - a stronger field relative to the gas, B0²/p = 0.6 against 0.1.

  Call it a variant of van der Holst's, not a new test.
- **Rotor.** Del Zanna, Bucciantini & Londrillo 2003 (A&A 400, 397).
- **Four-quadrant problem.** Defined for relativistic hydro by Del Zanna &
  Bucciantini 2002. **No magnetised RMHD four-quadrant problem was
  found.** Our 45° field version would be new as a test, but check again
  before saying so.

## How to phrase the claims

1. *An exact Riemann solver for relativistic MHD as a production flux:* how
   often it can be solved, why it fails, and what it changes. These are
   §1, §2 and §5 together; the strongest and most clearly new.
2. *Canonical orientation:* making an iterative or learned Riemann solver
   mirror-consistent by construction. This is §4, short and general, and
   could be a separate note.
3. *Seeds:* report the ceiling as a measured negative result, citing
   Giacomazzo & Rezzolla 2006 and Sharony et al. 2024. Do not lead with it.

**Say exactly what the flux is.** In production it comes from three
sources:
- the exact Riemann flux above a relative jump of 1e-2;
- a linearised flux, second order in the jump, below it;
- the upwind state where every wave moves one way.

The share below the gate grows with resolution: 73% of the faces at 64²,
99.5% at 256² on Orszag–Tang. Results from the runs with the gate open
(TAU_WEAK=1e-12) are what license "exact everywhere".

**Do not claim:**
- the test problems;
- the adaptive gate or the fallback design (Toro 1991, Quirk 1994);
- the failure classes themselves (§2);
- that the Riemann solver matters less than reconstruction in general
  (Balsara 2004, 2025, qualified by Kiuchi et al. 2022).

## Before a paper

- **Open what could not be opened:**
  - Dai & Woodward 1994b (JCP 115, 485): do they run the nonlinear solver
    at every face in multi-D?
  - Kim & Balsara 2014;
  - Miyoshi & Kusano 2005;
  - Torrilhon 2002 (the ETH report);
  - Clain, Diot & Loubère 2011 (MOOD);
  - Andreev et al. 2008 (an online exact MHD solver);
  - Gyrya et al. 2024 (networks predicting p*).
- **Search what this did not:**
  - theses from the Rezzolla, Martí–Ibáñez and Balsara groups;
  - GPU or batched exact solvers;
  - Scopus and Web of Science for "exact Riemann solver" with
    "magnetohydrodynamics" and "Godunov".
- **Methods to state in the paper**, because the literature shows they move
  results:
  - the CT electric field (Mignone & Del Zanna 2021);
  - that the 64²–512² rotor is below its nominal W (Balsara & Kim 2016).

## References (all opened)

- Aloy, Ibáñez, Martí & Müller 1999, ApJS, [astro-ph/9903352](https://arxiv.org/abs/astro-ph/9903352)
- Antón et al. 2010, ApJS 188, 1, [arXiv:0912.4692](https://arxiv.org/abs/0912.4692)
- Balsara 2004, ApJS, doi:10.1086/381377, [astro-ph/0308249](https://arxiv.org/abs/astro-ph/0308249)
- Balsara & Kim 2016, JCP, doi:10.1016/j.jcp.2016.02.001, [arXiv:1602.00633](https://arxiv.org/abs/1602.00633)
- Balsara et al. 2025, [arXiv:2506.11181](https://arxiv.org/abs/2506.11181)
- Barmin, Kulikovskiy & Pogorelov 1996, JCP 126, doi:10.1006/jcph.1996.0121, [OSTI 440757](https://www.osti.gov/biblio/440757)
- Beckwith & Stone 2011, ApJS 193, 6, [arXiv:1101.3573](https://arxiv.org/abs/1101.3573)
- Del Zanna & Bucciantini 2002, A&A, [astro-ph/0205290](https://arxiv.org/abs/astro-ph/0205290)
- Del Zanna, Bucciantini & Londrillo 2003, A&A 400, 397, [astro-ph/0210618](https://arxiv.org/abs/astro-ph/0210618)
- Dieselhorst, Cook, Bernuzzi & Radice 2021, Symmetry 13, 2157, [arXiv:2109.02679](https://arxiv.org/abs/2109.02679)
- Dym, Lawrence & Siegel 2024, ICML, [PMLR 235](https://proceedings.mlr.press/v235/dym24a.html)
- Falle & Komissarov 2001, J. Plasma Phys. 65, 29, [physics/9904035](https://arxiv.org/abs/physics/9904035)
- Fleischmann, Adami & Adams 2019, Comput. Fluids 189, 94, [TUM](https://portal.fis.tum.de/en/publications/numerical-symmetry-preserving-techniques-for-low-dissipation-shoc/)
- Giacomazzo & Rezzolla 2006, JFM 562, 223, [gr-qc/0507102](https://arxiv.org/abs/gr-qc/0507102)
- Giacomazzo & Rezzolla 2007, CQG 24, S235, [gr-qc/0701109](https://arxiv.org/abs/gr-qc/0701109)
- Kaba et al. 2023, ICML, [PMLR 202](https://proceedings.mlr.press/v202/kaba23a.html)
- Kayanikhoo et al. 2024, MNRAS 527, 10151, [arXiv:2308.16062](https://arxiv.org/abs/2308.16062)
- Kim, Ha & Kang 2024, [arXiv:2404.16015](https://arxiv.org/abs/2404.16015)
- Kiuchi, Held, Sekiguchi & Shibata 2022, [arXiv:2205.04487](https://arxiv.org/abs/2205.04487)
- Komissarov 1999, MNRAS 303, 343, [OUP](https://academic.oup.com/mnras/article/303/2/343/1067242)
- Liska & Wendroff 2003, SIAM J. Sci. Comput. 25, 995, [PDF](https://www.math.umd.edu/~tadmor/centpack/publications/files/Liska-Wendroff_1-2D_comparison_tests_SISC2003-6-centpack.pdf)
- Ma, Zhao, Islam, Narkhede & Wang 2023, JCP 493, 112474, [arXiv:2303.08743](https://arxiv.org/abs/2303.08743)
- Magiera, Ray, Hesthaven & Rohde 2020, JCP 409, 109345, [arXiv:1904.12794](https://arxiv.org/abs/1904.12794)
- Mattia & Mignone 2022, MNRAS 510, 481, [arXiv:2111.09369](https://arxiv.org/abs/2111.09369)
- Mignone & Bodo 2006, MNRAS 368, 1040, [astro-ph/0601640](https://arxiv.org/abs/astro-ph/0601640)
- Mignone & Del Zanna 2021, JCP 424, 109748, [arXiv:2004.10542](https://arxiv.org/abs/2004.10542)
- Mignone, Plewa & Bodo 2005, ApJS 160, 199, [astro-ph/0505200](https://arxiv.org/abs/astro-ph/0505200)
- Mignone, Ugliano & Bodo 2009, MNRAS 393, 1141, [arXiv:0811.1483](https://arxiv.org/abs/0811.1483)
- Mudimadugula et al. 2025, [arXiv:2503.08289](https://arxiv.org/abs/2503.08289)
- Musolino 2025, [arXiv:2505.18914](https://arxiv.org/abs/2505.18914)
- Okhuegbe, Ademola & Liu 2024, IEEE Trans. Ind. Appl. 61, [OSTI 3030312](https://www.osti.gov/pages/biblio/3030312)
- Pakmor, Bauer & Springel 2011, MNRAS, [arXiv:1108.1792](https://arxiv.org/abs/1108.1792)
- Pons, Martí & Müller 2000, JFM 422, 125, [astro-ph/9910462](https://arxiv.org/abs/astro-ph/9910462)
- Porth et al. 2017 (BHAC), [arXiv:1611.09720](https://arxiv.org/abs/1611.09720)
- Puny et al. 2022, ICLR, [arXiv:2110.03336](https://arxiv.org/abs/2110.03336)
- Quirk 1994, IJNMF 18, 555, [NTRS](https://ntrs.nasa.gov/api/citations/19930015894/downloads/19930015894.pdf)
- Romero et al. 2005, JFM 544, 323, [astro-ph/0506527](https://arxiv.org/abs/astro-ph/0506527)
- Ryu & Jones 1995, ApJ 442, 228, [astro-ph/9404074](https://arxiv.org/abs/astro-ph/9404074)
- Sharony et al. 2024, [arXiv:2411.02158](https://arxiv.org/abs/2411.02158)
- Snow, Hillier, Murtas & Botha 2021, Exp. Results, [arXiv:2111.02242](https://arxiv.org/abs/2111.02242)
- Stone, Tomida, White & Felker 2020 (Athena++), [arXiv:2005.06651](https://arxiv.org/abs/2005.06651)
- Takahashi & Yamada 2013, J. Plasma Phys. 79, [arXiv:1210.5584](https://arxiv.org/abs/1210.5584)
- Takahashi & Yamada 2014, J. Plasma Phys. 80, 255, [arXiv:1310.2330](https://arxiv.org/abs/1310.2330)
- Toro 1991, Cranfield CoA Report 9107, [Cranfield](https://aerade.cranfield.ac.uk/handle/1826/226)
- Torrilhon 2003, JCP 192, 73, [PDF](https://www.math.umd.edu/~tadmor/centpack/publications/files/Torrilhon_JCP2003-centpack.pdf)
- van der Holst, Keppens & Meliani 2008, CPC 179, 617, [arXiv:0807.0713](https://arxiv.org/abs/0807.0713)
- Wakimura, Takagi & Xiao 2021, [arXiv:2106.01872](https://arxiv.org/abs/2106.01872)
- Wen, Panaitescu & Laguna 1997, ApJ 486, 919, [astro-ph/9612045](https://arxiv.org/abs/astro-ph/9612045)
- Wu & Shu 2021, Numer. Math., [arXiv:2002.03371](https://arxiv.org/abs/2002.03371)
- Zhang, Lani & Poedts 2025, [arXiv:2507.10420](https://arxiv.org/abs/2507.10420)
- Zhang, Wichitrnithed, Cai, Dutta, Mandli & Dawson 2026, [arXiv:2603.30007](https://arxiv.org/abs/2603.30007)
- Athena (C version), `src/rsolvers/exact_sr.c`, [GitHub](https://github.com/PrincetonUniversity/Athena-Cversion/blob/master/src/rsolvers/exact_sr.c)
