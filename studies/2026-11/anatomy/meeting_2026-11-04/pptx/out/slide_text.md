<!-- Slide number: 1 -->
# Wed Nov 04
Meeting with Luciano
Exact RMHD — the four-quadrant problem at 128²: what works, what fails, and why

### Notes:
Three parts: what we have (accuracy, cost, networks), what fails (the 0.8% of attempted faces lost to HLLD, and where), why (two wave-speed degeneracies explain every rescue and every loss), and what would fix it.

<!-- Slide number: 2 -->
# Topics
What we have
What fails
Why it fails
What would fix it

### Notes:
All on the four-quadrant 128² run with the retrained network ft3 (calea 77226).

<!-- Slide number: 3 -->
# The four-quadrant problem: initial data
ρ = 0.1   p = 1
vx = 0.99  →
ρ = 0.1   p = 0.01
at rest
Uniform field B = 0.5 at 45°
Bx = By = 0.354,  Bz = vz = 0
Ideal gas, Γ = 5/3
Domain [−0.5, 0.5]², outflow, t = 0.4
Two shear layers (left, bottom): a 0.99c quadrant slides past the dense one. Two shocks (top, right): the fast quadrants run into the low-pressure one
ρ = 0.5   p = 1
at rest
ρ = 0.1   p = 1
vy = 0.99  ↑

### Notes:
The relativistic version of Lax and Liu's configuration (Del Zanna and Bucciantini 2002; also the CAFE code paper and Kiuchi et al. 2022), (rho, p, vx, vy): top left (0.1, 1, 0.99, 0), top right (0.1, 0.01, 0, 0), bottom left (0.5, 1, 0, 0), bottom right (0.1, 1, 0, 0.99). The field is ours (no published RMHD version found): B = 0.5 (cos 45°, sin 45°, 0), uniform, so div B = 0; at 45° Bx = By exactly, which keeps the problem's x-y mirror symmetry. Magnetic pressure 0.125 lies between the two gas pressures; the magnetisation runs from about 0.05 in the fast quadrants to about 2 in the top right (plasma beta down to 0.08 there). The moving quadrants have Lorentz factor 7.1. The run: 128² cells, piecewise-linear (MC) reconstruction, SSP-RK3, constrained transport, the exact flux with the network ft3, HLLD below the weak-jump gate (1e-2).

<!-- Slide number: 4 -->
# Exact flux vs HLLD at 128²
![L1 of the density at t = 0.4 against the 512² HLLD reference for HLLE, HLLC, HLLD and the exact flux at 64² and 128²](Picture 2)
−5.8%
density error against HLLD on the same grid
as good as HLLD at about 145², at 28× its wall time

### Notes:
Four-quadrant problem, ft3 network, t = 0.4. L1(rho) against the 512² HLLD run: exact 0.0703, HLLD 0.0747 at 128² (−5.8%); against the 512² HLLC and HLLE runs −8.1% and −7.6%. At 64²: −7.0%. HLLD converges at order 0.56 from 64 to 128, so the exact flux at 128² matches HLLD at about 145² (at 64² about 74²). Pressure: −2.2%. Faces answered exactly: 99.20% of 4.73M attempted (64²: 99.28%). Cost: 26.1 h against HLLD's 55 min. The 512² run is at 16%, done around Sun 15 Nov.

<!-- Slide number: 5 -->
# Where the exact flux helps
![Density error at t = 0.4 for HLLD and the exact flux at 128², and their difference: blue where the exact flux is closer to the reference](Picture 2)
Closer than HLLD in half the cells — gains in the low-density pocket where the shear layers meet, and along the shear layers

### Notes:
Right panel: |error of the exact run| minus |error of the HLLD run| per cell, against the 512² HLLD reference averaged to 128². Blue = exact closer. Closer in 8,262 of 16,384 cells; net −5.8% of HLLD's L1. Losses at the tip of the dense tongue and in the inner ring of the low-density pocket; the alternating blue-red rings are features that sit at slightly different places in the two runs. Dots: the 1,276 faces lost to HLLD at least once.

<!-- Slide number: 6 -->
# Retrained networks win on their own problem

| problem | production network | retrained on it |
| --- | --- | --- |
| Rotor | 94.2 / 78.4 | 96.6 / 85.5 |
| Orszag–Tang | 97.0 / 86.2 | 98.5 / 94.4 |
| Four-quadrant | 87.8 / 90.5 | 91.9 / 97.4 |

% of held-out faces solved with one Newton attempt (all faces / field-rotating faces). None transfers to the other problems → one network per equation of state.

### Notes:
Proper split: deduplicated faces, held-out time blocks with buffers, so test faces are never near-copies of training faces. Cross test: the rotor net drops to 20% on the four-quadrant problem's rotating faces; the OT net is trained at gamma 4/3, which the network does not see as an input. Proposal (open): one net per EOS, OT at 4/3, a joint rotor + 4Q fine-tune at 5/3. Since today the training files exclude answers with entropy-falling or Lax-violating slow shocks and no-root answers (1.9 / 2.8 / 0.6% of training rows for 4Q / rotor / OT). Calea 78178, 78179, 77245, 78249.

<!-- Slide number: 7 -->
# What fails
0.8%
of the attempted faces are lost to HLLD
37,940 of 4.71 million — every face traced to where and when it was lost

### Notes:
Every attempted face of the run was mapped back to its grid position and time from the harvest (rows come in face order within a sweep; the planar answers' unsolved twins anchor the alignment; 0.06% ambiguous); all counts equal the run's own totals.

<!-- Slide number: 8 -->
# What answers each face, step by step
![Attempted faces per step by the solver that answered them, and the shares of the rescues and losses over time](Picture 2)
seven-wave 90% · planar rescue 8.4% · retry 0.7% · four-unknown 0.5% · lost 0.8%

### Notes:
4,714,804 attempted faces with harvest rows (plus 12,446 three-wave degenerate answers). The planar share stays near 8% all run. Losses come in bursts at t ≈ 0.02-0.05 and 0.2-0.26 (5-7% of the attempted faces then), along the shear layers, and settle near 0.7% later.

<!-- Slide number: 9 -->
# Where the faces are lost
![Density of the four-quadrant problem at t = 0.4 with its structures named — the four starting quadrants, the two shear layers, the dense tongue and its tip, the low-density pocket where the shear layers meet — and every lost face after t = 0.1 coloured by type](Picture 2)
● A (52%)
along the two shear layers, and the edges of the dense tongue
● B, weak field (34%)
around the low-density pocket where the shear layers meet
● B, strong field (14%)
at the tip of the tongue, and where the shear layers end

### Notes:
A shear layer is where two regions slide past each other: the velocity along the layer jumps across it. Here the top-left quadrant moves right at 0.99c above the dense resting quadrant (the layer along y = 0, x < 0), and the bottom-right quadrant moves up at 0.99c beside it (along x = 0, y < 0). The two fast quadrants squeeze gas into the top-right one: the dense tongue along the diagonal. Where the two shear layers meet, just below-left of the centre, a low-density pocket forms. The pattern grows with time but keeps its shape, so every lost face after t = 0.1 is drawn where it would sit at t = 0.4. Counts: A 18,479, B weak 12,331, B strong 5,021 lost faces; type A mostly early (t < 0.1 and 0.2-0.3) on the shear layers.

<!-- Slide number: 10 -->
# What we tried on the lost faces
83%
of the lost faces have no root that any of our methods finds
tried: more seeds and retries · every planar branch · continuation · intermediate shocks · compound waves with a field flip

### Notes:
Failure ledger (calea 78407) on 872 faces still lost after replay: recovered by diverse seven-wave seeds 30, planar with every rotation branch 39, continuation and tube seed 18, the intermediate-shock branch 107, the compound slow wave (Brio-Wu element, calea 78446) 17; nothing found for 722 (83%), the same as the October 64² ledger. Type A: 79% nothing, intermediate shocks take 17%. Type B: 86% nothing. Controls: another root on at most 1.8% of solved faces.

<!-- Slide number: 11 -->
# Two situations explain every rescue and loss
A — two waves at the same speed
B — field straight through the face
![A fan of seven waves leaving the face; two of them, the slow and the Alfven wave, run on top of each other](Picture 4)
![The face as a vertical line and the magnetic field as an arrow almost perpendicular to it](Picture 8)
The slow and the Alfvén wave travel within 0.1% of each other
2.2M faces · 0.9% lost · 14% need the planar rescue
The field points almost straight through the face: hardly any tangential field
0.5M faces · 3.4% lost · 15% planar · 5% four-unknown
Everything else — 42% of all faces — is never lost

### Notes:
Both are places where two wave families nearly coincide. A: the slow and Alfven speeds of the two states within 1e-3 (it happens at high plasma beta and along the 0.99c shear layers), with the field at an angle to the face. B: the field nearly normal to the face (|Bn|/|B| above 0.9); with no tangential field the slow and Alfven speeds (weak field) or the fast and Alfven speeds (strong field) merge. Retries needed: A 0.6%, B 3.9%. The outcome is predictable from the two input states alone: a classifier trained on t < 0.3 and tested on t >= 0.3 separates planar rescues from first-try faces with AUC 0.90 and lost from rescued with 0.96; how close the slow and Alfven speeds are is the strongest feature.

<!-- Slide number: 12 -->
# Why the planar rescue is needed
![Left: the answer's slow wave against the Alfven speed for first-try and planar faces. Right: convergence of the seven-wave Newton from starts near the known answer](Picture 2)
A slow shock riding the Alfvén speed makes the seven-wave system singular — the formulation, not the seed

### Notes:
94% of planar answers have a slow wave within 1e-4 of the Alfven speed (first-try answers: 33%). At those answers the seven-wave Jacobian cannot be built on 52% (a 1e-6 probe leaves the constructible set) and on the rest its fifth singular value is 7.5e-7 of the first, at the Newton's 1e-6 cut; the planar 3x3 stays at 1.4e-2. The answers are genuine seven-wave roots (99.2%), yet from 3e-4 away only 35% converge, from the network's seed 3.7%. Opening the slow-shock window past the Alfven speed removes the construction failures but picks other roots on 12-50% of faces; its convergence test is still running.

<!-- Slide number: 13 -->
# Loss type A: two slow waves on the face
![Two lost faces on the 0.99c shear lines: their Riemann fans coloured by the tangential field, and the tangential field where the slow and Alfven waves sit](Picture 2)
Flow along the face at 0.987c
Slow wave, Alfvén wave and contact all within 0.02 of x/t = 0
Bt changes twice there: a compound wave (shock + fan), outside the seven-wave family

### Notes:
Each column: one lost face's own Riemann problem evolved as a 1D tube (HLLD + MC, 4096 / 2048 / 1024 cells). Top: the fan in the x-t plane coloured by B_t; bottom: B_t where the slow (blue) and Alfven (pink) speeds of the two states sit; they coincide to 4e-5. In a planar problem an Alfven wave can only flip B_t, not change its size, so the change of |B_t| at the Alfven speed is a second slow-type wave: a slow shock running at the Alfven speed with a slow fan behind it, converged in resolution. The existing compound element assumes a field reversal and cannot build it. Both faces: no method finds a root. HLLD's flux there is off 3e-4 to 6e-3 against the tube.

<!-- Slide number: 14 -->
# Loss type B: field nearly normal
![Two lost faces with the field nearly normal: their Riemann fans coloured by the tangential field, and the tangential field where the slow and Alfven waves sit](Picture 2)
Weak field: Bt reverses across one jump — an intermediate shock
Strong field: the slow wave moves pressure, not Bt — the variable the solver steers by

### Notes:
|Bn|/|B| above 0.99 in both. Left (weak field, beta 20; 34% of all losses): with B_t nearly zero the slow and Alfven speeds merge, and B_t flips sign across one compressive jump at that speed, an intermediate shock. Here both states have B_t = -0.022, yet inside the fan it goes -0.065, +0.088, -0.022: one reversal per side. Our intermediate-shock branch recovered only 2 of 449 type-B faces, so as built it does not reach these. Right (strong field, beta 0.33; 14% of all losses): the slow wave carries a 17% pressure jump but changes B_t only a little; this is the four-unknown solver's regime, and it fails here too.

<!-- Slide number: 15 -->
# What would fix it
A · 52%
a compound wave without a field flip: slow shock + attached fan
B weak · 34%
intermediate shocks on both sides of the face
B strong · 14%
slow waves steered by pressure on every face
And for speed: send the shear faces straight to the planar solver

### Notes:
Each is a new wave structure in the planar solver, not a better seed or a longer search. Type A is the largest share and local to the shear layers. Intermediate shocks fail the Lax admissibility test, so they would enter as a last resort before HLLD (keep-if-better-than-HLLD rule). The router: the outcome is predictable from the two states (AUC 0.90), and each planar face now burns 2-3 failed seven-wave attempts first. Running: the widened-window convergence test (78447), the 512² run (done ~15 Nov), GR06 Balsara 3 at 1600 cells (Fri-Sat).

<!-- Slide number: 16 -->
# Intermediate shocks, and the fix for type A
Intermediate shock
Type A: a compound wave, no flip
![Tangential field across an intermediate shock: positive on one side, negative on the other, a single jump through zero](Picture 4)
![Size of the tangential field across the type-A structure: a sharp drop at the slow shock, then a smooth rise through the attached slow fan, no change of sign](Picture 8)
The tangential field flips sign across one jump — possible only if the flow crosses the Alfvén speed inside the shock
It conserves everything, but ideal MHD discards it: the slightest out-of-plane twist of the field splits it into an Alfvén wave and a slow shock. Our flow is exactly planar, so nothing twists it and the scheme keeps it — that is what the weak-field B faces form
A slow shock moving exactly at the slow speed behind it, with a slow fan glued on — the field keeps its sign
Our compound element builds only the flipping kind (Brio–Wu: an intermediate shock + fan); type A needs the non-flipping kind — one new piece in the planar solver, still three unknowns
Test: build it, replay the 335 type-A faces nothing solves today, and check solved faces for second solutions

### Notes:
Why the flip needs an Alfven crossing: across any discontinuity (non-relativistic form, u = flow speed relative to the wave, a = normal Alfven speed) rho1 (u1² − a1²) Bt1 = rho2 (u2² − a2²) Bt2, so Bt changes sign only if the flow goes from faster than Alfven to slower across the jump. Fast shocks raise |Bt|, slow shocks lower it, both keep its sign; shocks that cross the Alfven speed (intermediate shocks) flip it. In ideal MHD they are not evolutionary (the wrong number of characteristics enter and leave), so the standard Riemann solution excludes them; with any dissipation they can persist, which is the long-known non-uniqueness of coplanar MHD Riemann problems (Brio and Wu 1988; Wu; Torrilhon 2003). Our solver allows slow shocks only slightly past the Alfven speed (window 0.03); a fully opened intermediate branch recovered 17% of type-A and about 0% of type-B losses in the ledger. Type A, as in the right sketch (example A1, read from the undisturbed side on the right): |Bt| drops 0.378 to 0.318 across a sharp jump at the slow-Alfven speed, then rises smoothly back to 0.365: a shock and a fan of the same (slow) family glued together, which happens when the shock moves at exactly the characteristic speed behind it (a sonic, or compound, wave). HLLD scripts/compound_wave.py already builds the reversing version (Brio-Wu: an intermediate shock with a fan attached; it solved 17 of 872 losses); the type-A element is the same construction with the sign kept, and keeps one free parameter (the contact's Bt), so the planar system stays [ln p_LF, Bt*, ln p_RF].
