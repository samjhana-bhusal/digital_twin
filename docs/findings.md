# Findings log

Observations the harness already reports. Each is a *measurement*, not yet a
conclusion — the weeks 3–5 study is what turns these into claims. Recorded here
with the command that produced them so nothing rests on memory.

---

## F1 · A clean twin at the plant's own parameters is exact

With a deterministic bus, the identification residual at the true parameter
vector is **exactly 0.0**, and 10% parameter error shows up at **2530σ**.

Why it matters: it means every nonzero residual later is a real effect rather
than harness noise. Without this check, an interface artefact and a physics error
are indistinguishable.

`tests/test_harness.py::test_loss_is_zero_at_true_parameters_and_deterministic`

---

## F2 · Tip friction is *exactly* unidentifiable, by construction

`mu_tip` has an eigenvalue of exactly zero. The cause is physical, not numerical:
MuJoCo combines contact friction as the elementwise **max** of the two geoms, so
while the fingertip's μ (0.45) sits below the box's (0.50) it has no effect at
all. Verified directly — sweeping `mu_tip` over 0.05, 0.45, 0.50 leaves the box
trajectory bit-identical, and only μ > 0.50 changes anything.

Why it matters: this is a structural unidentifiability created by the *contact
model's combination rule*, not by insufficient data. No amount of extra robot
time would fix it. It also means the parameter has a one-sided kink at
μ_tip = μ_box, so gradient-based identification will behave differently on either
side — worth reporting as a caution about identifying contact parameters
generally.

`tests/test_harness.py::test_tip_friction_is_structurally_unidentifiable`

---

## F3 · Most of a realistic twin's parameter error is unidentifiable

With 40 push + 10 free episodes, whitened by realistic sensor noise:

- effective rank **6 of 18** (at `flat_rel_tol = 1e-3`)
- exact null directions: **1**
- stiff condition number: **8.9e2**
- over 32 twins mis-specified at 25% per parameter, the fraction of parameter
  error lying in unidentifiable directions is **0.783 ± 0.119**

The flattest directions are dominated by object and contact parameters
(`f_bt`, `f_bx`, `f_by`, `m_box`, `mu_box`), while arm parameters are comparatively
well constrained. Physically sensible: contact is intermittent, so it is observed
for only part of each episode.

**Caveat to resolve in week 4.** Effective rank is sensitive to both
`flat_rel_tol` and the assumed sensor noise — whitening with the artificially
tiny noise of a perfect bus gave rank 9/18 instead of 6/18. The reported rank
must therefore come with a sensitivity sweep over both, or it is an artefact of
two arbitrary constants.

`experiments/r1_spectrum.py`

---

## F4 · An unmodelled servo bus dominates everything else

With a realistic bus (12 ms latency, 4 ms jitter, 1% dropout, 12-bit encoder)
the identification residual at the **true** parameters sits at **7.9×** sensor
noise.

Why it matters: before any parameter is mis-fitted, the interface alone puts the
achievable fit floor almost an order of magnitude above the sensors. Any
identification run on this plant that reports convergence without modelling the
bus has absorbed latency into a physical parameter. Which parameter, and by how
much, is the R-bus measurement for weeks 5–6.

---

## F5 · Identifiability is relative to the dataset, and a relative threshold mislabels it

**This supersedes F3's headline number.** F3 reported "effective rank 6 of 18"
from a mixed dataset. That figure is an artefact of how the cutoff is defined.

Measured two ways, same model, same noise, same perturbation magnitude
(0.45 in log space, a 57% parameter error):

| identification set | effective rank | flat dim | divergence from a twin wrong in a *stiff* direction | ... in a *flat* direction |
|---|---|---|---|---|
| 30 push episodes | 17 / 18 | 1 | 6.58 (8.77 mm box error) | **0.000 (0.00 mm)** |
| 30 push + 8 free chirps | 9 / 18 | 9 | 1.21 | 5.00 |

**Push-only behaves exactly as the theory says.** There is one unidentifiable
direction, it is a true null direction (`mu_tip`, per F2), and a twin wrong by
57% along it is *provably* indistinguishable from the plant — divergence is
identically zero, not merely small.

**Mixed data inverts the classification.** Contact-free chirps identify the arm
parameters so sharply that the largest eigenvalue grows by orders of magnitude.
Every contact parameter then falls below a *relative* cutoff and gets labelled
unidentifiable, despite being well constrained by the push episodes. A twin
perturbed along those "flat" directions then diverges **4× more** than one
perturbed along the "stiff" directions — and the inversion is already present at
a perturbation of 0.01, so this is not merely a breakdown of the local
linearisation:

| perturbation magnitude | stiff | flat | ratio |
|---|---|---|---|
| 0.01 | 0.199 | 0.261 | 0.76× |
| 0.03 | 0.209 | 0.444 | 0.47× |
| 0.10 | 0.344 | 1.267 | 0.27× |
| 0.45 | 1.212 | 5.004 | 0.24× |

**What this means, and it sharpens R1 rather than refuting it.** Identifiability
is a property of (model, dataset, noise model) jointly. A direction that is flat
with respect to an identification loss can be the direction a downstream task is
most sensitive to, because the loss is averaged over whatever episodes happen to
be in the dataset while the task is not. Adding more data of the *wrong kind*
makes this worse rather than better.

Two consequences for the plan:

1. **Replace the relative cutoff with an absolute, noise-referenced one** — keep
   a direction if a unit parameter change along it moves the residual by more
   than 1σ. `flat_rel_tol` in `configs/default.yaml` should not survive week 4.
2. **The stiff/flat decomposition must be computed against a task-weighted
   metric**, not against whatever episodes were convenient to collect. That
   comparison is now the centre of the week 5 experiment.

`experiments/make_visuals.py`

---

## F6 · A benchmark at ceiling measures nothing, and clutter was the wrong dial

The first version of the S2 benchmark scored **150/150**. That is not a good
result, it is a broken instrument: it cannot show that a later stage helped or
hurt.

The obvious fix was to pack the objects closer together. **It did almost
nothing.** Sweeping the minimum object separation while holding everything else
fixed:

| minimum separation | clean success |
|---|---|
| 85 mm | 95.7% |
| 62 mm | 97.1% |
| 56 mm | 92.9% |
| 50 mm | 95.7% |

A top-down grasp with thin parallel fingers tolerates close neighbours, so
geometric clutter was never the binding constraint. Difficulty had to come from
**task structure** instead, and two changes produced it:

1. **Occupied destinations.** "Put the red cube where the blue cube is" cannot
   be completed by a single pick-and-place — the blocker has to move first.
   That class sits at **48–58%**.
2. **Scoring collateral damage.** The naive planner often satisfies an
   occupied-destination goal anyway, by dropping its cube on the spot and
   shoving the blocker out of the way. Physically that is a real strategy; for a
   field robot it is making a mess. Requiring that no unlicensed object moves
   more than 30 mm is what separates doing the job from bulldozing through it.

The resulting benchmark reads 82.7% (open) and 75.3% (cluttered), with the
failures concentrated in the class the next stage is designed to fix.

**The incidental finding is the more useful one.** In roughly **9%** of
episodes the skill reported success while the task had actually failed. The
skill is not lying — it grasped, carried and released, and from inside the skill
that is indistinguishable from success. Only a goal predicate or an external
verifier can tell the difference. That is the argument for the S4
verification loop, now measured rather than assumed.

`experiments/bench_tasks.py`

---

## F7 · At this sample size, the benchmark's headline number moves 7 points on the seed alone

Building S4 meant running the S2 pipeline again as a control arm, and it came
back **90.0%** where S2 had reported **82.7%**. The pipeline had changed in
three ways, so each was reverted in turn and re-measured on S2's own seeding:

| | clean success | occupied-spot |
|---|---|---|
| S2 as published | 82.7% | 58.2% |
| + settle 20 → 160 control steps | 83.3% | 58.2% |
| + park the arm for a camera look, then return to ready | 84.7% | 60.0% |
| + the planner's placement sampler instead of the benchmark's | 82.7% | 58.2% |

None of them. The difference was the **seeding scheme**. S2 drew every layout
from one continuing random stream; S4 seeds each layout independently so that
different arms get identical worlds. Running *S2's unmodified logic* under S4's
seeding gives **89.5%** clean and **68.7%** on occupied-spot — the whole gap,
from nothing but the draw.

So at n = 150–200, an absolute success rate on this benchmark carries roughly
±7 points of seed noise. That is larger than most of the effects the remaining
stages are trying to measure, and it means **S2's 82.7% and 75.3% should be read
as "about 85%" and "about 78%", not as calibrated constants.**

Two changes follow, and both are now in `experiments/bench_loop.py`:

1. **Report across several seeds**, with the per-seed spread printed next to the
   pooled number, so the reader can see the noise instead of being asked to
   trust the mean.
2. **Compare arms paired, never rate against rate.** Every arm runs the same
   layouts and the same instructions, so each episode is a matched pair and the
   comparison reduces to the episodes where the two arms disagreed — reported
   as *rescued* / *broke* counts with McNemar's exact test. A paired difference
   survives the seed noise that swamps a difference of two independent rates.

The general lesson is the same one as F6, arriving from the other side: a number
is not a result until you know what it does when you change something that
should not matter.

`experiments/bench_loop.py`

---

## F8 · Simulating the plan first works — and a *more accurate* twin made it worse

Three arms, four seeds, 300 episodes each per regime, every arm on identical
layouts and identical instructions so the comparison is paired.

| | open | cluttered |
|---|---|---|
| direct (one pick-and-place, no simulation) | 88.3% | 79.3% |
| **loop** (simulate, repair, execute) | **96.7%** | **90.3%** |
| loop with a *perfect* twin | 95.7% | 85.3% |

Paired, episode by episode, against the direct arm:

| | rescued | broke | McNemar |
|---|---|---|---|
| loop, open | 25 | 0 | p = 6 × 10⁻⁸ |
| loop, cluttered | 34 | 1 | p = 2 × 10⁻⁹ |

The lift lands exactly where S2 said the headroom was. Occupied-destination
tasks — *"put the red cube where the blue cube is"* — go from **66.7% to 90.5%**
(open) and **55.6% to 80.6%** (cluttered). The repair responsible is the only
one that changes the length of the plan: move the blocker, then place. It was
chosen in 61/300 and 59/300 episodes and succeeded in 100% and 94.9% of them.

It is cheap. The loop costs a mean of **1.2–1.3 simulated rollouts per episode**
and roughly doubles wall-clock time, 0.26 s → 0.52 s (open) and 0.35 s → 0.71 s
(cluttered). Simulating before acting is affordable *because* an episode is
cheap, which is the whole reason the project is built on a fast simulator.

### The inversion

`TwinFidelity.perfect()` — since renamed `exact_physics()` — was meant to be the
easy ablation: give the twin the plant's true mass, friction and damping and the
loop should do at least as well. **It did worse**, and on
the cluttered regime significantly so — paired against the ordinary loop it
rescued 1 episode and broke 16 (p = 0.0003).

The evidence for why is in what the two arms chose to do:

| cluttered | direct plan chosen | of those, succeeded | clear-then-place chosen |
|---|---|---|---|
| loop (mis-specified twin) | 236 | 89.8% | 59 |
| loop-perfect | 246 | 84.6% | 51 |

The accurate twin endorses the one-step plan more often, and the one-step plans
it endorses are worse. Dropping a cube onto an occupied spot is a contact-rich,
chaotic manoeuvre: whether it lands or skitters off turns on millimetres. A twin
that reproduces the plant almost exactly will faithfully simulate one particular
roll of that die and report success. A twin that is slightly wrong rolls a
*different* die, so a plan only survives it if the plan does not depend on which
die was rolled — and it falls through to the two-step plan, which does not
depend on it at all.

So the twin's job is not to be accurate. Its job is to **reject plans whose
outcome is sensitive**, and a deliberately imperfect twin is better at that than
an accurate one. This is domain randomisation with a sample size of one.

Caveats, because this is one point and not a curve:

1. ~~`perfect()` changes two things at once — the physics parameters *and*
   whether the twin knows object orientation. This experiment cannot say which
   did it.~~ **Wrong, and the correction matters.** Building S5 showed that
   `known_yaw` was never passed from `choose` into `Twin.sync`: the flag was
   inert, so `perfect()` differed from the default in physics parameters *only*
   and there was no confound. The inversion is attributable to the physics
   mis-specification alone, which makes the result cleaner than it was claimed
   to be. The flag is now plumbed through, with a regression test
   (`tests/test_fidelity.py`) whose standing rule is that **every knob must be
   observable in the twin's state** — setting it has to change something you can
   measure. `perfect()` is also renamed `exact_physics()`, because the twin's
   object *positions* come from the camera under every setting and no
   configuration here has ever been perfect.
2. The direction could be specific to this mis-specification (heavier, more
   slippery cubes bias the twin towards predicting failure). A mis-specification
   biased the other way might simply make the loop worse.
3. One mis-specification magnitude, not a sweep.

Caveats 2 and 3 are what S5 measures, with a sharper question than the stage
started with: not *how accurate must the twin be*, but *is accuracy even the
right axis*.

### The verifier

Across all 1,800 episodes the camera verifier made **1,599 'success' calls and
not one of them was wrong**. Its only errors run the other way: 8 episodes it
called failures that were in fact clean, where a ~3 mm position estimate fell on
the wrong side of a tolerance.

Set against the skill's own self-report on the same episodes:

| | claimed success wrongly |
|---|---|
| the skill, from inside the action | 94 / 1,693 of its success calls |
| the camera, from one photograph | **0 / 1,599** |

That is the S2 finding paid off. The skill cannot see its own failures, because
grasping, carrying and releasing is all it knows about; 94 times it did all
three and the task was still not done. Every one of those was visible in a
single overhead image. The verification half of the loop is not redundancy — it
is the only part of the system that can tell.

`experiments/bench_loop.py`, `experiments/demo_loop.py`

---

## F9 · The twin's parameter error was standing in for the state it could not see

S4 said "a more accurate twin made the loop worse" and read that as *accuracy is
the wrong axis*. Crossing the twin's two kinds of ignorance — wrong physics, and
blindness to which way a cube is turned — shows that reading was too broad. The
2×2, cluttered regime, 300 episodes per cell, every cell on identical layouts
and instructions:

| clean success | yaw unseen | yaw known |
|---|---|---|
| **exact physics** | **85.3%** | 90.7% |
| **mis-specified physics** | 90.3% | 90.3% |

(no twin at all: 79.3%)

The two ignorances are **substitutes, not additive**. Fixing either one lifts
the worst cell by about five points; fixing both lifts it by the same five
points and no more. Paired against the worst cell, exact/yaw-known rescued 23
and broke 7 (p = 0.005), mis-specified/yaw-unseen rescued 16 and broke 1
(p = 0.0003), and both-fixed rescued 25 and broke 10 (p = 0.017).

So the S4 ablation did not discover that accuracy is bad. It discovered the
**one cell where accuracy is spent on the wrong thing**: a twin that is exactly
right about mass, friction and damping while being wrong about the *scene*
simulates a grasp on a square-on cube, faithfully, and reports that it works.
Every cube in these layouts is turned by up to 23°. The parameter error in the
ordinary twin was not insight — it was noise that happened to break that false
confidence, and it is worth exactly as much as actually seeing the orientation,
and no more.

The twin's own prediction accuracy ranks the cells the same way, and the gap is
where you would want it — in the errors that matter:

| | twin agreed with reality | said clean, was not |
|---|---|---|
| exact physics, yaw unseen | 86.7% | **12.7%** |
| mis-specified, yaw unseen | 91.3% | 7.3% |
| exact physics, yaw known | 93.0% | 6.3% |
| mis-specified, yaw known | 94.0% | 5.7% |

**A twin with deliberately wrong physics and correct geometry predicts better
than one with correct physics and wrong geometry** — 94.0% against 86.7%. For
this task, knowing *where things are and how they sit* is worth more than
knowing what they are made of.

The share of episodes choosing the two-step plan barely moves across the four
cells (17.0–19.7%), which narrows the mechanism S4 proposed. The plan *mix* is
not what separates the arms; what separates them is **which one-step plans get
endorsed**. The blind-but-exact twin endorses more of the marginal ones.

One consequence for the project, and it is not the one this stage expected:
effort spent identifying the twin's physics is worth less than effort spent on
the perception feeding it. That points S3 at something sharper than "ground the
instruction" — a VLM that can report a cube's orientation would be repairing the
twin's dominant error, not just reading the instruction.

`experiments/sweep_fidelity.py factorial`

---

## F10 · The true parameters are the worst parameters, and accuracy alone is not the point

The mis-specification is a signed magnitude along a direction in log-parameter
space, so the S4 default sits at +1 and −1 is its exact mirror: where the
default twin believes cubes are heavier and more slippery than they are, the
mirror believes they are lighter and grippier. Nine points, cluttered regime,
300 episodes each, all paired.

Two quantities, measured in two different runs on purpose:

* **useful** — clean success when the twin is free to choose and repair its plan.
* **accurate** — how often the twin's verdict was right about a plan *it did not
  choose*. Every twin was asked about the same fixed one-step plan, which the
  plant then executed. Without this separation the accuracy number is
  endogenous: a cautious twin picks easy-to-predict plans and scores well for it,
  which is a statement about plan selection wearing the clothes of a statement
  about fidelity.

| mis-specification | −3 | −2 | −1 | −0.5 | **0** | +0.5 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|---|---|
| task succeeded | 84.3 | 86.7 | 88.3 | 87.3 | **85.3** | 87.0 | **90.3** | 89.0 | 90.0 |
| verdict was right | 83.3 | 84.0 | 81.0 | 79.0 | **78.0** | 77.3 | 84.0 | **84.7** | 81.3 |
| said clean, was not | 13.0 | 10.3 | 9.0 | 10.7 | **12.0** | 10.3 | **7.0** | 7.7 | 7.3 |
| predicted clean | 88.7 | 84.0 | 78.3 | 79.7 | 81.3 | 77.3 | 77.3 | **79.3** | 75.3 |

(the fixed plan actually succeeded 79.3% of the time)

**The twin holding the plant's true parameters is the minimum of both curves.**
It is neither the most useful (85.3%, against 90.3% at +1) nor the most accurate
(78.0%, against 84.7% at +2). Paired against it, +1 rescued 16 episodes and
broke 1 (p = 0.0003); +2 and +3 are also significant (p = 0.04, p = 0.02). On
the mirror side only −1 approaches it (+9 net, p = 0.064).

Taken with F9 this is not mysterious. The twin's dominant error is not its
physics but the cube orientation it cannot see, which makes it *optimistic*
about grasps. Pessimistic physics error cancels part of that optimism. Exact
parameters leave the optimism uncorrected, which is why zero is a minimum and
not a maximum, and why the gains saturate around 90% — the same place that
simply telling the twin the true orientation gets to (F9: 90.7%).

### Accuracy that comes from agreeing with the common case is worthless

The most useful comparison in the table is **−3 against +1**. Both score about
the same on accuracy (83.3% and 84.0%). Their task success is 84.3% and 90.3%,
the worst and the best arms in the sweep.

The reason is in the bottom two rows. The optimistic twin at −3 predicts a clean
outcome **88.7%** of the time when the truth is 79.3%: it is right often because
most plans do work, not because it can tell which ones will not. Its false
confidence is the highest in the sweep, 13.0%, and false confidence is the error
you *act on*. The pessimistic twin at +1 is right about as often overall while
getting the failures right — 7.0% false confidence, nearly half.

So the slogan from S4 — "the twin's job is not accuracy" — survives, but needs
its second half: **the twin's job is to be right about the failures.** A twin
scored on overall agreement can buy a good number by saying yes to everything,
and that twin is the worst one to plan with.

`experiments/sweep_fidelity.py sweep|fixedplan`

---

## F11 · The parameter the data pins down is not the parameter the task cares about

F5 made this claim on the 3-DOF planar rig. Here it is on the system that
matters, over the three parameters the twin actually has, in one set of units.

**Identification.** Push a cube with a scripted motion, log what the camera and
the joint encoders would see, and finite-difference the residual against
log-parameters. The residual is exactly 0.000σ at the true parameters, which is
the check that the harness is sound. A push rather than a pick, deliberately: a
grasp is a discrete event whose outcome flips with parameters, the residual
jumps, and a finite difference across that jump measures the jump.

**Task.** How far clean success moves per unit of log-parameter error along each
axis, from the ±1 runs.

| | identification (σ per unit log error) | task (points of success per unit log error) |
|---|---|---|
| cube friction | **174** | 6.8 |
| cube mass | 65.6 | **9.1** |
| joint damping | 29.5 | **0.0** |

**The top two are swapped.** The push data constrains friction nearly three
times better than mass; the task is half again as sensitive to mass as to
friction. As unit vectors the two sensitivities sit **33.6° apart** (cosine
0.83) — not orthogonal, but far enough that optimising one is not optimising the
other, and the *ordering* a practitioner would act on is reversed.

Joint damping is the sharper case. It is comfortably identifiable — 29.5σ per
unit log error, an eigenvalue well clear of zero, nothing like the exact null
direction F4 found. And it is worth **nothing**: mis-specifying it by ±1 gives
85.5% either way, against a baseline of 85.3%, to three significant figures in
every column. A perfectly well-posed parameter that the task cannot feel.

The practical consequence for this project is a reordering of effort. Spending
the identification budget where the data is most informative would spend it on
friction and damping. What would actually move the system is mass — and, well
ahead of any of them, the cube orientation the camera never reports (F9), which
is not a physics parameter at all.

`experiments/identify_task.py`, `experiments/sweep_fidelity.py directions`

---

## F12 · The loop's endorsement flag is a usable abstention signal, and it was free

The loop executes a plan even when the twin never endorsed one inside its
budget, flagging it `confident=False`. That flag was built as an output worth
having rather than as something to measure; re-reading the S4 episode logs for
the writeup shows it separates two very different populations.

Pooled over both loop arms and both regimes (1,200 episodes):

| | n | clean success |
|---|---|---|
| the twin endorsed the plan | 1,181 | **93.0%** (91.4–94.3) |
| the twin never endorsed one | 19 | **31.6%** (15.4–54.0) |

Coverage is 98.4%. The Wilson intervals do not overlap, so the ordering is
solid; the flagged population is 19 episodes and its interval is wide, so the
*ordering* is the claim and the point estimate is not.

Two things make it more than a curiosity. It **never fires at all in the open
regime** — 0/600 — and fires 19 times in clutter, which is consistent with it
tracking genuine difficulty rather than firing at random. And it costs nothing:
it is a by-product of the budget the loop already spends, not an extra
computation. A system that can say *I could not check this one* before acting
can ask for help; one that cannot tell the difference cannot.

Worth stating plainly what it is not. It is a hard endorsement, not a
probability, so there is one operating point and no risk–coverage curve to
trade along. Getting one would mean having the twin return a margin rather than
a boolean, which is a change to `Twin.predict` and not an analysis.

`paper/build_tables.py` (Table 6), from `results/s4_loop.json`

---

## F13 · The collateral rule never binds in S4, and the headline survives tightening it

An embarrassing one found while writing up, and the reason to write up.

Clean success requires the goal predicate **and** that no unlicensed object
moved more than 30 mm. F6 introduced that clause because the naive planner was
satisfying occupied-destination goals by bulldozing the blocker aside. Across
all 1,800 S4 episodes:

- episodes that met their goal and failed on displacement: **0**
- largest displacement observed anywhere: **29.4 mm**

So the constant that does design work in this project — it is the reason the
repair catalogue contains *turn the wrist* instead of *shove the neighbour* —
is never exercised by the experiment that reports it, and the largest value
seen sits just under the line. That is a limitation, not a result.

Re-scoring every episode at tighter thresholds (goal predicate is independent
of τ, so this is exact, not a re-run):

| τ | direct, open | loop, open | direct, cluttered | loop, cluttered | paired, cluttered |
|---|---|---|---|---|---|
| 10 mm | 88.0 | 96.3 | 76.7 | 86.7 | +31/−1, p = 2×10⁻⁸ |
| 15 mm | 88.3 | 96.7 | 77.3 | 87.7 | +32/−1, p = 8×10⁻⁹ |
| 20 mm | 88.3 | 96.7 | 78.0 | 88.7 | +33/−1, p = 4×10⁻⁹ |
| 30 mm | 88.3 | 96.7 | 79.3 | 90.3 | +34/−1, p = 2×10⁻⁹ |
| 50 mm | 88.3 | 96.7 | 79.3 | 90.3 | +34/−1, p = 2×10⁻⁹ |

The headline conclusion does not depend on where the line is drawn — tightening
to 10 mm costs both cluttered arms about three points and leaves the paired
result intact. What remains untested is the rule's **deterrent** function, since
in these runs no arm attempted the strategy it exists to forbid.

The general lesson is F6's and F7's again from a third side: a constant nobody
has watched fire is a constant nobody has tested.

`paper/build_tables.py` (Table 7), from `results/s4_loop.json`
