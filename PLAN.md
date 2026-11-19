# fieldtwin — plan

**The system.** A language-commanded tabletop manipulator that simulates its
plan before acting and verifies the outcome afterwards.

**The claim worth earning.** A manipulation system that checks its own plan in a
digital twin before committing, and checks its own result with a VLM afterwards,
fails *detectably* rather than silently — and the disagreements between those two
checks are a usable signal, not noise.

**Constraints.** One person alongside a full-time job, roughly 12–15 h/week.
Apple Silicon only: MuJoCo on CPU, MLX for the VLM, no GPU cluster. Simulation
only — no hardware, no CAD. These are limits to state, not to hide.

---

## Stages

### S1 · Arm, scene, skills — **done**

Panda from Menagerie composed with a tabletop scene through `MjSpec`; IK through
mink; pick, place and push written here, each with an explicit success
condition; randomised scene generator; 24 tests.

Baseline: **59/60** on randomised layouts, 9.8 mm mean placement error, 0.16 s per episode (p95 0.17 s).

### S2 · Tasks and a benchmark — **done**

A task is a **reference** (which object) plus a **goal** (a predicate on
simulator state). The generator refuses two kinds of bad instruction: ambiguous
references (a reference must beat its runner-up by 55 mm) and goals that are
already true at the start. Roughly 53 well-posed instructions per layout, across
five reference types and five goal types.

Scoring is *clean success*: the goal predicate holds **and** nothing the
instruction did not license moving was shifted by more than 30 mm.

| | open | cluttered |
|---|---|---|
| clean success | 82.7% | 75.3% |
| `where Y is` (occupied destination) | 58.2% | 48.3% |
| skill claimed success, task had failed | 14/150 | 13/150 |

Read those two rates as "about 85%" and "about 78%": building S4 showed that at
this sample size the absolute number carries roughly ±7 points of seed noise
([docs/findings.md](docs/findings.md) F7). The per-class ordering is what held
up, and it is what mattered.

Two things this measurement established, both of which change later stages:

* **Packing objects closer barely matters.** Sweeping minimum separation from
  85 mm to 50 mm moved success only between 93% and 97%: a top-down grasp with
  thin fingers tolerates close neighbours. Difficulty had to come from *task
  structure*, not from geometry.
* **The hard class is the occupied destination.** "Put the red cube where the
  blue cube is" cannot be done in one pick-and-place. The naive planner often
  gets the predicate true anyway by bulldozing the blocker aside — which is why
  collateral damage is scored, and why that class sits near 50%.

### S3 · Grounding — instruction to object

Render the `front` and `wrist` cameras, hand the image and the instruction to a
small open VLM under MLX, get back the referred object.

S5 added a second, better-motivated job for it. The twin's dominant error is not
any physics parameter but **the cube orientation the camera never reports**
(F9, F11) — worth more than every parameter in the model put together. A VLM
that can say which way a cube is turned would be repairing that error, and that
is a sharper reason to want one than reading the instruction.

Measure it honestly: grounding accuracy per reference type, against the
simulator's ground truth. Spatial references will be much worse than colour
references; that gap is the result. Include a geometric baseline that uses no
VLM at all — if a colour threshold matches the VLM on colour references, say so
plainly.

### S4 · The loop — simulate before acting, verify after — **done**

The centrepiece, and cheap because an episode costs a fraction of a second.
Both halves are built and measured:

1. Look at the table through the overhead camera (`perceive.py`).
2. Build a plan (`plan.py`) and **roll it out in the twin** (`twin.py`).
3. If the twin says it fails, repair it — clear the blocker, re-aim, turn the
   wrist 90°, push instead of pick — and re-simulate, up to a budget.
4. Execute. A plan that never earned an endorsement is still executed, but
   flagged `confident=False`, which is the output that lets a system ask for
   help instead of failing silently.
5. Verify from a second photograph (`verify.py`) and log the 2×2s.

| | open | cluttered |
|---|---|---|
| direct (S2 behaviour) | 88.3% | 79.3% |
| **the loop** | **96.7%** | **90.3%** |
| occupied-destination tasks | 66.7% → **90.5%** | 55.6% → **80.6%** |

Paired per episode: +25/−0 on open, +34/−1 on cluttered. Both bars S2 set were
cleared — the occupied-destination class, and the ~9% of episodes where the
skill claimed a success that had not happened, every one of which the camera
verifier caught (0 wrong success calls in 1,599).

Two results came out of it that were not planned for, both in
[docs/findings.md](docs/findings.md): the benchmark's seed sensitivity (F7), and
a *more accurate twin performing worse* (F8).

One repair is deliberately absent. Shoving a neighbour aside to make room for
the fingers fixes the grasp and fails the episode on collateral damage, so the
scoring rule — not taste — rules it out and leaves turning the wrist instead.

### S5 · How good does the twin have to be? — **done**

The stage's question turned out to be the wrong one, and answering the right one
took four experiments. All cluttered regime, 300 episodes per cell, paired.

**The factorial.** The twin's two kinds of ignorance — wrong physics, and
blindness to cube orientation — are *substitutes*, not additive. Fixing either
lifts the worst cell about five points; fixing both lifts it the same five
points and no more. The worst cell is exact physics with unseen orientation:
accuracy spent faithfully simulating a scene that is not there (F9).

**The sweep.** Mis-specification as a signed magnitude, so −1 mirrors +1.
**The twin with the plant's true parameters is the minimum of both curves** —
neither the most useful (85.3% against 90.3%) nor the most accurate (78.0%
against 84.7%). Pessimistic error cancels the optimism the unseen orientation
causes, and saturates at about what knowing the orientation is worth (F10).

**Accuracy measured honestly.** Scored on a plan it chose, a cautious twin looks
accurate because it picks predictable plans. Scored on a *fixed* plan, the
optimistic twin at −3 is as accurate as the one at +1 and the worst arm to plan
with: it is right by agreeing with the common case, and carries the sweep's
highest false confidence. The twin's job is to be right about the **failures**.

**Identification against task sensitivity.** F5's claim, on the real system:
push data constrains friction ≈3× better than mass, while the task is half again
more sensitive to mass than friction — the top two swapped, the two sensitivity
vectors 33.6° apart. Joint damping is cleanly identifiable and worth exactly
nothing to the task (F11).

What this changes downstream: effort on identifying the twin's physics is worth
less than effort on the perception feeding it. The twin's dominant error is a
state it cannot see, not a parameter it has wrong.

### S6 · Evaluation and writeup — **done**

The report is [`paper/paper.pdf`](paper/paper.pdf): 20 pages, 45 references, a
literature review placing the twin-fidelity result against domain
randomisation, objective mismatch and classical experiment design, and a
reproduction command per figure and table.

Two things came out of writing it rather than out of running anything new.
Re-reading the S4 episode logs showed the loop's `confident=False` flag is a
usable abstention signal — 31.6% clean when it fires against 93.0% when it does
not, at 98.4% coverage (F12). And the collateral threshold that does design work
throughout this project turns out never to bind in the experiment that reports
it; the headline survives tightening it to 10 mm, but the rule is untested and
is now written down as such (F13).

The build is set up so the document cannot drift from its data: every number in
the text is a macro generated by `paper/build_tables.py` from `results/*.json`,
so a stale figure is a build error rather than a reader's problem.

Still open from this stage: a failure taxonomy by *reference type* rather than
by cause, and the no-verifier ablation (the verifier is currently always on).

---

## Order of value

If time runs out, this is the order things get cut from the bottom.

| | |
|---|---|
| **Done** | S1, S2, S4 (the loop), S5 (the twin-fidelity question), S6 (the report) |
| **Next** | S3 — and F9/F11 gave it a sharper target than grounding alone |
| **Then** | the two gaps S6 left: a per-reference-type failure taxonomy, and the no-verifier ablation |

S4 was deliberately put above S3, and that paid: *the loop is the
contribution*, and it is now demonstrated end to end without a VLM anywhere in
it. S3 slots a VLM into two sockets that already exist and already have a
measured geometric baseline to beat — grounding the reference, and verifying the
outcome — rather than being a prerequisite for anything.

## Standing rules

1. Every skill reports *why* it failed, never just that it did.
2. Success is a predicate on simulator state, never a human judgement.
3. Libraries do the solved parts. Write only what is actually the contribution.
4. Nothing is a result until its reproduction script runs from a clean checkout.
5. The simulation-only limitation goes in the abstract, not just the discussion.

## Out of scope

Hardware, CAD, real-robot transfer, VLA fine-tuning, reinforcement learning from
scratch, Isaac Sim (does not run on macOS).
