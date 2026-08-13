# Deck probe: eight pinned arms from one checkpoint

## What this measures

A submission pilots one deck. The agent trains over a field of decks, so nothing in the
training curve says which single deck it plays best. This experiment answers that directly:
take one checkpoint, make eight copies of it, pin each copy to a different deck, train them
for a day, and compare the eval curves.

Everything except the pinned deck is held constant across the eight arms. They share the
warm start, the self-play league settings, the eval panel and the eval opponents, so a
difference between arms is a difference between decks.

## Deck selection

The candidates come from `decks/heuristic-resolved/manifest.json`, which holds 28,670 lists
with their tournament appearances scraped from Limitless. Each appearance carries an event
date, the player's match record and their placing.

Three steps, applied on 2026-08-13:

1. **Count appearances in the last 60 days, not all time.** The stored `observation_count`
   reaches back to April and rewards decks that have since left the format. Ogerpon Meganium
   Arboliva and Dragapult Dudunsparce rank 4th and 7th all-time inside `decks/top20` and have
   zero appearances in the 60-day window.
2. **Drop archetypes with fewer than 200 recorded games** in that window, which removes about
   120 tail archetypes whose win rate would be noise.
3. **Drop the losers.** Aggregate each archetype's records into wins over wins plus losses,
   and keep 0.494 and above.

Thirty-two archetypes clear step 2. Eight clear step 3.

| Arm | Archetype | Win rate | Entries | Share | Games | ex cards | List |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flareon | Flareon Noctowl | 0.557 | 61 | 0.7% | 287 | 11 | `flareon-noctowl/flareon-noctowl-13.csv` |
| basic-box | Basic Box | 0.521 | 312 | 3.5% | 1501 | 14 | `basic-box/basic-box-37.csv` |
| blaziken | Dragapult Blaziken | 0.518 | 360 | 4.0% | 1794 | 7 | `dragapult-blaziken/dragapult-blaziken-3.csv` |
| alakazam | Alakazam Dudunsparce | 0.511 | 693 | 7.7% | 3568 | 1 | `alakazam-dudunsparce/alakazam-dudunsparce-4.csv` |
| honchkrow | Rocket's Honchkrow | 0.508 | 710 | 7.9% | 3559 | 0 | `rockets-honchkrow/rockets-honchkrow-2.csv` |
| slowking | Slowking | 0.506 | 1227 | 13.6% | 5818 | 8 | `slowking/slowking.csv` |
| lopunny | Lopunny Dudunsparce | 0.499 | 162 | 1.8% | 732 | 5 | `lopunny-dudunsparce/lopunny-dudunsparce-2.csv` |
| ogerpon-hydrapple | Ogerpon Meganium Hydrapple | 0.495 | 245 | 2.7% | 1193 | 9 | `ogerpon-meganium-hydrapple/ogerpon-meganium-hydrapple-2.csv` |

Paths are relative to `decks/heuristic-resolved/`. Five of the eight sit outside
`decks/top20`, so the full corpus must be present on the machine that runs this.

Decks left out and why: Hop's Trevenant 0.445 over 1292 games, about four standard errors
below even; Dragapult Dusknoir 0.461, about three below; Lucario Hariyama 0.483, which the
current agent also beats 90% of the time when it appears in the eval panel; Raging Bolt
Ogerpon 0.490, which additionally carries 16 ex cards; Festival Lead 0.460.

Two notes on the table. Alakazam uses list `-4` although `-8` is marginally more played
(46 appearances against 40), because every existing eval is calibrated on `-4` and switching
would break comparison with earlier runs. The ex-card column is a tiebreak rather than a
filter: Crustle blanks ex attackers and sits at 94 entries in the same window, so Honchkrow
at zero and Alakazam at one (Fezandipiti ex, a support rather than an attacker) are the two
arms that matchup cannot wall.

## What is held constant

All eight arms load `conf/experiment/deck_probe_8way.yaml`, which inherits
`conf/experiment/deck_pinned_selfplay_local.yaml` and through it `conf/experiment/weighted_field.yaml`.

| Setting | Value | Why |
| --- | --- | --- |
| `train.pool_size` | 24 | League memory of 24M frames at 1M spacing. |
| `train.snapshot_interval` | 1000000 | Inherited from the weighted-field arm. |
| `train.pfsp_min_weight` | 0.15 | Stops `hard` weighting from forgetting beaten members. |
| `env.agent_deck_mirror` | true | Both seats pilot the pinned deck, so the league opponent stays competent. |
| `env.agent_deck_field_prob` | 0.0 | Every training episode is a mirror. |
| `deck_corpus` | top20 | The pool the eval panel is drawn from. Not used in training here, see below. |
| `env.eval_panel_size` | 10 | The same 10 opponent lists, round-robin, for every arm. |
| `train.eval_interval` | 500000 | 50 episodes per round. |
| `train.eval_opponents` | anchor, random, 10.14M MLP | The anchor interpolates `train.warmup_checkpoint`. |

The pinned deck does not need to belong to the sampling pool. `_pin_agent_deck` in
`src/training/env_factory.py:493` loads it from its own path and wraps the field spec, so an
arm can pilot a list from the full corpus while the panel stays on `top20`.

**Training never draws from the corpus in these arms.** `agent_deck_field_prob` is 0.0, and
`_sample_mirror` in `src/env/decks/agent_deck_sampler.py:171` rolls that probability first, so
the branch that would deal field decks never fires. Every training episode deals the pinned
deck to both seats and leaves the field sampler untouched, cursor and RNG included.

Evaluation is the opposite: `_pin_agent_deck` forces `field_probability` to 0.0 and `mirror` to
false on the eval split, so the agent keeps its pinned deck and the opponent draws from the
panel. `deck_corpus` therefore decides the eval panel and nothing else, which makes changing it
a change to the measurement rather than to what the arms learn.

`env.agent_deck` interpolates `${env.eval_agent_deck}`, so one override per arm pins both the
training seat and the eval seat.

## What varies

Per arm: `env.eval_agent_deck`, `wandb.name`, and the run directory. Separate run directories
are required rather than tidy. The PFSP league scans the run's own `checkpoints/`, so two arms
sharing a directory would recruit each other's snapshots and stop being independent.

## Resources

One RTX Pro 6000, not eight. The bottleneck is CPU-side collection: the model is 0.93M
parameters, the collector workers step the engine on CPU, and only each arm's main process
holds a CUDA context. One arm measured 7.5 GB of VRAM, so eight is about 60 GB of the card's
96 GB. What the arms compete for is cores.

```
--nodes=1 --ntasks=1
--gpus-per-node=rtx_pro_6000:1
--cpus-per-task=64
--mem=160G
--time=1-00:00:00
--signal=B:TERM@300
```

For reference, `slurm-conf/train_weighted_field.sh` runs a single arm at 32 cores and 32
workers and reaches 901 fps. This job takes twice the cores and splits them eight ways, so
each arm runs 8 workers at roughly 225 fps, about 19M frames per arm over 24 hours and about
155M frames of total experience.

Memory came from the running arm: 4.5 GB in the main process and 11.7 GB across 16 workers,
so about 10 GB per arm at 8 workers. The 160 GB request is generous and stays under the 192 GB
that one GPU's share of a 1536 GB node works out to.

`--signal=B:TERM@300` delivers SIGTERM five minutes before the wallclock. The batch script
forwards it to each arm's process group so every trainer writes `train_state.pt` and flushes
W&B before SIGKILL. `scripts/train_supervised.sh:201` reads exit code 143 as a deliberate stop
and does not restart into it.

## Frame budget

`collector.total_frames` is 100,000,000 on top of the warm start, which no arm will reach in a
day. This is deliberate. If an arm finished early it would hand its cores back to the others
and speed up whatever was still running, which would confound the comparison with finishing
order. Instead the wallclock stops every arm at roughly the same point.

The launcher reads the warm start's frame count once and adds the budget, so all arms get the
same absolute target. Compare arms at a common frame count afterwards, not at whatever each
one happened to reach.

## Running it

```bash
CHECKPOINT=/scratch/s4325621/pokemon-tcg-ai/outputs/weighted-field-20260808/\
tf-ptr-weighted-15m-s42/checkpoints/snapshot_000176357376.pt \
  sbatch slurm-conf/train_deck_probe_8way.sh
```

`CHECKPOINT` is required. `GROUP`, `NUM_WORKERS` and `BUDGET` are optional overrides.
`RESUME=1` continues from each arm's `train_state.pt` instead of warm-starting, which is what
a requeued job wants.

One arm on its own, on any machine:

```bash
DECK=decks/heuristic-resolved/slowking/slowking.csv \
RUN_NAME=pinned-slowking \
CHECKPOINT=/path/to/snapshot.pt \
TOTAL_FRAMES=276357376 \
  ./scripts/train_deck_probe_arm.sh 2>&1 | tee logs/pinned-slowking.log
```

Arm logs land in `logs/deck-probe/<group>-<arm>-<jobid>.log`. The Slurm job's own `.out` file
carries only the launcher's progress, so it stays readable.

### Before launching

`decks/heuristic-resolved` must be on scratch, since five of the eight lists sit outside
`decks/top20`. The third eval opponent,
`outputs/deck-pinned-20260808/mlp-ptr-pinned-15m-s42/checkpoints/snapshot_000010141696.pt`,
must be there too. It is the only reference shared with the existing runs, so it is what ties
this experiment back to them.

Check that file yourself before submitting. `trainer_builder.py:385` validates the warm start
at startup, but the eval opponent paths are not checked there, so a missing reference does not
surface until the first eval round 500,000 frames in.

`train.warmup_checkpoint` is not set in the config and must be passed. It is deliberately not
marked `???`: OmegaConf reads a missing value in a merged config as "no value supplied" and
keeps the inherited one, so the marker would silently anchor all eight arms to the 150M
snapshot from `weighted_field.yaml` instead of failing. Both launchers require it instead.

## Reading the results

Compare `eval/<anchor>/win_rate` across arms. It measures the finetuned policy piloting its
deck against the shared frozen ancestor piloting the 10-list panel, so across arms it reads as
"how well does this agent pilot this deck into the field".

Do not rank on a single round. Each round is 50 episodes and swings about 0.06. At ~19M frames
per arm the run produces roughly 38 rounds, so average the last 20, which brings the error
down to about 0.015. `eval/<anchor>/archetype_win_rate/*` gives the per-matchup breakdown, and
an arm with a good mean but one archetype at zero is walled by that matchup rather than
generally weak.

`train/win_rate` is a diagnostic, not a result. Under a mirror it should sit near 0.5. The
earlier fixed-deck arm without a mirror sat at 0.771 because the league opponent was dealt
field decks it never practised; the local mirror arm plateaued near 0.60. A value well above
0.5 means the learner is farming an opponent it handicapped itself.

## Known limitations

The eval panel is the 10 most-observed lists all time, so four of its slots (Lucario Hariyama,
Ogerpon Meganium, Dragapult Dudunsparce, Starmie Dusknoir) are decks with little or no play in
the last 60 days. Every arm is scored against the same panel, so the comparison between arms
is fair, but the absolute numbers describe a field that is about 40% out of date. Rebuilding
the panel from a recent window costs only comparability with earlier eval curves, since
`deck_corpus` does not touch training here.

The win rates that drove selection are humans against humans. Tournament statistics have
already mispredicted which deck this agent pilots well, which is the reason this experiment
exists rather than a deck being chosen from the table above.

No field matchups enter the gradient, because `agent_deck_field_prob` is 0.0. The gap between
a collected win rate near 0.5 and the eval curve on the 10-list panel is exactly the
generalisation this buys or loses. If a per-archetype breakdown starts falling on archetypes
the pinned deck never meets, raising the value to about 0.05 is the lever.
