# Gravity conditioning and rollout training in SG-JEPA

This experiment separates two parts of Semigroup-JEPA (SG-JEPA):

1. Supplying the episode's physical parameter, gravity, as an action input.
2. Training the predictor on its own five-step latent rollouts.

The result supports a narrow conclusion: in this setup, the gravity input did
not help by itself. Its benefit appeared when rollout training gave the model a
reason to use it.

## Question

Does SG-JEPA improve because it receives gravity, because it trains on
multi-step predictions, or because the two changes work together?

## Factorial design

All four cells use the same ViT-Tiny encoder, GRU predictor, optimizer, data
split, 20-epoch schedule, and Square evaluation cohort. Each cell has one
world-model training seed, `42`. Each checkpoint has five probe and evaluation
seeds, `42` through `46`.

| Cell | Gravity input | Training objective |
| --- | --- | --- |
| `correct_onestep` | True normalized episode gravity | One-step prediction |
| `constant_onestep` | Normalized training mean, `0` | One-step prediction |
| `correct_rollout` | True normalized episode gravity | Five-step autoregressive rollout |
| `constant_rollout` | Normalized training mean, `0` | Five-step autoregressive rollout |

The constant condition uses the same zeroed gravity input during training and
evaluation. The correct condition receives the true gravity value in both
phases.

The rollout cells replace the one-step objective. Their configs set
`prediction_weight: 0.0` and `rollout_weight: 1.0`. This experiment therefore
tests one-step-only training against rollout-only training. It does not test
whether adding a rollout term to a one-step loss is better.

## Results

The evaluation uses the model's autoregressive `predicted` trajectories. Each
cell evaluates 5,000 episodes from the same fixed test manifest. Values below
are means across the five evaluation seeds. Lower is better. Each entry is
`normalized target MSE / position L2`.

| Horizon | Correct, one-step | Constant, one-step | Correct, rollout | Constant, rollout |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.155 / 0.277 | 0.041 / 0.164 | 0.051 / 0.186 | 0.050 / 0.174 |
| 5 | 0.824 / 1.180 | 0.120 / 0.342 | 0.108 / 0.349 | 0.133 / 0.356 |
| 20 | 1.076 / 2.734 | 0.677 / 1.593 | 0.331 / 0.743 | 0.637 / 1.530 |
| 44 | 1.402 / 2.559 | 1.434 / 2.154 | 0.922 / 1.086 | 1.388 / 2.036 |

At horizon 20, rollout training reduces normalized MSE by 69% and position
error by 73% in the correct-gravity cell. In the constant-gravity cell, the
same change reduces those errors by only 6% and 4%.

The interaction is large. At horizon 20, the rollout effect in the
correct-gravity cell minus the rollout effect in the constant-gravity cell is
`-0.705` normalized MSE and `-1.929` position L2. Negative values mean that
rollout training helps more when the model receives the correct gravity.

Gravity conditioning alone does not produce a consistent gain. At horizons 1
and 5, `correct_onestep` is worse than `constant_onestep` on both metrics. At
horizon 44, correct gravity gives slightly lower MSE but higher position error
under one-step training. The gain is not a general one-step effect.

The probe results point in the same direction. Mean best validation MSE is
`0.01383` for `correct_onestep`, `0.01358` for `constant_onestep`, `0.01234`
for `correct_rollout`, and `0.01253` for `constant_rollout`. Rollout-trained
representations decode state slightly better, while grounding alone does not.

## Interpretation

A one-step objective can fit local transitions without making the predictor
use gravity. The visual history already contains much of the current state, so
the model can take a shortcut and predict an average local change.

Rollout training changes the pressure. The predictor must feed its own latent
back into the next step. If it ignores a gravity value that changes the
dynamics, the error compounds across the rollout. When the gravity input is
constant, the model has no evidence that it must represent a family of
dynamics. It can learn a more stable predictor for one average regime, but it
cannot use gravity to distinguish regimes.

This matches the explanation in [Semigroup-JEPA](https://arxiv.org/abs/2609.10464):
the paper reports a small teacher-forced gap, a much larger free-rollout gap,
and higher rollout error when it supplies incorrect gravity. Those results do
not isolate the two factors, but they are consistent with rollout training
making gravity operationally useful.

## Limits

- Each factorial cell has one world-model training seed. The five downstream
  seeds measure probe and evaluation variability, not training variability.
- The current evaluation compares free autoregressive predictions with an
  oracle probe baseline. Its `oracle` rows are ground-truth trajectories, not
  teacher-forced model predictions.
- The rollout cells replace one-step training instead of adding rollout loss
  to it. An additive-loss ablation remains open.
- The experiment uses one simulated task and one scalar physical parameter.

## Follow-up tests

1. Evaluate every checkpoint with true, zeroed, shuffled, and deliberately
   wrong gravity inputs. A correct-rollout model should degrade when gravity is
   wrong if it has learned to use the parameter.
2. Add a teacher-forced model evaluation. If the advantage appears only in free
   rollouts, it is a stability and consistency effect. If it appears under
   teacher forcing too, it also improves local state prediction.
3. Train additive-loss cells with both one-step and rollout terms, while
   matching optimizer steps and total compute.
4. Repeat all four training cells with several world-model seeds.

## Reproduction

The source patch for the upstream SG-JEPA checkout is
[`upstream.patch`](upstream.patch). It adds the four configs, gravity
conditioning, rollout training, Slurm entry points, probe fitting, and
evaluation.

The completed outputs were stored on Nexus under:

```text
/cmlscratch/ssoni11/sg-jepa-factorial/output/
```

The run used 20 training epochs, one world-model seed per cell, five probe
seeds per checkpoint, five evaluation seeds per checkpoint, and 5,000 test
episodes per evaluation.
