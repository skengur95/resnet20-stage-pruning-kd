# ResNet-20 Stage-wise Structured Pruning + Knowledge Distillation

**English** | [한국어](README.ko.md)

![MACs vs accuracy](figures/pareto_macs.png)

## TL;DR

- The combination that prunes the early stage (A) more, (0.4, 0.3, 0.2), has **18% fewer MACs than uniform 20% pruning, at about −0.9%p accuracy** (KD, mean of 5 runs: acc_kd 0.814 vs 0.823)
- At the same compute, **pruning the early stage (A) more favors accuracy, while pruning the late stage (C) more favors parameter reduction** (an accuracy–size trade-off)
- **KD beat FT (fine-tuning) in 64 of 65 comparisons**, and KD's gain over FT grew with the pruning level (+0.9 → +2.5%p)

## Contents

1. [Questions](#questions)
2. [Setup](#setup)
3. [Method](#method)
4. [Fair comparison](#fair-comparison)
5. [Design: 125 → 83 → 11](#design-125--83--11)
6. [Results](#results)
7. [Insights](#insights)
8. [Limitations](#limitations)
9. [Future work](#future-work)
10. [Reproduction](#reproduction)
11. [Credits](#credits)

## Questions

1. Which of ResNet-20's three stages (A, B, C) is most sensitive to pruning?
2. With a different pruning ratio per stage, can we get a model that is **smaller** than uniform 20% pruning with similar accuracy?
3. At the same compute, which stage is it better to prune more?

## Setup

| Item | Value |
|---|---|
| Model | ResNet-20 (pretrained on CIFAR-10, FP32 accuracy 0.846, 41.2M MACs, 272,474 params) |
| Data | CIFAR-10 subsets — 10,000 train / 2,000 eval images |
| Pruning | Structured (channel-wise), L1-norm importance, torch-pruning 1.6.0 `DependencyGraph` |
| Fine-tuning | SGD (lr 1e-2, momentum 0.9, weight decay 5e-4), cosine schedule, batch 128, 3 epochs |
| Augmentation | Horizontal flip, 4-px padding + random crop |
| KD | Teacher = original ResNet-20, T (temperature) = 4, α (CE weight) = 0.5, otherwise identical to FT |
| Repetition | 5 runs, different seed per run (run k → seed k−1) |
| Environment | Google Colab, NVIDIA T4 (≈5 s FT, ≈7 s KD per combination) |

KD loss: `α·CE + (1−α)·T²·KL(teacher ‖ student)` — a mix of matching the ground truth (CE) and matching the teacher's probability distribution (KL).

## Method

### ResNet-20 structure and tied channels

Because of the shortcut connections, the output channels of all blocks in a stage are tied together and cannot be pruned independently.

| Stage | Scope | Channels | Resolution | Params | MACs |
|---|---|---|---|---|---|
| A | stem · layer1 | 16 | 32×32 | 14,480 (5.3%) | 14.84M (36.0%) |
| B | layer2 | 32 | 16×16 | 51,648 (19.0%) | 13.22M (32.1%) |
| C | layer3 | 64 | 8×8 | 205,696 (75.5%) | 13.16M (31.9%) |
| Head | pool · fc | — | — | 650 (0.2%) | 0.005M (0.0%) |
| **Total** | | | | **272,474** | **41.24M** |

For a single conv, params = input channels × output channels × 3×3, and MACs = params × output resolution (width × height).
Each time the channel count doubles, the resolution drops to 1/4, so **MACs are spread fairly evenly across the three stages (32–36% each), while parameters grow about 4× per stage, putting 75% of them in C.**
This structure is key to interpreting the results.

### Part 1 — Stage sensitivity (no recovery training)

We prune only the shortcut path of each stage by 20% and compare the accuracy drop, both individually and cumulatively.
We use 0.2 because at 0.5, pruning just two paths already drops accuracy to 0.10 (random guessing over 10 classes), hiding any difference.

### Part 2 — Stage-wise ratios + recovery training

- Choose one ratio per stage `(rA, rB, rC)` and prune both the shortcut path and the block internals (conv1 outputs) at that ratio.
- Baseline: uniform 20% pruning of all layers with torch-pruning `MetaPruner` (fc excluded).
- For each combination: prune → measure accuracy (as-is / FT / KD).

## Fair comparison

- **Same scope**: block internals are pruned too, not just the paths, so the pruned scope matches the all-layer baseline
- **Only smaller models**: only combinations with **both** fewer MACs **and** fewer params than the baseline are compared
- **Pre-selection**: the 11 representative combinations were chosen from MACs and params only, before measuring any accuracy (not picked after seeing eval results)
- **Same conditions**: identical data, epochs, learning rate, eval set and seed for every combination
- **Baselines get the same procedure**: MetaPruner 20% and stage-uniform 20% are trained with the same FT/KD
- **Repeated runs**: mean and standard deviation over multiple runs

## Design: 125 → 83 → 11

1. 5 levels (0.1–0.5) × 3 stages = **125** combinations
2. Combinations smaller than the baseline (24.89M MACs, 170,772 params) in both = **83**
3. **11** representatives: within each group of similar MACs, only the pruning location (shape) changes

| Group | uniform | increasing (more at C) | decreasing (more at A) | middle (more at B) |
|---|---|---|---|---|
| ~20M | (0.3, 0.3, 0.3) | (0.2, 0.3, 0.4) | (0.4, 0.3, 0.2) | (0.2, 0.5, 0.2) |
| ~15M | (0.4, 0.4, 0.4) | (0.3, 0.4, 0.5) | (0.5, 0.4, 0.3) | (0.3, 0.5, 0.3) |
| ~11M | (0.5, 0.5, 0.5) | (0.4, 0.5, 0.5) | (0.5, 0.5, 0.4) | — |

Each tuple is `(rA, rB, rC)` = the pruning ratio of stages A, B and C.

## Results

### Part 1 — Sensitivity (ratio 0.2, no recovery training)

| Pruned paths | Accuracy | Note |
|---|---|---|
| none | 0.846 | original |
| A | 0.7955 | |
| A + B | 0.6770 | |
| A + B + C | 0.4795 | |
| B only | 0.7340 | |
| C only | 0.7665 | |

### Part 2 — After recovery training (mean ± std over 5 runs)

| Group | Shape | (rA, rB, rC) | MACs (M) | Params | acc_ft | acc_kd | kd_gain (%p) |
|---|---|---|---|---|---|---|---|
| baseline | MetaPruner 20% | all layers 0.2 | 24.89 | 170,772 | 0.814 ± 0.004 | 0.823 ± 0.003 | +0.88 |
| baseline | stage 20% | (0.2, 0.2, 0.2) | 27.04 | 175,362 | 0.820 ± 0.004 | 0.828 ± 0.004 | +0.85 |
| ~20M | uniform | (0.3, 0.3, 0.3) | 19.95 | 133,608 | 0.798 ± 0.002 | 0.809 ± 0.003 | +1.09 |
| ~20M | increasing | (0.2, 0.3, 0.4) | 21.02 | 108,869 | 0.778 ± 0.004 | 0.798 ± 0.004 | +1.99 |
| ~20M | decreasing | (0.4, 0.3, 0.2) | 20.44 | 159,597 | 0.805 ± 0.003 | **0.814** ± 0.003 | +0.95 |
| ~20M | middle | (0.2, 0.5, 0.2) | 21.51 | 149,922 | 0.781 ± 0.003 | 0.802 ± 0.004 | +2.12 |
| ~15M | uniform | (0.4, 0.4, 0.4) | 15.33 | 97,343 | 0.760 ± 0.004 | 0.783 ± 0.006 | +2.36 |
| ~15M | increasing | (0.3, 0.4, 0.5) | 15.30 | 78,524 | 0.742 ± 0.003 | 0.767 ± 0.004 | +2.50 |
| ~15M | decreasing | (0.5, 0.4, 0.3) | 14.92 | 122,580 | 0.772 ± 0.006 | 0.788 ± 0.002 | +1.60 |
| ~15M | middle | (0.3, 0.5, 0.3) | 16.96 | 119,904 | 0.774 ± 0.003 | **0.792** ± 0.004 | +1.76 |
| ~11M | uniform | (0.5, 0.5, 0.5) | 10.53 | 68,786 | 0.717 ± 0.003 | 0.742 ± 0.005 | +2.51 |
| ~11M | increasing | (0.4, 0.5, 0.5) | 12.68 | 71,132 | 0.723 ± 0.001 | 0.748 ± 0.005 | +2.47 |
| ~11M | decreasing | (0.5, 0.5, 0.4) | 11.80 | 88,790 | 0.746 ± 0.007 | **0.767** ± 0.006 | +2.08 |

Bold marks the highest KD accuracy within each group.
The dashed line is the Pareto front for KD (points with no more accurate alternative at equal or lower cost); × marks the FT result of the same combination.

![MACs vs accuracy](figures/pareto_macs.png)
![params vs accuracy](figures/pareto_params.png)

## Insights

1. **The sensitivity ranking flips with the pruning ratio.** (Part 1, no recovery training)
   - At ratios ≤ 0.2, A is the least sensitive (0.2: A 0.796 > C 0.767 > B 0.734)
   - At ratios ≥ 0.25, A is the most sensitive (0.25: A 0.586 < B 0.674 < C 0.722)
   - A has a cliff: going from 3 to 4 pruned channels drops accuracy 0.796 → 0.586, **21%p for a single channel**
   - → Don't conclude "where is sensitive" from a single ratio

2. **Accuracy right after pruning (or sensitivity before recovery) is not a reliable predictor of the ranking after recovery.**
   - ~20M decreasing (0.4, 0.3, 0.2) scored 0.101 right after pruning (essentially random), the lowest in its group, yet reached 0.814 after KD, the highest
   - In Part 1, A was the most sensitive stage at ratios ≥ 0.25, yet decreasing, which prunes A heavily (0.4–0.5), was the best after recovery
   - → Compare candidates by their accuracy after recovery training

3. **When reducing the same amount of MACs, pruning early (A) is better for accuracy; to reduce params, pruning late (C) is better.**
   - MACs are spread fairly evenly across stages (32–36% each), but parameters are concentrated in C (75%) (see the table in Method)
   - So pruning A reduces MACs while reducing params (capacity) only slightly, whereas pruning C reduces params substantially
   - decreasing (more at A) ranked first in KD accuracy within its group in 4 of 5 runs at ~20M (0.4, 0.3, 0.2) and in all 5 runs at ~11M (0.5, 0.5, 0.4)
   - At ~15M, middle (0.3, 0.5, 0.3) ranked first (0.792) but with 2M more MACs; decreasing (0.5, 0.4, 0.3) was second (0.788)
   - increasing (more at C) had the lowest accuracy at ~20M and ~15M but the fewest params (~15M (0.3, 0.4, 0.5): 78,524, −71% vs. the original)
   - Pareto front by MACs:
     ~11M uniform (0.5, 0.5, 0.5) → ~11M decreasing (0.5, 0.5, 0.4) → ~15M decreasing (0.5, 0.4, 0.3)
     → ~15M middle (0.3, 0.5, 0.3) → ~20M uniform (0.3, 0.3, 0.3) → ~20M decreasing (0.4, 0.3, 0.2)

4. **KD almost always beat FT.**
   - KD > FT in **64** of 65 comparisons (5 runs × 13 combinations); the only exception was stage 20% (0.2, 0.2, 0.2) in run 3 (−0.45%p)
   - Individual differences are small, but the direction is consistent, so KD can be considered better in this experimental setting
     (ResNet-20 · CIFAR-10 subset of 10,000 training images · 3 epochs · SGD lr 1e-2 · KD T = 4, α = 0.5)

5. **KD's gain in %p grows with pruning, but the share of the remaining gap it closes stays similar.**
   - kd_gain: +0.9%p at the baseline → +2.1–2.5%p at ~11M
   - Gap closed = kd_gain ÷ (original − FT): mostly 20–33%, and even lower at ~11M (about 20%)
   - → This can be read as the %p growing simply because there is a larger gap to close

6. **KD does not make up for the loss of a group pruned one step further.**
   - For the same shape, we compare "one group smaller + KD" with "one group larger + FT":

     | Shape | Smaller group + KD | Larger group + FT | KD higher? |
     |---|---|---|---|
     | uniform | ~15M (0.4, 0.4, 0.4): 0.783 | ~20M (0.3, 0.3, 0.3): 0.798 | ✗ |
     | increasing | ~15M (0.3, 0.4, 0.5): 0.767 | ~20M (0.2, 0.3, 0.4): 0.778 | ✗ |
     | decreasing | ~15M (0.5, 0.4, 0.3): 0.788 | ~20M (0.4, 0.3, 0.2): 0.805 | ✗ |
     | middle | ~15M (0.3, 0.5, 0.3): 0.792 | ~20M (0.2, 0.5, 0.2): 0.781 | ✓ |
     | uniform | ~11M (0.5, 0.5, 0.5): 0.742 | ~15M (0.4, 0.4, 0.4): 0.760 | ✗ |
     | increasing | ~11M (0.4, 0.5, 0.5): 0.748 | ~15M (0.3, 0.4, 0.5): 0.742 | ✓ |
     | decreasing | ~11M (0.5, 0.5, 0.4): 0.767 | ~15M (0.5, 0.4, 0.3): 0.772 | ✗ |

   - Holds in only 2 of 7 pairs. In those 2 pairs, the FT loss between groups was small (middle 0.7%p, increasing 1.9%p)
   - → KD's gain (+1–2.5%p) is generally smaller than the loss from compressing one group further (mostly 2.6–4.3%p)

7. **Against the baseline: less compute, but consistently slightly lower accuracy.**
   - ~20M decreasing (0.4, 0.3, 0.2) vs. MetaPruner 20%: MACs −18%, params −7%, KD accuracy −0.87%p
   - Per-run difference: −0.55, −0.30, −1.05, −1.65, −0.80%p → **lower in all 5 runs**
   - → "Same accuracy" is hard to claim; the accurate statement is "traded about 0.9%p accuracy for 18% fewer MACs"

## Limitations

- **Eval set of 2,000 images**: one image = 0.05%p; standard error near 0.8 accuracy ≈ ±0.9%p. Repeating runs reduces training randomness only, not this error
- **Short training**: 10,000 training images, 3 epochs. With longer training the FT–KD gap may shrink
- **Fixed KD setting**: T = 4 and α = 0.5 were fixed, not tuned. KD's advantage holds for this setting
- **MACs differ within a group**: MACs are not exactly equal within a group (~15M: 14.92M–16.96M, up to about 14% apart)
- **MACs ≠ actual speed**: latency was not measured. In particular, when the remaining channel count is not a multiple of 8 (e.g. 13, 51), hardware that computes channels in tiles may not speed up as much as the expected MACs reduction suggests
- **Baseline size mismatch**: due to rounding, the stage 20% baseline has more MACs than MetaPruner 20% (27.04M vs 24.89M), so the two baselines' accuracies cannot be compared without caveats
- **Rounding**: channel counts are integers, so the actual ratio differs from the nominal one (e.g. A: 16 channels × 0.4 → 6 channels = 37.5%)
- **GPU nondeterminism**: results can vary slightly even with the same seed because of cuDNN operation ordering
- Results are limited to **a single model (ResNet-20) on CIFAR-10**

## Future work

- Does KD's advantage over FT persist with more epochs?
- Pruning + INT8 quantization: does robustness to quantization depend on where we prune (shape)?
- Channel counts aligned to multiples of 8 vs. unaligned — accuracy and real latency
  (A has 16 channels, so only 0% or 50% is possible; on small models, fine-grained ratios can conflict with hardware alignment)
- Does KD's benefit drop under more extreme stage pruning (ratio 0.7–0.8)? (teacher–student capacity gap)
- Latency measurement on real devices (e.g. NPUs)

## Reproduction

```
resnet20-stage-pruning-kd/
├── notebooks/
│   ├── 01_stage_pruning_kd.ipynb      ← source of truth (English)
│   └── 01_stage_pruning_kd.ko.ipynb   ← Korean version (identical code)
├── results/        run*.csv (raw), summary.csv (mean)
├── figures/        pareto_macs.png, pareto_params.png
└── requirements.txt
```

**Colab**: open `notebooks/01_stage_pruning_kd.ipynb` → set runtime type to T4 GPU → run cells 1–3 and 7–9 →
repeat cell 10 with `RUN_ID = 1, 2, …` → cells 11 and 12.
Results are saved to Google Drive under `MyDrive/resnet20-stage-pruning-kd/`.

**Local**:

```
pip install -r requirements.txt
```

Then run the notebook from `notebooks/`. Without a GPU it runs on CPU.

## Credits

- Data subsets and pretrained weights: [imguru-mooc/AI_CAMPUS](https://github.com/imguru-mooc/AI_CAMPUS) (downloaded at runtime, not included in this repo)
- Pruning library: [Torch-Pruning](https://github.com/VainF/Torch-Pruning)