---
title: "SIRA Issue Report: Substation System-Fragility Aggregation (median-across-classes)"
date: "2026-06-12"
geometry: "left=2cm,right=2cm,top=2cm,bottom=2cm"
output: pdf_document
---

## 1. Summary

SIRA derives a **substation's** system fragility from the failure rates of its
component *classes* (following HAZUS). The current implementation combines the
per-class signals by taking the **median across classes**. This median can be
"outvoted": when a single critical class fails but is outnumbered by robust
classes, the median ignores the failing class.

**Consequence:** a substation whose only power transformer is destroyed -- and
which therefore delivers zero output -- can be reported by the fitted system
fragility as *essentially undamaged*. The reported vulnerability understates the
true risk.

This report documents the issue, a minimal reproduction, the root cause, and the
recommended fix (HAZUS OR-logic / max-across-classes, computed per realisation).

- **Component:** `sira/infrastructure_response.py`, function
  `exceedance_prob_by_component_class` (approx. lines 1978-2006).
- **Scope:** substations only. Other facility classes (power stations, water
  plants) use a different, loss-based curve and are unaffected.
- **Severity:** the headline output a substation analysis produces -- its fitted
  system fragility curve -- can materially understate damage probability.

---

## 2. Background: how a substation's system fragility is computed

SIRA produces two different "system fragility" curves and chooses one per system
class for model fitting (`sira/__main__.py`, ~line 810):

| Curve | Built in | Used by |
|---|---|---|
| `pe_sys_econloss` | `write_system_response` | power stations, water plants, etc. |
| `pe_sys_cpfailrate` | `exceedance_prob_by_component_class` | **substations** |

The substation curve follows HAZUS, which defines a substation's damage state from
the fraction of its subcomponents (by type) that fail. The function docstring
cites *HAZUS MH MR3, p 8-66 to 8-68*.

The current algorithm, per hazard level:

1. For each component class: `failure_rate = (#components at damage state >= 2) /
   (#components in class)`.
2. Map that rate to a class damage index by counting how many of the class's
   limit thresholds (e.g. `[0.05, 0.40, 0.70, 1.00]`) it exceeds.
3. For each system damage state `d`: per class, compute the exceedance
   probability = mean over realisations of `(class_index >= d)`.
4. **System exceedance probability = median across the component classes** of
   those per-class probabilities.

Step 4 is the problem.

---

## 3. The issue

The **median across classes** is robust to outliers -- which is exactly the wrong
property here, because a single critical class *is* the outlier that matters.

If a substation has three component classes and only one of them (say the power
transformer) fails at a given hazard level, the per-class exceedance
probabilities look like `[1.0, 0.0, 0.0]`, and `median([1.0, 0.0, 0.0]) = 0.0`.
The system fragility therefore reports ~0 probability of damage, even though the
transformer -- and with it the whole substation -- is gone.

Two distinct departures from the intended HAZUS logic are present:

- **Wrong combiner.** HAZUS combines classes with **OR** ("40% of switches OR 40%
  of breakers OR ..."), i.e. the *worst* class governs (a maximum), not a median.
- **Wrong order of operations.** The code averages over realisations first (per
  class) and then combines across classes. The correct order is to combine across
  classes first (per realisation) and then average over realisations.

---

## 4. Root cause (code)

`sira/infrastructure_response.py`, `exceedance_prob_by_component_class`:

```python
# Probability of Exceedance -- Based on Failure of Component Classes
pe_sys_cpfailrate = np.zeros((len(infrastructure.system_dmg_states),
                              hazards.num_hazard_pts))
for d in range(len(infrastructure.system_dmg_states)):
    exceedance_probs = []
    for compclass in cp_classes_costed:
        if compclass in comp_class_frag:
            class_exceed = (comp_class_frag[compclass] >= d).mean(axis=0)
            exceedance_probs.append(class_exceed)
    if exceedance_probs:
        pe_sys_cpfailrate[d, :] = np.median(exceedance_probs, axis=0)  # <-- issue
```

`comp_class_frag[compclass]` is the per-realisation class damage index. The code
reduces it to a probability per class (`.mean(axis=0)`), then takes the **median**
across classes -- discarding the minority class that may be the only one that
matters.

---

## 5. Minimal reproduction

A purpose-built substation isolates the behaviour.

**Model** `tests/models/substation__median_caveat/` -- a 9-component substation,
single series chain, with three component classes:

```
SUPPLY -> SW_1 -> SW_2 -> SW_3 -> TX -> CB_1 -> CB_2 -> CB_3 -> OUT
```

- `TX` (Power Transformer): fragile (fails at low PGA) and the series bottleneck
  -- if it fails, system output is zero.
- `SW_1..3` (Disconnect Switch) and `CB_1..3` (Circuit Breaker): robust, almost
  never fail in the 0-1.2 g range.

Three classes total, so the single failing class (Power Transformer) is the
minority and is outvoted by the median.

**Result** (N = 2000 Monte-Carlo samples), comparing three views of the same run:

| PGA  | system output | econloss P(>=DS1) | cpfailrate MEDIAN (current) |
|-----:|--------------:|------------------:|----------------------------:|
| 0.20 | 0.766         | 0.234             | 0.000 |
| 0.30 | 0.265         | 0.735             | 0.000 |
| 0.40 | 0.053         | 0.947             | 0.000 |
| 0.60 | 0.004         | 0.997             | 0.000 |
| 0.80 | 0.000         | 1.000             | 0.000 |
| 1.00 | 0.000         | 1.000             | 0.001 |

At 0.8 g the substation delivers **zero output** (it is physically dead), the
loss-based fragility correctly reports **near-certain damage (1.000)**, yet the
median-across-classes curve -- the one a substation actually fits and reports --
says **~0.000 (undamaged)**.

A characterisation test pinning this behaviour is provided in
`tests/test_substation_median_caveat.py`.

---

## 6. Recommended fix: HAZUS OR-logic (max-across-classes, per realisation)

Assign a substation damage state **per realisation** as the worst-affected class
(OR-logic), then take the probability across realisations:

```python
# Per realisation, the substation damage index is the WORST class (HAZUS OR-logic)
sys_frag = np.zeros((num_samples, num_events), dtype=int)
for compclass in cp_classes_costed:
    indices = cp_class_indices[compclass]
    if len(indices) == 0:
        continue
    failures = (response_array[:, :, indices] >= 2).sum(axis=2) / len(indices)
    ds_lims = np.array(infrastructure.get_ds_lims_for_compclass(compclass))
    class_index = (failures[:, :, np.newaxis] > ds_lims).sum(axis=2)
    sys_frag = np.maximum(sys_frag, class_index)          # OR across classes

# P(system >= DSk) = fraction of realisations reaching at least state k
pe_sys_cpfailrate = np.array(
    [(sys_frag >= d).mean(axis=0)
     for d in range(len(infrastructure.system_dmg_states))]
)
```

This replaces both the per-class `.mean(axis=0)` and the `np.median(...)` combiner.
It now assigns a system damage state per realisation and counts over realisations
-- structurally the same shape as the loss-based path, but driven by HAZUS class
thresholds.

**Validation on the reproduction model** (max replaces median):

| PGA  | system output | econloss P(>=DS1) | cpfail MEDIAN (now) | cpfail MAX (proper) |
|-----:|--------------:|------------------:|--------------------:|--------------------:|
| 0.20 | 0.766         | 0.234             | 0.000               | 0.234 |
| 0.30 | 0.265         | 0.735             | 0.000               | 0.736 |
| 0.40 | 0.053         | 0.947             | 0.000               | 0.947 |
| 0.80 | 0.000         | 1.000             | 0.000               | 1.000 |
| 1.00 | 0.000         | 1.000             | 0.001               | 1.000 |

The corrected curve no longer masks the failure and agrees with the loss-based
curve to ~0.001 (because here the transformer drives both signals).

---

## 7. Important implementation subtlety

**Do not simply change `np.median` to `np.max` in the existing code.** The current
code operates in *probability space* -- each class has already been averaged over
realisations before being combined. Taking the `max` of per-class *probabilities*
underestimates the OR (union) probability, because different realisations can have
different worst classes.

```
WRONG (probability space):  P_sys(>=k) = max_classes[ mean_realisations(index_c >= k) ]
RIGHT (realisation space):  P_sys(>=k) = mean_realisations[ max_classes(index_c) >= k ]
```

The two coincide only when a single class dominates (as in the reproduction). The
correct fix must build the per-realisation `sys_frag` array with `np.maximum`
accumulated across classes, and average **after** that.

---

## 8. Secondary observations (optional, beyond the median fix)

1. **Hardcoded "failed = damage state >= 2".** The per-class fraction is evaluated
   at a single component-damage level and then compared against all system
   thresholds. A stricter HAZUS reading would tie the fraction at each *system*
   level to the corresponding *component* damage level. The max fix is the primary
   correction; this is a refinement.

2. **Small-class coarseness.** For a class with one or two components, the failure
   rate can only take a couple of discrete values (1 component -> {0, 1};
   2 components -> {0, 0.5, 1}). This makes adjacent system-damage curves coincide
   and can yield equal fitted medians for DS1 and DS2. This is inherent to the
   class-fraction method and is independent of the median-vs-max issue, but it
   compounds it.

3. **`max` and `econloss` are not the same metric.** They agree in the
   reproduction because one component dominates both. In a model with diffuse,
   partial class damage they legitimately diverge -- one counts subcomponent
   failures against thresholds, the other measures economic loss. Both are
   defensible; they answer different questions.

---

## 9. Impact

- Changing the combiner alters the fitted system fragility for **every**
  substation model and any committed reference outputs derived from it.
- Direction of the change: the corrected curve reports **higher** damage
  probability wherever a minority of critical classes drives failure -- i.e. it
  removes an optimistic bias. Risk and loss estimates based on the substation
  fragility would increase accordingly in affected scenarios.

---

## 10. Recommendation

1. Replace the median-across-classes combiner with the realisation-space
   max-across-classes implementation in Section 6.
2. Regenerate committed substation reference outputs.
3. Convert the caveat characterisation test into a regression test that asserts
   the corrected curve tracks the loss-based curve (no masking).
4. Separately consider the secondary refinements in Section 8.

---

## 11. Artifacts and references

- Code: `sira/infrastructure_response.py` -> `exceedance_prob_by_component_class`
  (combiner at approx. lines 1998-2006); fit-input selection at
  `sira/__main__.py` (approx. line 810).
- Reproduction model: `tests/models/substation__median_caveat/`.
- Characterisation test: `tests/test_substation_median_caveat.py`.
- Methodology reference: HAZUS-MH MR3 Earthquake Technical Manual,
  Electric Substations (damage-state definitions, p 8-66 to 8-68).
