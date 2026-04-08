# Current Simulation Workflow Review Note

## 1. Purpose of this note

This document summarizes the current simulation workflow in `/sim`, the modeling logic behind it, and the main questions that should be reviewed before the next development step.

It is intentionally narrower and more reviewer-facing than `note.md`. The goal is not to archive every idea, but to make it easy for someone else to answer:

- What the current simulation is trying to represent
- How data flows from CSV files into the optimization model
- What assumptions are already implemented in code
- Which scripts represent the current working analysis path
- Which modeling choices still need technical review


## 2. Current modeling position

The current simulation does **not** treat MESS as an hourly vehicle-routing asset.

Instead, MESS is modeled as a **relocatable storage capacity pool** that:

- can be reassigned across service areas at discrete reconfiguration windows
- becomes available only after a deployment delay
- then participates in hourly charging/discharging and SOC evolution at the destination site

This means the current code is designed to study:

- medium-term or long-term spatial reallocation flexibility
- how mobile storage responds to cross-site demand heterogeneity
- when mobile storage starts to provide value beyond fixed storage

This version is **not** designed to study:

- explicit travel paths
- road-by-road routing
- vehicle dispatch at hourly granularity
- fine network physics such as voltage or branch flows


## 3. End-to-end workflow

The current workflow is:

1. Read the revised `g2_*` CSV files.
2. Convert them into `CoreModelData`.
3. Build a Pyomo MILP in `core_model.py`.
4. Solve with HiGHS.
5. Extract allocation, SOC, grid purchase, unserved load, and cost outputs.
6. Run scenario scripts that stress specific mechanisms such as seasonal heterogeneity or holiday staggering.

In code, the main chain is:

- data loader: `sim/src/eep_sim/g2_loader.py`
- optimization model: `sim/src/eep_sim/core_model.py`
- runnable entry points:
  - `sim/src/eep_sim/run_g2_loader_demo.py`
  - `sim/src/eep_sim/run_monthly_site_heterogeneity_demo.py`
  - `sim/src/eep_sim/run_holiday_stress_demo.py`
  - `sim/src/eep_sim/run_holiday_stagger_demo.py`
  - `sim/src/eep_sim/run_synthetic_reconfig_scan.py`


## 4. Data pipeline

### 4.1 Input files

The loader currently consumes three revised files from `data/raw/`:

- `data/raw/g2_beijing_shanghai_service_areas_hourly_load_revised.csv`
- `data/raw/g2_beijing_shanghai_service_areas_hourly_price_absolute_corrected.csv`
- `data/raw/g2_beijing_shanghai_service_areas_load_grid_params_revised.csv`

### 4.2 What the loader does

`load_g2_core_model_data()` builds model-ready inputs by:

- keeping only overlapping service areas across load, price, and parameter files
- reading nominal hourly load and grid-cap values
- resolving duplicate price rows by `(service, month, hour)` using a configurable rule
- mapping service-level metadata such as archetype, corridor position, peak time, and symbolic penalty hints
- constructing either:
  - a `representative_12x24` timeline, or
  - an `expanded_8760` timeline
- applying optional load transformations:
  - base-load scaling
  - monthly multipliers
  - site-month heterogeneity
  - holiday shocks
  - holiday site staggering
- deriving fixed-storage capacity by site archetype
- deriving total mobile capacity from fixed storage by ratio unless manually overridden
- constructing reconfiguration windows and the delayed mobile-capacity activation map

### 4.3 Current default baseline

Running:

```bash
uv run python sim/src/eep_sim/run_g2_loader_demo.py --base-dir .
```

currently gives the following baseline summary on this repo:

- sites: 23
- months: 12
- hours_of_day: 24
- timeline_hours: 288
- windows: 12
- window_size_steps: 24
- timeline_mode: `representative_12x24`
- fixed_total_mwh: 18.0
- mobile_total_mwh: 13.0
- duplicate_price_keys: 4032
- duplicate_price_rows: 6024
- price_duplicate_policy: `max`

Interpretation:

- the default baseline is a 12-month x 24-hour representative timeline
- each representative day is also one reconfiguration window under the default `reconf_hours=24`
- mobile capacity is currently derived externally, not optimized endogenously
- the default real-data baseline is not a strict same-budget comparison because `fixed_only` uses 18 MWh while `mobile_only` defaults to 13 MWh
- a capacity-fair fixed-versus-mobile check can be reproduced with `uv run python -m eep_sim.run_capacity_fair_baseline_demo --base-dir .`


## 5. Core optimization model

### 5.1 Decision logic

The model uses two linked time scales:

- **window scale** for mobile-capacity deployment
- **hourly scale** for operation

Key variables are:

- `x[i,w]`: mobile capacity assigned to site `i` in decision window `w`
- `m_active[i,w]`: mobile capacity actually available at site `i` in window `w` after delay
- `p_ch_*`, `p_dis_*`: charging/discharging power
- `soc_fixed[i,t]`, `soc_mobile[i,t]`: fixed and mobile SOC
- `p_grid[i,t]`: grid purchase
- `unserved[i,t]`: unmet load

### 5.2 Core constraints already implemented

The model in `sim/src/eep_sim/core_model.py` currently implements:

- mobile capacity pool:
  - `sum_i x[i,w] <= M_total`
- delayed activation:
  - `m_active[i,w] = x[i,w-delay]` for windows after the delay
  - early windows use `active_init`
- reconfiguration bound:
  - `sum_i |x[i,w]-x[i,w-1]| <= R_w`
- hourly power balance
- separate fixed/mobile SOC dynamics
- mobile energy conservation at window boundaries through `energy_shift_pos/neg`
- charging/discharging mutual exclusivity via a binary variable
- charge/discharge power limits from capacity and C-rate
- SOC upper/lower bounds
- grid import limit
- unserved load cap

### 5.3 Objective

The current solve objective is:

`C_grid + C_unserved + C_reconf`

where:

- `C_grid` is grid purchase cost
- `C_unserved` is unserved-load penalty
- `C_reconf` penalizes changes in mobile capacity allocation across windows

`C_storage` is also calculated for reporting, but is **not** in the optimization objective in the current solve path.

That means the current packaged comparisons are best read as operational-economic comparisons under exogenous storage sizing, not as a full endogenous investment-planning optimization.

### 5.4 Why this is now a MILP

The model is no longer a pure LP because hourly charge/discharge mutual exclusivity is enforced with a binary `is_charging[i,t]`.


## 6. Scenario scripts and what they are for

### 6.1 `run_g2_loader_demo.py`

Use this to verify that the revised CSVs can be loaded and mapped into a valid `CoreModelData` instance. It is mainly a data-interface smoke test.

### 6.2 `run_monthly_site_heterogeneity_demo.py`

This is the main representative-time script for asking:

- if site-specific seasonal demand waves differ across the corridor, does mobile storage start shifting meaningfully across windows?

It modifies demand at the site-month level while keeping the representative 12x24 structure.

### 6.3 `run_holiday_stress_demo.py`

This isolates full holiday windows from the `expanded_8760` timeline and solves them independently. It is useful for checking whether major holiday periods create enough pressure for mobile storage to matter.

### 6.4 `run_holiday_stagger_demo.py`

This is a stronger stress test than the plain holiday script. It injects synthetic cross-site holiday staggering, which is important because corridor-total holiday load growth alone may still be too spatially smooth to induce much reallocation.

### 6.5 `run_synthetic_reconfig_scan.py`

This is a diagnostic script, not a final case-study script. It builds a fully synthetic corridor and scans:

- hotspot intensity
- grid margin
- delay
- reconfiguration cost
- unserved-penalty multiplier

The purpose is to identify the threshold at which reconfiguration becomes material. This helps answer whether weak reconfiguration results come from:

- a modeling error, or
- a scenario that simply does not create enough spatial pressure


## 7. Current modeling thought process

The current line of thinking behind the simulation is:

### 7.1 Main research shift

The earlier idea of hourly path-based MESS movement was judged to be misaligned with the research objective and likely too detailed for the available data. The current approach keeps the key differentiator of MESS, namely **cross-site redeployability**, without forcing a vehicle-routing model.

### 7.2 Why use continuous mobile capacity first

The current model uses continuous mobile-capacity allocation rather than discrete units because it is intended to answer first-order questions:

- which sites want extra mobile capacity
- when reconfiguration happens
- how delay suppresses value
- whether heterogeneity is strong enough to justify mobility at all

This continuous version is acting as a screening model before any integer-unit design.

### 7.3 Why the recent scenario design emphasizes heterogeneity

If all sites have similar temporal patterns, MESS has little reason to move. The current simulation direction therefore deliberately increases:

- seasonal site heterogeneity
- holiday surges
- corridor-position-dependent holiday staggering

The point is not to exaggerate demand arbitrarily, but to create conditions where the value proposition of relocatable storage can actually be tested.

### 7.4 Why energy-shift logic was added

Once mobile capacity is allowed to move across windows, energy cannot simply disappear at one site and reappear at another. The `energy_shift_pos/neg` mechanism was added so that mobile SOC is transferred consistently at window boundaries when active mobile capacity changes.


## 8. What has already been validated

The automated test suite currently checks the main mechanics of the platform, including:

- build and solve of the core model
- mobile-capacity pool constraint
- delay activation and hard reconfiguration limit
- charge/discharge efficiency
- charge/discharge mutual exclusivity
- mobile energy conservation across reconfiguration
- no energy shift without capacity reconfiguration
- representative and expanded loader modes
- monthly multipliers
- site-month heterogeneity
- holiday shocks and holiday staggering

Verification command:

```bash
uv run python -m unittest discover -s sim/tests -p "test_*.py" -v
```

Current status in this repo: `23 tests, all passing`.


## 9. Main outputs reviewers should pay attention to

At this stage, the most informative outputs are:

- `C_grid`
- `C_unserved`
- `C_reconf`
- total unserved load
- `reconf_total_mwh`
- site-by-window `x[i,w]`
- top moved sites across windows

Interpretation guidance:

- low `reconf_total_mwh` does not automatically mean the model is wrong
- it may indicate that the scenario does not create enough cross-site scarcity asymmetry
- if `C_unserved` stays high while `reconf_total_mwh` stays low, that is a signal to inspect either:
  - delay and reconfiguration frictions, or
  - whether the stress pattern is too corridor-wide and not spatially differentiated enough


## 10. Main review questions

The following points are the most useful review targets right now:

### 10.1 MESS definition

Is the current definition of MESS as a delayed, window-level relocatable capacity pool an acceptable abstraction for the thesis objective?

### 10.2 Fairness of the comparison

Is the current comparison between fixed storage and MESS fair enough when:

- total mobile capacity is externally set
- storage investment is reported but not optimized in the operation solve
- mobile power rating is tied implicitly to energy capacity via C-rate

Current answer: not by default. The packaged baseline remains useful for studying timing and deployment logic under the default sizing rule, but it should not be described as a strict same-budget result unless `m_total_mwh` is overridden explicitly.

### 10.3 Reconfiguration realism

Is `sum_i |x[i,w]-x[i,w-1]| <= R_w` sufficient as a first-order realism constraint, or does the current setup need a stronger physical interpretation for redeployment limits?

### 10.4 Scenario credibility

Are the current site-month and holiday-stagger mechanisms a reasonable proxy for spatial demand heterogeneity, or do they need calibration/justification before being used in the main narrative?

### 10.5 Timeline choice

Should the main analysis continue to use:

- `representative_12x24` for broad scanning, and
- `expanded_8760` only for stress cases,

or is that split likely to create interpretation issues?

### 10.6 Next modeling priority

What should come next after this continuous-capacity screening stage?

Likely candidates are:

- better scenario calibration
- more formal baseline comparison design
- endogenous capacity sizing
- discrete mobile unit quantization


## 11. Suggested review stance

The best way to review the current simulation is:

1. First review whether the **abstraction level** is correct.
2. Then review whether the **scenario construction** is strong enough to reveal MESS value.
3. Only after that, review whether the model should become more detailed.

The current risk is less about solver feasibility and more about **research alignment**: if the abstraction and the stress design are wrong, adding more detail will not fix the core problem.


## 12. Reproducible commands

Quick data/model smoke test:

```bash
uv run python sim/src/eep_sim/run_g2_loader_demo.py --base-dir .
```

Representative monthly heterogeneity run:

```bash
uv run python sim/src/eep_sim/run_monthly_site_heterogeneity_demo.py --base-dir .
```

Holiday stress windows:

```bash
uv run python sim/src/eep_sim/run_holiday_stress_demo.py --base-dir .
```

Holiday stagger stress test:

```bash
uv run python sim/src/eep_sim/run_holiday_stagger_demo.py --base-dir .
```

Synthetic threshold scan:

```bash
uv run python sim/src/eep_sim/run_synthetic_reconfig_scan.py
```

Test suite:

```bash
uv run python -m unittest discover -s sim/tests -p "test_*.py" -v
```
