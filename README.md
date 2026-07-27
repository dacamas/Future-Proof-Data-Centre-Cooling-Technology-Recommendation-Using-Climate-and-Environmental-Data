# Future-Proof Data Centre Cooling Technology Recommendation

**A machine learning system that recommends the optimal cooling technology for a proposed AI data centre, anywhere on Earth, using only public climate, hydrological and geospatial data.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/datacentre_cooling_recommendation.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this does

Given a latitude/longitude and a design assumption about rack power density, the system retrieves that location's climate, terrain, hydrology and land cover in real time, then recommends one of four cooling strategies with a calibrated confidence and a SHAP explanation.

| Class | Strategy | Recommended when |
|---|---|---|
| `AIR` | Air cooling (dry coolers, free cooling, CRAH) | Cool climate, low design wet-bulb, many free-cooling hours, zero water use |
| `HYBRID` | Hybrid / adiabatically assisted dry cooling | Temperate climate with a distinct hot season; water used only on peak days |
| `EVAPORATIVE` | Cooling towers / direct-indirect evaporative | Hot **dry** climate (large wet-bulb depression) with accessible, un-stressed water |
| `LIQUID` | Direct-to-chip liquid cooling with water-lean rejection | High wet-bulb, water scarcity, high rack density, high projected warming |

This is **not** a prediction of what existing facilities currently use. It is a design-recommendation problem: what *should* be built at a location that has no data centre yet.

**Everything is automated.** No manual downloads, no uploads, no API keys. Open the notebook in Colab and run all.

---

## Results

Reference run, seed 42, 480 candidate sites, 70 model features, 36 minutes end to end on a free Colab CPU runtime.

| Metric | Value |
|---|---|
| Test accuracy | **0.917** |
| Test macro F1 | **0.906** |
| Balanced accuracy | 0.900 |
| ROC AUC (one-vs-rest) | 0.981 |
| Log loss | 0.381 |
| **Spatially blocked CV macro F1** | **0.850 ± 0.033** |
| Generalisation gap (random split − spatial) | 0.056 |

Per class on the held-out test split:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Air | 0.941 | 1.000 | 0.970 | 16 |
| Hybrid | 0.903 | 0.966 | 0.933 | 29 |
| Evaporative | 1.000 | 0.800 | 0.889 | 15 |
| Liquid | 0.833 | 0.833 | 0.833 | 12 |

**Quote the spatially blocked number, not the random-split one.** Nearby sites share a climate, so a random split lets the model see a test site's neighbours during training. Grouping by 10-degree geographic cell removes that, and the 0.056 gap is the honest cost of geographic generalisation.

### Where the model fails

Six errors out of 72. They are not randomly distributed:

| | Misclassified | Correct |
|---|---|---|
| Mean label entropy | **0.504** | 0.212 |
| Share above 0.5 entropy | 33 % | 14 % (all sites) |

Mann-Whitney p = 0.002. **Errors land where the engineering rule sets themselves disagree** — the sites where two competent engineers would specify different plant. The notebook measures this rather than asserting it, and prints the opposite conclusion if a run does not support it.

Equally important: **zero of the six errors are made confidently** (predicted probability ≥ 0.90). Mean predicted probability is 0.687 on errors against 0.787 on correct predictions — the model is measurably less sure when it is wrong. An earlier uncalibrated build got all six wrong at p ≥ 0.90, two at p ≈ 1.000; a recommendation tool that promises to flag borderline cases cannot be certain when it is wrong.

### Calibration selection

Three candidates scored on out-of-fold predictions over train+val (408 rows). The test split is never involved in the choice.

| Method | Macro F1 | Log loss | ECE | Mean confidence on errors | Confidently wrong |
|---|---|---|---|---|---|
| none | 0.853 | 0.672 | 0.118 | 0.885 | **62 %** |
| **sigmoid** (selected) | 0.846 | **0.440** | 0.120 | 0.583 | **1.7 %** |
| isotonic | 0.834 | 0.785 | **0.049** | 0.633 | 6.2 % |

Selection is by log loss, a proper scoring rule. The uncalibrated model is confidently wrong on 62 % of its mistakes — the pathology this step exists to remove. Isotonic wins on ECE but loses badly on log loss and costs the most accuracy, which is exactly the kind of trade-off a single hardcoded rule would have hidden.

### Climate scenario analysis

Re-running the full feature pipeline under perturbed climates and re-predicting:

| Scenario | Air | Hybrid | Evaporative | Liquid | Mean design wet-bulb |
|---|---|---|---|---|---|
| Baseline | 21.9 % | 41.0 % | 20.0 % | 17.1 % | 23.6 °C |
| +1 °C | 16.5 % | 43.8 % | 21.9 % | 17.9 % | 24.4 °C |
| +2 °C | 6.9 % | 48.1 % | 24.2 % | 20.8 % | 25.2 °C |
| +3 °C | 4.0 % | 45.6 % | 25.6 % | 24.8 % | 26.0 °C |
| +2 °C, −30 % rainfall | 3.3 % | 54.0 % | 19.8 % | 22.9 % | 25.2 °C |
| +2 °C, +8 pp humidity | 5.8 % | 42.7 % | 23.3 % | 28.1 % | 26.6 °C |

**26 % of sites change their recommended technology at +3 °C.** Air cooling nearly disappears; liquid cooling grows by half. Wet-bulb is recomputed from the perturbed dry-bulb and humidity via Stull rather than shifted naively, so hot-humid sites correctly degrade faster than hot-dry ones.

---

## How it works

```
Candidate sites → Data acquisition → Feature engineering → Rules engine → ML → Explanation → App
   480 global      8 public sources    ~110 features       4 classes    9 models   SHAP    Gradio
```

### 1. Candidate sites
A seeded Halton quasi-random sample over land (rejected against Natural Earth polygons) plus 60 real data-centre and industrial corridors with seeded jitter, filtered to a 40 km minimum separation so near-duplicates cannot leak across the train/test split.

### 2. Data acquisition — all automatic, all cached

| Source | Provides | Notes |
|---|---|---|
| **NASA POWER** (daily + climatology) | Daily T, RH, precipitation, solar, wind | Primary climate backbone, one cached request per site |
| **Open-Meteo / ERA5** | Independent cross-check on a subset | Rate-limit aware; quantifies reanalysis agreement |
| **Open-Meteo Elevation** (SRTM/Copernicus) | Elevation + terrain slope | 4-point stencil for slope |
| **Natural Earth** | Coastline, rivers, lakes, countries, populated places | Haversine BallTree nearest-distance queries |
| **OpenStreetMap / Overpass** | Substations, plants, transmission, reservoirs, industrial land, data centres | Health-probed, circuit-broken, time-budgeted |
| **Köppen-Geiger** | Climate zone | Computed from the monthly climatology (Peel et al. 2007) |
| **ESA WorldCover 10 m** | Land cover | Single-pixel windowed reads from cloud-optimised GeoTIFFs |
| **WRI Aqueduct 4.0** | Baseline water stress | Attempted; documented physical proxy otherwise |

### 3. Feature engineering
Around 110 features, then pruned to 70 for modelling. The physics-aware ones matter most:

- **Design wet-bulb** (99th-percentile, Stull 2011) — a cooling tower can never produce water colder than the ambient wet-bulb, which is why this is the model's top feature.
- **Wet-bulb depression** — how much evaporative cooling is physically available.
- **Approach-temperature headroom** — how far a wet tower (~4 K approach) or dry cooler (~10 K approach) sits below a 32 °C facility supply target.
- **Air density ratio** — the ISA barometric relation. Fans move volume, heat rejection scales with mass flow, so this is the altitude derating factor.
- **Reference evapotranspiration** via Hargreaves-Samani, giving a reproducible aridity index `P/ET0`.
- **Future design wet-bulb** under a latitude-amplified warming increment.
- Interaction terms (density × wet-bulb, water stress × CDD, aridity × summer temperature).

### 4. Label generation — the honest part

No public dataset lists the "correct" cooling technology per location, and the technology that *is* installed at existing sites reflects build vintage, corporate standards and water permits rather than optimal engineering. So labels are synthesised from published guidance:

- Each class scores in [0, 1] from weighted **linear membership ramps** over engineering variables. Ramps, not step functions — 22.0 °C design wet-bulb is not categorically different from 22.1 °C.
- Every threshold and weight lives in a single `RULES` dictionary and is exported to `rules_engine.json`.
- **A Monte-Carlo ensemble of 250 perturbed rule sets** then votes, with every threshold multiplied by `1 + N(0, 0.08)`. This models the fact that published thresholds are ranges, not points.

The modal vote becomes the label; the vote distribution gives `label_confidence` and `label_entropy`. Without this the labels would be a deterministic function of a handful of features and any decent model would score ~100 % — impressive-looking and meaningless. With it, boundary sites are genuinely ambiguous, and the model's errors land where they should: mean label entropy of misclassified sites is **0.504 vs 0.212 for correct ones** (Mann-Whitney p = 0.002).

Reference basis: ASHRAE TC 9.9 *Thermal Guidelines* (classes A1–A4, liquid W1–W5), ASHRAE 90.4, The Green Grid WUE white paper #35, Uptime Institute surveys, LBNL free-cooling studies, WRI Aqueduct 4.0.

### 5. Models
Nine classifiers benchmarked (logistic regression, decision tree, random forest, extra trees, gradient boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, plus an MLP), then Optuna tuning with 5-fold CV on macro F1.

Two engineering decisions worth noting:
- **Tuning is cost-aware.** Families whose measured single fit exceeds a threshold are benchmarked but not tuned — a 4-minute fit needs 12+ hours for a 40-trial 5-fold study.
- **Probability calibration is chosen from data.** The app's output *is* a probability that drives a borderline/settled judgement, and boosted trees are systematically overconfident on a few hundred rows. Three candidates — none, Platt/sigmoid and isotonic — are scored on **out-of-fold predictions over train+val** by log loss, with expected calibration error and the confidence placed on the model's own mistakes reported alongside. The test split is never involved in the choice.

  Calibration is not free, and the notebook shows the trade-off rather than assuming it: on the reference run it cut test log loss from 0.462 to 0.381 and eliminated confidently-wrong predictions, while nudging ECE from 0.091 to 0.138 as the model moved from mildly overconfident to mildly underconfident. Underconfidence is the safer failure mode for a screening tool, but both numbers are reported so the reader can judge.

### 6. Explainability, maps and app
SHAP global/local/dependence/interaction plots (against the uncalibrated base estimator, since calibration is a monotone per-class transform), interactive Folium maps with four layers, and a Gradio app that runs live inference for any coordinate with three independent borderline triggers: model confidence, rules-engine margin, and thermal headroom.

### Figures

<p align="center">
  <img src="figures/05_label_map.png" width="90%" alt="Recommended cooling technology by location"><br>
  <em>Rules-engine recommendation for 480 candidate sites</em>
</p>

<p align="center">
  <img src="figures/16_scenarios.png" width="90%" alt="Technology mix under climate scenarios and transition matrix"><br>
  <em>Technology mix under warming scenarios, and the baseline &rarr; +3 &deg;C transition matrix</em>
</p>

<p align="center">
  <img src="figures/15_shap_global.png" width="70%" alt="Global SHAP importance"><br>
  <em>Global SHAP importance &mdash; design wet-bulb and the density interactions dominate</em>
</p>

---

## Running it

**Colab (recommended):** click the badge above, then Runtime → Run all. Expect 30–50 minutes on the first run; subsequent runs reuse the cache and take a few minutes.

**Locally:** Jupyter with Python 3.10+. The first cell installs dependencies. Set `CFG["use_google_drive"] = False`.

Everything is configured in one place:

```python
CFG = {
    "n_sites": 420,                 # candidate locations
    "climate_start": "2020-01-01",  # reanalysis window
    "optuna_trials": 40,
    "enable_osm": True,             # set False to skip Overpass entirely
    "use_google_drive": True,       # cache in Drive, falls back to local
    ...
}
```

### Outputs

```
<project root>/
  data/      dataset (CSV + Parquet), metadata.json, feature_dictionary.csv, rules_engine.json
  models/    calibrated model, base model, preprocessor, Optuna studies, inference bundle
  figures/   18 analysis figures
  metrics/   leaderboard, classification report, test metrics, importances, scenarios
  shap/      global importance, per-class summaries, dependence plots, interactions
  maps/      interactive HTML map
  app/       Gradio launch configuration
  cache/     raw API payloads (delete to force a fresh download)
```

Reuse the trained model:

```python
import joblib
bundle = joblib.load("models/inference_bundle.joblib")
proba = bundle["model"].predict_proba(bundle["preprocessor"].transform(X))
```

---

## Engineering notes

Things that were genuinely hard, in case they are useful to anyone building similar pipelines:

- **Open-Meteo's archive API bills by `locations × variables × days`.** A request covering 12 sites, 10 variables and 8 years consumes thousands of call units, and the quota dies after a handful of requests. NASA POWER's daily endpoint has no such multiplicative cost and is the primary source for that reason.
- **Overpass is a shared, queued public service** and Colab's egress IPs are frequently throttled. A 15-second health probe, a circuit breaker and a hard time budget stop it consuming 20 minutes to return nothing. When coverage is thin the infrastructure features are automatically dropped from the model — a 95 %-imputed column is a constant wearing a disguise.
- **Google Drive's FUSE mount in Colab can vanish mid-run** (`OSError 107`). All writes go through helpers that fail over to local storage and continue.
- **Near-duplicate features destroy attribution.** The 30-year climatology restatements correlate with their daily-derived equivalents at r > 0.99; permutation importance collapses to near-zero for *both* members of such a pair. Features are pruned at |r| ≥ 0.98 with a canonical-survivor list so design wet-bulb, not some derived affine transform of it, is what SHAP reports.

---

## Limitations

Read this before trusting a number.

1. **Labels are synthetic.** They encode published guidance, not observed installations. The rules are auditable and configurable, but they remain a model of expert judgement, and four classes simplify a continuum — many real facilities run hybrid plant with liquid-cooled AI halls.
2. **High test accuracy measures how well the model learned the rules engine**, not how well it predicts real-world cooling choices. The meaningful results are calibration, the spatially blocked score, and where the errors land.
3. **Water stress is a proxy** unless the Aqueduct download succeeds. It captures aridity, seasonality, demand pressure and access — not basin hydrology, upstream withdrawal, groundwater depletion or transboundary allocation.
4. **No cost, energy price, carbon intensity, or permitting data.** In practice water abstraction permits, grid connection queues and electricity price often decide the technology before thermodynamics does.
5. **Reanalysis resolution is ~25–50 km.** Urban heat islands and plot-scale topography are invisible. Real design work uses on-site or airport TMY data.
6. **Future climate uses a parametric warming increment**, not downscaled CMIP6 ensembles. The scenario section is a sensitivity study, not a projection.
7. **Rack density is a design input, not a measurement** — the user controls it in the app.

### Natural extensions
Real Aqueduct basin polygons; downscaled CMIP6 projections; electricity price and grid carbon intensity for multi-objective TCO/WUE/PUE optimisation; multi-label output since hybrid architectures combine strategies; validation against a hand-curated set of real facilities with published cooling designs.

---

## References

- ASHRAE TC 9.9, *Thermal Guidelines for Data Processing Environments*, 5th ed.
- ASHRAE Standard 90.4, *Energy Standard for Data Centers*
- The Green Grid, WP#35, *Water Usage Effectiveness (WUE)*
- Stull, R. (2011). Wet-bulb temperature from relative humidity and air temperature. *J. Appl. Meteorol. Climatol.* 50, 2267–2269
- Peel, M. C., Finlayson, B. L., & McMahon, T. A. (2007). Updated world map of the Köppen-Geiger climate classification. *HESS* 11, 1633–1644
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS* 146, 1999–2049
- Allen, R. G. et al. (1998). *FAO-56: Crop evapotranspiration* (Hargreaves-Samani, eq. 52)
- WRI (2023). *Aqueduct 4.0*; Zanaga, D. et al. (2022). *ESA WorldCover 10 m 2021 v200*
- NASA POWER project; OpenStreetMap contributors; Natural Earth

**Data licences:** ERA5 via Open-Meteo (CC-BY-4.0), NASA POWER (public domain), OpenStreetMap (ODbL — attribution required), Natural Earth (public domain), ESA WorldCover (CC-BY-4.0).

## License

MIT for the code. Data products retain their own licences as listed above.
