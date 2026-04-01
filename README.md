# Rally Intelligence for Badminton-Style Shot Sequences

This project turns rally-level shot sequences into tactical analysis.

It is built for questions like:

- Which rally patterns are associated with winning?
- Does a player overuse defense or neutral play?
- Which shot transitions appear strongest?
- What does the model think is tactically important inside a rally?
- What happens if we simulate a different next shot or a replacement shot?

The repository combines:

- rule-based tactical labeling,
- feature engineering,
- pattern mining,
- an attention-based sequence model,
- and a Streamlit dashboard for exploration.


## Quick Start

If you just want to run the project end to end:

```bash
pip install -r requirements.txt
python analyze_rallies.py
python train_sequence_model.py
streamlit run dashboard.py
```

Then look at:

- [outputs/rally_analysis.csv](./outputs/rally_analysis.csv) for rally-by-rally tactical features
- [outputs/pattern_stats.csv](./outputs/pattern_stats.csv) for the strongest tactical patterns
- [outputs/shot_pattern_stats.csv](./outputs/shot_pattern_stats.csv) for the strongest raw shot-name patterns
- [outputs/sequence_training_dataset.csv](./outputs/sequence_training_dataset.csv) for the exact full-sequence training rows
- [outputs/attention_shot_impact_trajectories.csv](./outputs/attention_shot_impact_trajectories.csv) for shot-by-shot win-probability changes
- [outputs/attention_pattern_stats.csv](./outputs/attention_pattern_stats.csv) for attention-based sequence signals
- [outputs/strategy_cluster_summary.csv](./outputs/strategy_cluster_summary.csv) for learned play-style clusters


## Architecture Diagram


```text
                    ┌────────────────────┐
                    │     Input Layer    │
                    │  CSV (Rally Data)  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Shot Parsing     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Tactical Labeling  │
                    │ (A / D / N / Term) │
                    └───────┬────────────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
        ▼                   ▼                    ▼
┌──────────────┐   ┌────────────────┐   ┌────────────────┐
│ Confidence   │   │ Feature Engg   │   │ Pattern Mining │
│ Scoring      │   └──────┬─────────┘   └──────┬─────────┘
└──────────────┘          │                    │
                          ▼                    ▼
                 ┌──────────────┐     ┌──────────────┐
                 │ Feature Tbls │     │ Pattern Stats│
                 └──────┬───────┘     └──────┬───────┘
                        ▼                    ▼
                 ┌──────────────┐     ┌──────────────┐
                 │ Rally CSV    │     │ Winning      │
                 │ Analysis     │     │ Patterns     │
                 └──────────────┘     └──────────────┘


        ┌────────────────────────────────────────────┐
        │         Sequence Modeling Layer            │
        └────────────────────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Sequence Encoding  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ BiLSTM + Attention │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌────────────┐     ┌──────────────┐     ┌──────────────┐
│ Attention  │     │ Embeddings   │     │ Counterfactual│
│ Weights    │     │              │     │ Simulation    │
└────┬───────┘     └──────┬───────┘     └──────┬───────┘
     ▼                    ▼                    ▼
┌────────────┐   ┌──────────────┐     ┌──────────────┐
│ Key Shots  │   │ Strategy     │     │ What-if      │
│ / Moments  │   │ Clustering   │     │ Recommendations│
└────────────┘   └──────┬───────┘     └──────────────┘
                        ▼
                 ┌──────────────┐
                 │ Play Styles  │
                 └──────────────┘


        ┌────────────────────────────────────────────┐
        │             Presentation Layer             │
        └────────────────────────────────────────────┘
                              │
        ┌────────────────────────────────────────────┐
        │          Streamlit Dashboard               │
        │                                            │
        │  - Rally Analysis CSV                      │
        │  - Winning Tactical Patterns               │
        │  - Key Moments                             │
        │  - Play Styles                             │
        │  - What-if Recommendations                 │
        └────────────────────────────────────────────┘
```






Plain-English flow:

- raw rally sequences are converted into tactical representations
- the tactical-analysis path creates interpretable features and mined patterns
- the deep-model path learns directly from shot order and exposes attention and embeddings
- both paths feed the dashboard and output files used for analysis


## 1. What Problem This Project Solves

The raw data gives one row per rally, with:

- `ShotTypes`: a comma-separated sequence such as `Serve, Lift, Drop, Smash`
- `Player1Won`: rally outcome (`1` if Player 1 won, `0` if Player 1 lost)

That means the real problem is not ordinary tabular classification.
It is:

1. sequence understanding,
2. tactical abstraction,
3. transition mining,
4. and explainable decision support.

Instead of only asking:

> Can we predict whether Player 1 wins?

this project asks:

> Which tactical sequences seem effective, how can we interpret them, and how can we simulate alternatives?


## 2. What We Built

There are two main modeling layers in the project.

### A. Tactical Analysis Pipeline

Implemented in [analyze_rallies.py](./analyze_rallies.py)

This layer:

- parses each shot sequence,
- maps shots into tactical categories,
- assigns confidence scores,
- extracts rally-level tactical features,
- mines repeated tactical patterns,
- and writes tactical analysis outputs.

This is the most interpretable part of the system.

### B. Sequence Intelligence Layer

Implemented in [sequence_intelligence.py](./sequence_intelligence.py) and [train_sequence_model.py](./train_sequence_model.py)

This layer:

- learns directly from raw shot order,
- uses a BiLSTM with temporal attention,
- trains on full rally sequences with the final win/loss label,
- exports attention weights per shot,
- exports win probability after every shot prefix during inference,
- mines high-attention subsequences,
- clusters rally embeddings into play styles,
- and supports counterfactual simulation.

This is the more advanced tactical-understanding layer.

### C. Interactive Dashboard

Implemented in [dashboard.py](./dashboard.py)

This layer:

- visualizes the tactical distribution of rallies,
- shows the strongest mined patterns,
- gives coach-style textual insights,
- analyzes user-entered rally sequences,
- plots the model win probability after each shot,
- and surfaces attention-model outputs in an interactive way.


## 3. Input Data Format

The current input file is:

- [final_corrected_input.csv](./final_corrected_input.csv)

Expected columns:

- `video_id`
- `RallyId`
- `StartTimestamp`
- `RallyEndTimestamp`
- `ShotTypes`
- `Player1Won`

Example:

```text
RallyId: 5_1
ShotTypes: Serve, Lift, Drop, Lift, Net
Player1Won: 0
```

Important assumptions:

- each row is one rally,
- `ShotTypes` is ordered in time,
- `Player1Won` may be missing for some rows,
- and the dataset is currently small, so all model outputs should be interpreted cautiously.


## 4. Tactical Categories Used

Shots are mapped into tactical labels through a configurable confidence map in [analyze_rallies.py](./analyze_rallies.py).

The main tactical labels are:

- `Attack`
- `Defense`
- `Neutral`
- `Terminal`

Examples:

- `Smash` is mostly `Attack`
- `Lift` is mostly `Defense`
- `Drop` is mostly `Neutral`
- `Miss`, `Out`, and `Fall` are treated as `Terminal`

Why `Terminal` matters:

- terminal events end the rally,
- they should not be treated as normal tactical behavior,
- and separating them keeps the rest of the tactical sequence cleaner.


## 5. End-to-End Approach

### Step 1. Parse rally sequences

Each `ShotTypes` string is split into a shot list:

```text
Serve, Lift, Drop, Smash
```

becomes:

```text
["Serve", "Lift", "Drop", "Smash"]
```

### Step 2. Tactical labeling with confidence

Each shot is assigned:

- a dominant tactical label,
- and a confidence distribution across the tactical classes.

Example:

- `Serve` -> mostly `Neutral`
- `Lift` -> mostly `Defense`
- `Smash` -> mostly `Attack`

### Step 3. Rally-level feature engineering

The analysis pipeline converts a rally into features such as:

- `attack_ratio`
- `defense_ratio`
- `neutral_ratio`
- `terminal_ratio`
- `rally_length`
- `rally_duration_sec`
- `attack_streak_max`
- `defense_to_attack_transitions`
- `neutral_to_attack_transitions`
- `early_attack`
- `last_tactical_attack`

This lets us ask questions like:

- does attacking early help?
- does defense-to-attack conversion matter?
- do long neutral rallies correlate with lower success?

### Step 4. Pattern mining

The project mines repeated tactical n-grams such as:

- `Defense -> Neutral`
- `Defense -> Neutral -> Attack`
- `Neutral -> Attack -> Defense`

It also mines repeated raw shot-name n-grams before categorization, such as:

- `Lift -> Drive`
- `Lift -> Drive -> Smash`
- `Drop -> Drive -> Drop`

For each pattern it tracks:

- total occurrences,
- number of rallies containing it,
- win rate,
- and lift versus the baseline win rate.

### Step 5. Attention-based sequence model

The advanced sequence model uses:

```text
Embedding -> BiLSTM -> Attention -> Dense -> Win Probability
```

Training mode:

- each rally is trained once as a complete ordered shot sequence
- the final `Player1Won` value is the supervision signal for that full sequence
- per-shot probability trajectories are produced later by scoring prefixes of the same rally at inference time

This model produces:

- predicted win probability,
- predicted win probability after each shot prefix,
- attention weights over shots,
- a learned context embedding for each rally,
- style clustering from those embeddings,
- and simulation outputs.

### Step 6. Counterfactual simulation

Two kinds of simulation are supported:

- append a possible next shot,
- replace one existing shot with another.

This makes it possible to ask:

- what if the player attacks earlier?
- what if a `Drop` becomes a `Smash`?
- what shot change improves the model estimate most?


## 6. Current Scripts and What They Do

### `analyze_rallies.py`

Purpose:

- offline tactical analysis and pattern mining

What it does:

- reads the CSV,
- labels each rally,
- computes features,
- mines tactical patterns,
- mines raw shot-name patterns,
- writes the main analysis outputs to `outputs/`

Run it with:

```bash
python analyze_rallies.py
```

### `train_sequence_model.py`

Purpose:

- train the advanced attention-based sequence model

What it does:

- builds the shot vocabulary,
- encodes and pads each rally as one full sequence,
- splits by rally first,
- trains a BiLSTM + Attention model,
- saves attention-based artifacts,
- mines high-impact attention patterns,
- clusters sequence embeddings into play styles,
- writes advanced outputs to `outputs/`

Run it with:

```bash
python train_sequence_model.py
```

### `dashboard.py`

Purpose:

- interactive exploration and tactical interpretation

What it does:

- loads the latest outputs,
- displays overview charts,
- shows top patterns,
- analyzes user-entered rally sequences,
- visualizes attention-model outputs if available

Run it with:

```bash
streamlit run dashboard.py
```

### `dashboard_support.py`

Purpose:

- shared logic for the dashboard

What it does:

- loads analysis bundles,
- creates coach insights,
- builds sequence recommendations,
- runs analyzer logic,
- integrates tactical analysis and attention-model outputs

### `sequence_intelligence.py`

Purpose:

- reusable deep-model utilities

What it does:

- defines the attention layer,
- builds the sequence model,
- loads saved sequence artifacts,
- predicts shot importance,
- runs simulations,
- extracts high-attention patterns,
- builds strategy clusters


## 7. How to Install and Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Generate tactical analysis outputs

```bash
python analyze_rallies.py
```

### Train the attention-based sequence model

```bash
python train_sequence_model.py
```

### Start the dashboard

```bash
streamlit run dashboard.py
```

Then open the local Streamlit URL shown in the terminal.


## 8. Where Results Appear

Most results are written into:

- [outputs/](./outputs)

This folder contains:

- CSV outputs,
- JSON metadata,
- trained model files,
- and dashboard-ready artifacts.


## 9. Output Files and How to Read Them

This is the most important section for a new user.

### A. `outputs/rally_analysis.csv`

What it is:

- the main rally-level output table

Important columns:

- `ShotSeqJson`: parsed shot sequence
- `LabelSeqJson`: tactical labels for that sequence
- `ConfidenceSeqJson`: confidence scores per shot
- `dominant_tactic`: dominant label in the rally
- `attack_ratio`, `defense_ratio`, `neutral_ratio`, `terminal_ratio`
- `rally_length`
- `rally_duration_sec`
- `attack_count`, `defense_count`, `neutral_count`, `terminal_count`
- `attack_streak_max`
- `defense_to_attack_transitions`
- `neutral_to_attack_transitions`
- `early_attack`
- `last_tactical_attack`

How to interpret it:

- this is the best file for rally-by-rally analysis
- a high `defense_to_attack_transitions` value suggests tactical conversion from recovery to pressure
- a high `neutral_ratio` suggests extended passive exchange

Example:

- if a rally has `attack_ratio = 0.45` and `early_attack = 1`, it means attacking started early and made up a large fraction of the rally

### B. `outputs/pattern_stats.csv`

What it is:

- tactical n-gram mining output from labeled tactical sequences

Important columns:

- `pattern_size`
- `pattern`
- `count`
- `rally_count`
- `wins`
- `losses`
- `win_rate`
- `rally_support_pct`
- `delta_vs_baseline`

How to interpret it:

- `count` tells how often the pattern appeared in total
- `rally_count` tells in how many distinct rallies it appeared
- `win_rate` tells how often Player 1 won when it appeared
- `delta_vs_baseline` tells whether the pattern is better or worse than the overall win rate

Good usage:

- trust patterns more when both `rally_count` and `win_rate` are reasonably strong
- be careful with tiny-support patterns

#### Also: `outputs/shot_pattern_stats.csv`

What it is:

- the same pattern-mining idea applied to the original shot names before categorization

Important columns:

- `pattern_size`
- `pattern`
- `count`
- `rally_count`
- `wins`
- `losses`
- `win_rate`
- `rally_support_pct`
- `delta_vs_baseline`

How to interpret it:

- this is the file to open when you want patterns like `Lift -> Drive -> Smash` rather than `Defense -> Attack -> Attack`
- it is usually the better file for coach-style shot-sequence analysis because it keeps the exact shot vocabulary
- it should be read with the same support caution as the tactical pattern table

### C. `outputs/attention_summary.json`

What it is:

- summary of the advanced attention-based sequence model

Important fields:

- `model_type`
- `train_rows`
- `test_rows`
- `vocab_size`
- `max_sequence_length`
- `test_accuracy`
- `test_roc_auc`
- `top_attention_pattern`
- `top_strategy_cluster`

How to interpret it:

- this tells how the sequence model performed and what it surfaced as its top learned signals
- on the current small dataset, these metrics are mainly experimental

Important note:

- the attention model is architecturally stronger than the plain LSTM baseline, but it still needs more data before its conclusions become reliable

### D. `outputs/attention_test_predictions.csv`

What it is:

- held-out test-rally predictions from the attention model

Important columns:

- `predicted_probability`
- `predicted_label`
- `strategy_cluster`
- `strategy_cluster_name`
- `top_attention_shot`
- `attention_weights_json`

How to interpret it:

- `top_attention_shot` tells which shot got the highest attention in that rally
- `attention_weights_json` shows how the model distributed attention across shot positions
- `prefix_win_probabilities_json` shows the inference-time win estimate after shot 1, shot 2, shot 3, and so on
- `shot_impact_deltas_json` shows how much each new shot moved the previous estimate
- use this file when inspecting single-rally sequence-model behavior

### E. `outputs/sequence_training_dataset.csv`

What it is:

- the explicit rally-level training table used for the sequence model

Important columns:

- `RallyId`
- `ShotTypes`
- `ShotSeqJson`
- `EncodedSequenceJson`
- `sequence_length`
- `Player1Won`
- `dataset_split`

How to interpret it:

- each row is one full rally
- `Player1Won` is the final win/loss label attached to the full shot sequence
- this is the clearest file to inspect if you want to see exactly what the full-sequence trainer sees

### F. `outputs/attention_shot_impact_trajectories.csv`

What it is:

- a shot-level trajectory table showing the model estimate after every shot prefix

Important columns:

- `RallyId`
- `Player1Won`
- `dataset_split`
- `position`
- `shot`
- `prefix_sequence`
- `win_probability_after_shot`
- `delta_vs_previous_shot`
- `cumulative_delta_vs_first_shot`

How to interpret it:

- `win_probability_after_shot` is the model estimate after observing the rally up to that shot
- `delta_vs_previous_shot` tells how much the newest shot changed the current outlook
- positive delta means that shot improved the modeled chance of winning
- negative delta means that shot reduced it

This is the main file to use when asking:

- which shot changed the rally outlook?
- where did momentum shift?
- how did win probability evolve from start to finish?

### G. `outputs/attention_pattern_stats.csv`

What it is:

- pattern mining output derived from high-attention subsequences

Important columns:

- `pattern`
- `count`
- `rally_count`
- `win_rate`
- `avg_predicted_win_probability`
- `avg_attention_weight`
- `rally_support_pct`

How to interpret it:

- these are not just frequent tactical patterns
- these are patterns that the sequence model tended to focus on
- `avg_attention_weight` is the key column here

Practical meaning:

- a pattern with high attention and decent support is a stronger candidate for tactical review than a pattern that is only frequent

### H. `outputs/strategy_cluster_summary.csv`

What it is:

- style clusters built from learned sequence embeddings

Important columns:

- `strategy_cluster`
- `strategy_cluster_name`
- `rallies`
- `win_rate`
- `avg_attack_ratio`
- `avg_defense_ratio`
- `avg_neutral_ratio`
- `avg_sequence_model_probability`

How to interpret it:

- this file groups rallies into learned play styles
- names such as `Aggressive`, `Defensive`, and `Balanced` are assigned heuristically after clustering
- use this to compare styles rather than judge an individual rally

Example:

- if a cluster has high `avg_attack_ratio` and better `win_rate`, that suggests more aggressive sequences may be associated with better outcomes in that sample

### I. `outputs/strategy_cluster_assignments.csv`

What it is:

- rally-level cluster assignments for all labeled rallies used in sequence-model training

Use it when:

- you want to know which style cluster each rally belongs to

### J. `outputs/attention_history.csv`

What it is:

- training history for the attention model

Contains:

- training loss,
- validation loss,
- training accuracy,
- validation accuracy,
- training AUC,
- validation AUC

How to interpret it:

- use it to check training stability
- if validation metrics are unstable, the dataset is probably too small or noisy

### K. `outputs/attention_metadata.json`

What it is:

- metadata needed to reuse the saved attention model

Contains:

- shot vocabulary,
- maximum sequence length,
- cluster name mapping,
- candidate shots for simulation

### L. `outputs/attention_sequence_model.keras`

What it is:

- saved Keras model for the attention-based sequence intelligence model

Use it when:

- loading the sequence model for inference or dashboard use

### N. Legacy `lstm_*` files

These files were produced by an earlier plain LSTM baseline:

- `lstm_summary.json`
- `lstm_history.csv`
- `lstm_test_predictions.csv`
- `lstm_metadata.json`
- `lstm_sequence_model.keras`

They are kept for comparison, but they are not the current recommended deep-model artifacts.

If you are new to the repo, focus on the `attention_*` files first.


## 10. How to Read the Dashboard

Run:

```bash
streamlit run dashboard.py
```

The dashboard has four main areas.

### Overview

Use this to understand:

- how much attack, defense, and neutral play exists on average,
- which shot types are most common,
- how tactical ratios relate to wins and losses,
- which tactical constructions are performing best

### Patterns

Use this to:

- inspect the strongest tactical patterns,
- switch between `Tactical Labels` and `Raw Shot Names`,
- filter by support and count,
- compare strategy buckets such as aggressive, defensive, and balanced constructions

### Rally Analyzer

Use this to enter a custom rally such as:

```text
Serve, Lift, Drop, Smash
```

The analyzer will show:

- shot-by-shot tactical classification,
- sequence-model predicted win probability when trained artifacts are available,
- sequence-model win probability after each shot,
- matched historical tactical patterns,
- matched raw shot-name patterns,
- next-shot simulation,
- attention-model key moments if the advanced artifacts exist,
- replacement-shot simulations

This is the best place for coach-style tactical interpretation.

### Deep Model

Use this to inspect:

- attention-model performance,
- high-impact attention patterns,
- learned strategy clusters,
- held-out test predictions


## 11. How to Interpret Results Responsibly

This project is useful, but the dataset is still small.

That means:

- model performance metrics are unstable,
- rare patterns can look stronger than they really are,
- attention weights are informative but not definitive,
- clustering is descriptive, not proof of true player archetypes

Recommended interpretation style:

- trust repeated patterns more than one-off sequences
- trust rally support more than raw count alone
- use pattern stats and sequence-model outputs together
- treat simulations as tactical suggestions, not guaranteed truth

Good question:

- "Does the data repeatedly suggest defense-to-attack transitions are valuable?"

Less safe question:

- "Does this prove Smash is always the best next shot?"


## 12. What the Current Results Suggest

On the current input file:

- several defense-to-neutral and defense-to-attack patterns appear promising
- the attention model runs successfully, but its performance is still weak because the dataset is small

So the main value right now is:

- exploratory tactical insight,
- sequence inspection,
- dashboard-driven rally review,
- and a strong foundation for future data growth

not a production-grade predictive model.


## 13. Recommended Workflow for a New Person

If you are opening this repo for the first time, use this order:

1. Read this README.
2. Open [outputs/rally_analysis.csv](./outputs/rally_analysis.csv) to inspect rally-level features.
3. Open [outputs/pattern_stats.csv](./outputs/pattern_stats.csv) to see tactical patterns.
4. Open [outputs/shot_pattern_stats.csv](./outputs/shot_pattern_stats.csv) to see raw shot-name patterns.
5. Run `streamlit run dashboard.py` for interactive exploration.
6. If needed, retrain the advanced model with `python train_sequence_model.py`.
7. Inspect [outputs/sequence_training_dataset.csv](./outputs/sequence_training_dataset.csv) if you want to see the exact full-sequence training rows.
8. Inspect [outputs/attention_shot_impact_trajectories.csv](./outputs/attention_shot_impact_trajectories.csv) to understand shot-by-shot probability changes.
9. Inspect [outputs/attention_pattern_stats.csv](./outputs/attention_pattern_stats.csv) and [outputs/strategy_cluster_summary.csv](./outputs/strategy_cluster_summary.csv) for advanced sequence insights.


## 14. Possible Future Improvements

Natural next steps:

- add more rally data,
- add player identity or match context,
- separate forced errors from unforced errors,
- add shot direction or court position,
- replace heuristics with more sport-specific taxonomy,
- upgrade from BiLSTM + Attention to Transformer models,
- add richer counterfactual search,
- add better evaluation protocols once the dataset grows


## 15. Short Summary

This project is an explainable rally intelligence system.

It starts from raw shot sequences and produces:

- tactical labels,
- rally features,
- winning patterns,
- attention-based shot importance,
- learned play-style clusters,
- and interactive tactical recommendations.

If you want the quickest path:

```bash
python analyze_rallies.py
python train_sequence_model.py
streamlit run dashboard.py
```

Then inspect:

- [outputs/rally_analysis.csv](./outputs/rally_analysis.csv)
- [outputs/pattern_stats.csv](./outputs/pattern_stats.csv)
- [outputs/shot_pattern_stats.csv](./outputs/shot_pattern_stats.csv)
- [outputs/sequence_training_dataset.csv](./outputs/sequence_training_dataset.csv)
- [outputs/attention_shot_impact_trajectories.csv](./outputs/attention_shot_impact_trajectories.csv)
- [outputs/attention_pattern_stats.csv](./outputs/attention_pattern_stats.csv)
- [outputs/strategy_cluster_summary.csv](./outputs/strategy_cluster_summary.csv)
