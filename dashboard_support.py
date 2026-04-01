from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_rallies import (
    classify_shot,
    count_transitions,
    extract_features,
    get_ngrams,
    get_patterns,
    parse_shot_sequence,
    run_analysis_pipeline,
)
from sequence_intelligence import (
    DEFAULT_MIN_RECOMMENDATION_DELTA,
    SequenceIntelligenceArtifacts,
    load_sequence_intelligence_artifacts,
    predict_prefix_probability_trajectory,
    predict_sequence_intelligence,
    simulate_next_shot_candidates,
    simulate_shot_replacements,
)

@dataclass
class DashboardBundle:
    input_path: Path
    output_dir: Path
    confidence_map: dict[str, dict[str, float]]
    rally_df: pd.DataFrame
    pattern_df: pd.DataFrame
    shot_pattern_df: pd.DataFrame
    unknown_shots: Counter[str]
    strategy_stats_df: pd.DataFrame
    seconds_per_shot: float
    sequence_artifacts: SequenceIntelligenceArtifacts | None


def load_dashboard_bundle(
    input_path: Path | None = None,
    output_dir: Path = Path("outputs"),
) -> DashboardBundle:
    pipeline = run_analysis_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        min_pattern_count=3,
        pattern_sizes=[2, 3],
        write_files=True,
    )

    rally_df = pipeline["rally_df"].copy()
    assert isinstance(rally_df, pd.DataFrame)
    sequence_artifacts = load_sequence_intelligence_artifacts(output_dir)
    input_path = pipeline["input_path"]
    if sequence_artifacts is not None and sequence_artifacts.summary.get("input_file") != input_path.name:
        sequence_artifacts = None

    return DashboardBundle(
        input_path=input_path,
        output_dir=output_dir,
        confidence_map=pipeline["confidence_map"],
        rally_df=rally_df,
        pattern_df=pipeline["pattern_df"],
        shot_pattern_df=pipeline["shot_pattern_df"],
        unknown_shots=pipeline["unknown_shots"],
        strategy_stats_df=build_strategy_stats(rally_df),
        seconds_per_shot=estimate_seconds_per_shot(rally_df),
        sequence_artifacts=sequence_artifacts,
    )


def estimate_seconds_per_shot(rally_df: pd.DataFrame) -> float:
    durations = rally_df["rally_duration_sec"] / rally_df["rally_length"].replace(0, pd.NA)
    durations = durations.dropna()
    if durations.empty:
        return 2.0
    return round(float(durations.median()), 2)


def resolve_min_recommendation_delta(bundle: DashboardBundle) -> float:
    if bundle.sequence_artifacts is None:
        return DEFAULT_MIN_RECOMMENDATION_DELTA
    return float(
        bundle.sequence_artifacts.metadata.get(
            "min_recommendation_delta",
            DEFAULT_MIN_RECOMMENDATION_DELTA,
        )
    )


def apply_minimum_delta_threshold(
    df: pd.DataFrame,
    min_delta: float,
) -> pd.DataFrame:
    if df.empty or "delta_vs_current" not in df.columns:
        return df
    filtered_df = df[df["delta_vs_current"] >= float(min_delta)].copy()
    return filtered_df.reset_index(drop=True)


def shot_frequency_table(rally_df: pd.DataFrame) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for sequence in rally_df["ShotSeq"]:
        counts.update(sequence)

    rows = [{"shot": shot, "count": count} for shot, count in counts.most_common()]
    return pd.DataFrame(rows)


def assign_strategy_bucket_from_labels(labels: list[str]) -> str:
    tactical_labels = [label for label in labels if label != "Terminal"]
    if not tactical_labels:
        return "Unclassified"

    counts = Counter(tactical_labels)
    total = len(tactical_labels)
    attack_ratio = counts["Attack"] / total
    defense_ratio = counts["Defense"] / total
    neutral_ratio = counts["Neutral"] / total
    defense_to_attack = count_transitions(tactical_labels, "Defense", "Attack")

    if defense_to_attack > 0:
        return "Defense -> Attack Transition"
    if attack_ratio >= 0.45:
        return "Aggressive Play"
    if defense_ratio >= 0.45 and attack_ratio < 0.2:
        return "Over-Defensive"
    if neutral_ratio >= 0.55:
        return "Neutral Rally"
    return "Balanced Construction"


def build_strategy_stats(rally_df: pd.DataFrame) -> pd.DataFrame:
    labeled_df = rally_df.dropna(subset=["Player1Won"]).copy()
    if labeled_df.empty:
        return pd.DataFrame(columns=["strategy_bucket", "rallies", "wins", "win_rate"])

    labeled_df["strategy_bucket"] = labeled_df["LabelSeq"].apply(assign_strategy_bucket_from_labels)
    grouped = (
        labeled_df.groupby("strategy_bucket", as_index=False)
        .agg(
            rallies=("RallyId", "count"),
            wins=("Player1Won", "sum"),
            avg_attack_ratio=("attack_ratio", "mean"),
            avg_defense_ratio=("defense_ratio", "mean"),
            avg_neutral_ratio=("neutral_ratio", "mean"),
        )
        .sort_values(by=["rallies", "wins"], ascending=False)
        .reset_index(drop=True)
    )
    grouped["win_rate"] = (grouped["wins"] / grouped["rallies"]).round(4)
    grouped["avg_attack_ratio"] = grouped["avg_attack_ratio"].round(4)
    grouped["avg_defense_ratio"] = grouped["avg_defense_ratio"].round(4)
    grouped["avg_neutral_ratio"] = grouped["avg_neutral_ratio"].round(4)
    return grouped


def build_coach_insights(bundle: DashboardBundle) -> list[str]:
    labeled_df = bundle.rally_df.dropna(subset=["Player1Won"]).copy()
    if labeled_df.empty:
        return ["No labeled rally outcomes are available yet, so the dashboard can only show descriptive stats."]

    insights: list[str] = []

    with_transition = labeled_df[labeled_df["defense_to_attack_transitions"] > 0]["Player1Won"]
    without_transition = labeled_df[labeled_df["defense_to_attack_transitions"] == 0]["Player1Won"]
    if not with_transition.empty and not without_transition.empty:
        delta = with_transition.mean() - without_transition.mean()
        insights.append(
            "Defense-to-attack conversions change the outcome by "
            f"{delta:+.1%} compared with rallies that never convert defense into attack."
        )

    early_attack = labeled_df[labeled_df["early_attack"] == 1]["Player1Won"]
    late_attack = labeled_df[labeled_df["early_attack"] == 0]["Player1Won"]
    if not early_attack.empty and not late_attack.empty:
        delta = early_attack.mean() - late_attack.mean()
        insights.append(
            f"Attacking within the first three tactical shots changes win rate by {delta:+.1%}."
        )

    neutral_heavy = labeled_df[labeled_df["neutral_ratio"] >= 0.5]["Player1Won"]
    neutral_light = labeled_df[labeled_df["neutral_ratio"] < 0.5]["Player1Won"]
    if not neutral_heavy.empty and not neutral_light.empty:
        delta = neutral_heavy.mean() - neutral_light.mean()
        insights.append(
            f"Neutral-heavy rallies shift win rate by {delta:+.1%} relative to rallies with more proactive pressure."
        )

    stable_patterns = bundle.pattern_df[bundle.pattern_df["rally_count"] >= 5]
    if not stable_patterns.empty:
        top_pattern = stable_patterns.sort_values(by=["win_rate", "rally_count"], ascending=[False, False]).iloc[0]
        insights.append(
            "Most reliable winning pattern in this sample: "
            f"{top_pattern['pattern']} ({top_pattern['win_rate']:.1%} win rate across {int(top_pattern['rally_count'])} rallies)."
        )

    raw_shot_patterns = bundle.shot_pattern_df[bundle.shot_pattern_df["rally_count"] >= 3]
    if not raw_shot_patterns.empty:
        top_shot_pattern = raw_shot_patterns.sort_values(by=["win_rate", "rally_count"], ascending=[False, False]).iloc[0]
        insights.append(
            "Most reliable raw shot sequence in this sample: "
            f"{top_shot_pattern['pattern']} ({top_shot_pattern['win_rate']:.1%} win rate across {int(top_shot_pattern['rally_count'])} rallies)."
        )

    return insights[:4]


def _analyze_shot_list(
    shots: list[str],
    bundle: DashboardBundle,
) -> dict[str, Any]:
    unknown_counter: Counter[str] = Counter()
    labels: list[str] = []
    confidence_seq: list[dict[str, float]] = []

    for shot in shots:
        label, scores = classify_shot(shot, bundle.confidence_map, unknown_counter)
        labels.append(label)
        confidence_seq.append(scores)

    estimated_duration = round(max(len(shots), 1) * bundle.seconds_per_shot, 2)
    features = extract_features(labels, 0.0, estimated_duration)

    shot_rows = []
    for shot, label, scores in zip(shots, labels, confidence_seq):
        shot_rows.append(
            {
                "shot": shot,
                "predicted_label": label,
                "attack_confidence": round(scores["Attack"], 4),
                "defense_confidence": round(scores["Defense"], 4),
                "neutral_confidence": round(scores["Neutral"], 4),
                "terminal_confidence": round(scores["Terminal"], 4),
            }
        )

    return {
        "shots": shots,
        "labels": labels,
        "confidence_seq": confidence_seq,
        "features": features,
        "win_probability": None,
        "shot_table": pd.DataFrame(shot_rows),
        "unknown_shots": unknown_counter,
    }


def find_matching_patterns(
    sequence: list[str],
    pattern_df: pd.DataFrame,
    limit: int = 8,
    exclude_values: set[str] | None = None,
) -> pd.DataFrame:
    if pattern_df.empty:
        return pattern_df

    present_patterns: set[str] = set()
    for pattern_size in sorted(pattern_df["pattern_size"].unique()):
        for pattern in get_ngrams(sequence, int(pattern_size), exclude_values=exclude_values):
            present_patterns.add(" -> ".join(pattern))

    matches = pattern_df[pattern_df["pattern"].isin(present_patterns)].copy()
    if matches.empty:
        return matches

    return matches.sort_values(by=["win_rate", "rally_count"], ascending=[False, False]).head(limit)


def suggest_candidate_shots(
    shots: list[str],
    bundle: DashboardBundle,
    candidate_shots: list[str] | None = None,
) -> pd.DataFrame:
    if bundle.sequence_artifacts is None:
        return pd.DataFrame(
            columns=[
                "candidate_shot",
                "predicted_win_probability",
                "delta_vs_current",
                "next_shot_probability",
                "recommendation_score",
                "resulting_strategy",
            ]
        )

    base_shots = shots[:]
    if base_shots:
        base_result = _analyze_shot_list(base_shots, bundle)
        if base_result["labels"] and base_result["labels"][-1] == "Terminal":
            base_shots = base_shots[:-1]

    next_shot_df = simulate_next_shot_candidates(
        base_shots,
        bundle.sequence_artifacts,
        candidate_shots=candidate_shots,
    )
    if next_shot_df.empty:
        return pd.DataFrame(
            columns=[
                "candidate_shot",
                "predicted_win_probability",
                "delta_vs_current",
                "next_shot_probability",
                "recommendation_score",
                "resulting_strategy",
            ]
        )

    next_shot_df = next_shot_df.copy()
    next_shot_df["resulting_strategy"] = next_shot_df["candidate_shot"].map(
        lambda candidate: assign_strategy_bucket_from_labels(
            _analyze_shot_list(base_shots + [candidate], bundle)["labels"]
        )
    )
    return apply_minimum_delta_threshold(
        next_shot_df,
        min_delta=resolve_min_recommendation_delta(bundle),
    )


def recommendation_text(
    labels: list[str],
    features: dict[str, Any],
    strategy_stats_df: pd.DataFrame,
    matching_patterns: pd.DataFrame,
    matching_shot_patterns: pd.DataFrame,
    what_if_df: pd.DataFrame,
    sequence_result: dict[str, Any] | None = None,
) -> str:
    strategy = assign_strategy_bucket_from_labels(labels)
    lines = [f"Detected strategy: {strategy}."]

    strategy_row = strategy_stats_df[strategy_stats_df["strategy_bucket"] == strategy]
    if not strategy_row.empty:
        lines.append(
            f"Historical win rate for this strategy in your dataset: {strategy_row.iloc[0]['win_rate']:.1%}."
        )

    if features["defense_to_attack_transitions"] > 0:
        lines.append("You are creating at least one defense-to-attack conversion, which is one of the strongest signals in this dataset.")
    elif features["defense_ratio"] >= 0.35:
        lines.append("You spend a lot of the rally defending without converting it. Look for a drive, steep drop, or net pressure to change the rally state earlier.")

    if features["neutral_ratio"] >= 0.5:
        lines.append("This rally is neutral-heavy. The current data suggests long neutral exchanges are usually less efficient than forcing an earlier attacking phase.")

    if features["attack_ratio"] >= 0.45 and features["terminal_count"] > 0:
        lines.append("The rally is aggressive but ends with a terminal event, so keep the pressure while trimming avoidable errors.")
    elif features["attack_ratio"] < 0.25:
        lines.append("Attack volume is low. Try to earn an earlier attacking shot instead of extending the exchange passively.")

    if not matching_patterns.empty:
        best_match = matching_patterns.iloc[0]
        lines.append(
            f"Best matching historical pattern: {best_match['pattern']} with {best_match['win_rate']:.1%} win rate."
        )

    if not matching_shot_patterns.empty:
        best_shot_match = matching_shot_patterns.iloc[0]
        lines.append(
            f"Best matching raw shot sequence: {best_shot_match['pattern']} with {best_shot_match['win_rate']:.1%} win rate."
        )

    if not what_if_df.empty and what_if_df.iloc[0]["delta_vs_current"] > 0:
        best_option = what_if_df.iloc[0]
        plausibility_text = ""
        if "next_shot_probability" in what_if_df.columns and pd.notna(best_option.get("next_shot_probability")):
            plausibility_text = f" with {best_option['next_shot_probability']:.1%} next-shot plausibility"
        lines.append(
            f"Best immediate what-if: add {best_option['candidate_shot']} next, which raises the model estimate by {best_option['delta_vs_current']:+.1%}{plausibility_text}."
        )
    elif sequence_result is not None:
        min_delta = float(sequence_result.get("min_recommendation_delta", DEFAULT_MIN_RECOMMENDATION_DELTA))
        lines.append(
            f"No next-shot option clears the current minimum meaningful gain threshold of {min_delta:.1%}."
        )

    if sequence_result is not None:
        attention_df = sequence_result.get("shot_importance_df", pd.DataFrame())
        if isinstance(attention_df, pd.DataFrame) and not attention_df.empty:
            top_shot = attention_df.iloc[0]
            lines.append(
                f"Sequence model key moment: shot {int(top_shot['position'])} ({top_shot['shot']}) carries {top_shot['attention_weight']:.1%} of the attention."
            )
        prefix_df = sequence_result.get("prefix_probability_df", pd.DataFrame())
        if isinstance(prefix_df, pd.DataFrame) and not prefix_df.empty:
            biggest_swing = prefix_df.iloc[prefix_df["delta_vs_previous_shot"].abs().argmax()]
            if abs(float(biggest_swing["delta_vs_previous_shot"])) > 0:
                lines.append(
                    f"Biggest probability swing comes after shot {int(biggest_swing['position'])} ({biggest_swing['shot']}), changing the win estimate by {biggest_swing['delta_vs_previous_shot']:+.1%}."
                )
        replacement_df = sequence_result.get("replacement_df", pd.DataFrame())
        if isinstance(replacement_df, pd.DataFrame) and not replacement_df.empty and replacement_df.iloc[0]["delta_vs_current"] > 0:
            best_change = replacement_df.iloc[0]
            plausibility_text = ""
            if "candidate_shot_plausibility" in replacement_df.columns and pd.notna(best_change.get("candidate_shot_plausibility")):
                plausibility_text = f" with {best_change['candidate_shot_plausibility']:.1%} prefix-conditioned plausibility"
            lines.append(
                f"Best replacement simulation: swap shot {int(best_change['position'])} from {best_change['original_shot']} to {best_change['candidate_shot']} for a {best_change['delta_vs_current']:+.1%} gain{plausibility_text}."
            )
        elif isinstance(replacement_df, pd.DataFrame) and replacement_df.empty:
            min_delta = float(sequence_result.get("min_recommendation_delta", DEFAULT_MIN_RECOMMENDATION_DELTA))
            lines.append(
                f"No replacement option clears the current minimum meaningful gain threshold of {min_delta:.1%}."
            )

    return " ".join(lines)


def analyze_user_sequence(sequence_text: str, bundle: DashboardBundle) -> dict[str, Any]:
    shots = parse_shot_sequence(sequence_text)
    analysis = _analyze_shot_list(shots, bundle)
    matching_patterns = find_matching_patterns(
        analysis["labels"],
        bundle.pattern_df,
        exclude_values={"Terminal"},
    )
    matching_shot_patterns = find_matching_patterns(
        shots,
        bundle.shot_pattern_df,
        exclude_values=None,
    )
    what_if_df = suggest_candidate_shots(shots, bundle)
    strategy = assign_strategy_bucket_from_labels(analysis["labels"])
    sequence_result = None

    if bundle.sequence_artifacts is not None and shots:
        sequence_result = predict_sequence_intelligence(shots, bundle.sequence_artifacts)
        analysis["win_probability"] = sequence_result["win_probability"]
        sequence_result["min_recommendation_delta"] = resolve_min_recommendation_delta(bundle)
        sequence_result["prefix_probability_df"] = predict_prefix_probability_trajectory(
            shots,
            bundle.sequence_artifacts,
        )
        what_if_df = suggest_candidate_shots(
            shots,
            bundle,
            candidate_shots=None,
        )
        sequence_result["next_shot_df"] = what_if_df.copy()
        replacement_df = simulate_shot_replacements(
            shots,
            bundle.sequence_artifacts,
            candidate_shots=None,
        )
        sequence_result["replacement_df"] = apply_minimum_delta_threshold(
            replacement_df,
            min_delta=resolve_min_recommendation_delta(bundle),
        ).head(12)

    recommendation = recommendation_text(
        analysis["labels"],
        analysis["features"],
        bundle.strategy_stats_df,
        matching_patterns,
        matching_shot_patterns,
        what_if_df,
        sequence_result=sequence_result,
    )

    return {
        **analysis,
        "strategy_bucket": strategy,
        "matching_patterns": matching_patterns,
        "matching_shot_patterns": matching_shot_patterns,
        "what_if_df": what_if_df,
        "recommendation": recommendation,
        "sequence_result": sequence_result,
    }
