from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


CATEGORIES = ("Attack", "Defense", "Neutral", "Terminal")

# Adjust these probabilities to reflect your tactical definitions.
DEFAULT_SHOT_CONFIDENCE = {
    "Backhand Serve": {"Attack": 0.10, "Defense": 0.00, "Neutral": 0.90, "Terminal": 0.00},
    "Clear": {"Attack": 0.05, "Defense": 0.75, "Neutral": 0.20, "Terminal": 0.00},
    "Cross Drop": {"Attack": 0.30, "Defense": 0.05, "Neutral": 0.65, "Terminal": 0.00},
    "Drive": {"Attack": 0.65, "Defense": 0.10, "Neutral": 0.25, "Terminal": 0.00},
    "Drop": {"Attack": 0.20, "Defense": 0.10, "Neutral": 0.70, "Terminal": 0.00},
    "Fall": {"Attack": 0.00, "Defense": 0.00, "Neutral": 0.00, "Terminal": 1.00},
    "Kill": {"Attack": 0.95, "Defense": 0.00, "Neutral": 0.05, "Terminal": 0.00},
    "Lift": {"Attack": 0.00, "Defense": 0.85, "Neutral": 0.15, "Terminal": 0.00},
    "Long Defence": {"Attack": 0.00, "Defense": 0.90, "Neutral": 0.10, "Terminal": 0.00},
    "Miss": {"Attack": 0.00, "Defense": 0.00, "Neutral": 0.00, "Terminal": 1.00},
    "Net": {"Attack": 0.15, "Defense": 0.05, "Neutral": 0.80, "Terminal": 0.00},
    "Net Kill": {"Attack": 0.95, "Defense": 0.00, "Neutral": 0.05, "Terminal": 0.00},
    "Net Shot": {"Attack": 0.15, "Defense": 0.05, "Neutral": 0.80, "Terminal": 0.00},
    "Out": {"Attack": 0.00, "Defense": 0.00, "Neutral": 0.00, "Terminal": 1.00},
    "Serve": {"Attack": 0.15, "Defense": 0.00, "Neutral": 0.85, "Terminal": 0.00},
    "Short Defence": {"Attack": 0.00, "Defense": 0.90, "Neutral": 0.10, "Terminal": 0.00},
    "Smash": {"Attack": 0.90, "Defense": 0.00, "Neutral": 0.10, "Terminal": 0.00},
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify rally shot sequences and mine winning patterns."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to the rally CSV. Defaults to the only CSV in the working directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where analysis files will be written.",
    )
    parser.add_argument(
        "--min-pattern-count",
        type=int,
        default=3,
        help="Minimum count required before a pattern is shown in the pattern table.",
    )
    parser.add_argument(
        "--pattern-sizes",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Pattern sizes to mine from label sequences.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path | None) -> Path:
    if path is not None:
        return path

    csv_files = sorted(Path.cwd().glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the working directory.")

    raise FileNotFoundError(
        "Multiple CSV files found in the working directory. Please pass --input explicitly."
    )


def normalize_confidence_map(confidence_map: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for shot, scores in confidence_map.items():
        merged_scores = {category: float(scores.get(category, 0.0)) for category in CATEGORIES}
        total = sum(merged_scores.values())
        if total <= 0:
            raise ValueError(f"Shot '{shot}' has no positive confidence scores.")
        normalized[shot] = {
            category: round(value / total, 4) for category, value in merged_scores.items()
        }
    return normalized


def parse_shot_sequence(raw_sequence: str) -> list[str]:
    if pd.isna(raw_sequence):
        return []
    return [shot.strip() for shot in str(raw_sequence).split(",") if shot.strip()]


def classify_shot(
    shot: str,
    confidence_map: dict[str, dict[str, float]],
    unknown_counter: Counter[str],
) -> tuple[str, dict[str, float]]:
    if shot not in confidence_map:
        unknown_counter[shot] += 1
        fallback = {"Attack": 0.0, "Defense": 0.0, "Neutral": 1.0, "Terminal": 0.0}
        return "Neutral", fallback

    scores = confidence_map[shot]
    label = max(scores, key=scores.get)
    return label, scores


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def max_streak(labels: Iterable[str], target: str) -> int:
    best = 0
    current = 0
    for label in labels:
        if label == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def count_transitions(labels: list[str], start: str, end: str) -> int:
    return sum(1 for left, right in zip(labels, labels[1:]) if left == start and right == end)


def last_non_terminal_label(labels: list[str]) -> str:
    for label in reversed(labels):
        if label != "Terminal":
            return label
    return "None"


def extract_features(
    labels: list[str],
    start_timestamp: float,
    end_timestamp: float,
) -> dict[str, float | int | str]:
    counts = Counter(labels)
    non_terminal_labels = [label for label in labels if label != "Terminal"]
    last_tactical = last_non_terminal_label(labels)

    return {
        "attack_ratio": safe_ratio(counts["Attack"], len(labels)),
        "defense_ratio": safe_ratio(counts["Defense"], len(labels)),
        "neutral_ratio": safe_ratio(counts["Neutral"], len(labels)),
        "terminal_ratio": safe_ratio(counts["Terminal"], len(labels)),
        "rally_length": len(labels),
        "non_terminal_length": len(non_terminal_labels),
        "rally_duration_sec": round(float(end_timestamp) - float(start_timestamp), 2),
        "attack_count": counts["Attack"],
        "defense_count": counts["Defense"],
        "neutral_count": counts["Neutral"],
        "terminal_count": counts["Terminal"],
        "attack_streak_max": max_streak(non_terminal_labels, "Attack"),
        "defense_to_attack_transitions": count_transitions(non_terminal_labels, "Defense", "Attack"),
        "neutral_to_attack_transitions": count_transitions(non_terminal_labels, "Neutral", "Attack"),
        "early_attack": int("Attack" in non_terminal_labels[:3]),
        "last_tactical_attack": int(last_tactical == "Attack"),
        "dominant_tactic": max(
            ("Attack", "Defense", "Neutral"),
            key=lambda category: counts.get(category, 0),
            default="Neutral",
        ),
    }


def get_ngrams(
    sequence: list[str],
    n: int,
    exclude_values: set[str] | None = None,
) -> list[tuple[str, ...]]:
    filtered_sequence = [item for item in sequence if exclude_values is None or item not in exclude_values]
    if len(filtered_sequence) < n:
        return []
    return [tuple(filtered_sequence[index : index + n]) for index in range(len(filtered_sequence) - n + 1)]


def get_patterns(labels: list[str], n: int) -> list[tuple[str, ...]]:
    return get_ngrams(labels, n, exclude_values={"Terminal"})


def build_pattern_table(
    df: pd.DataFrame,
    sequence_column: str,
    pattern_sizes: list[int],
    min_pattern_count: int,
    exclude_values: set[str] | None = None,
) -> pd.DataFrame:
    labeled_df = df.dropna(subset=["Player1Won"]).copy()

    pattern_stats: dict[tuple[int, tuple[str, ...]], dict[str, float | int]] = {}
    baseline_win_rate = labeled_df["Player1Won"].mean()

    for _, row in labeled_df.iterrows():
        won = int(row["Player1Won"])
        sequence = row[sequence_column]
        for pattern_size in pattern_sizes:
            row_patterns = get_ngrams(sequence, pattern_size, exclude_values=exclude_values)
            for pattern in row_patterns:
                key = (pattern_size, pattern)
                stats = pattern_stats.setdefault(key, {"count": 0, "wins": 0, "rally_count": 0})
                stats["count"] += 1
                stats["wins"] += won
            for pattern in set(row_patterns):
                key = (pattern_size, pattern)
                pattern_stats[key]["rally_count"] += 1

    records = []
    for (pattern_size, pattern), stats in pattern_stats.items():
        count = int(stats["count"])
        if count < min_pattern_count:
            continue
        wins = int(stats["wins"])
        losses = count - wins
        rally_count = int(stats["rally_count"])
        win_rate = wins / count
        records.append(
            {
                "pattern_size": pattern_size,
                "pattern": " -> ".join(pattern),
                "count": count,
                "rally_count": rally_count,
                "wins": wins,
                "losses": losses,
                "win_rate": round(win_rate, 4),
                "rally_support_pct": round(rally_count / len(labeled_df), 4),
                "delta_vs_baseline": round(win_rate - baseline_win_rate, 4),
            }
        )

    pattern_df = pd.DataFrame(records)
    if pattern_df.empty:
        return pd.DataFrame(
            columns=[
                "pattern_size",
                "pattern",
                "count",
                "rally_count",
                "wins",
                "losses",
                "win_rate",
                "rally_support_pct",
                "delta_vs_baseline",
            ]
        )

    return pattern_df.sort_values(
        by=["win_rate", "count", "pattern_size"], ascending=[False, False, True]
    ).reset_index(drop=True)


def enrich_rallies(
    df: pd.DataFrame,
    confidence_map: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, Counter[str]]:
    processed_rows = []
    unknown_counter: Counter[str] = Counter()

    for _, row in df.iterrows():
        shots = parse_shot_sequence(row["ShotTypes"])
        labels: list[str] = []
        confidence_seq: list[dict[str, float]] = []

        for shot in shots:
            label, scores = classify_shot(shot, confidence_map, unknown_counter)
            labels.append(label)
            confidence_seq.append(scores)

        features = extract_features(labels, row["StartTimestamp"], row["RallyEndTimestamp"])
        processed_rows.append(
            {
                "ShotSeq": shots,
                "LabelSeq": labels,
                "ConfidenceSeq": confidence_seq,
                **features,
            }
        )

    processed_df = pd.DataFrame(processed_rows)
    enriched_df = pd.concat([df.reset_index(drop=True), processed_df], axis=1)
    return enriched_df, unknown_counter


def write_outputs(
    df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    shot_pattern_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    export_df = df.copy()
    export_df["ShotSeqJson"] = export_df["ShotSeq"].apply(json.dumps)
    export_df["LabelSeqJson"] = export_df["LabelSeq"].apply(json.dumps)
    export_df["ConfidenceSeqJson"] = export_df["ConfidenceSeq"].apply(json.dumps)

    rally_output_columns = [
        "video_id",
        "RallyId",
        "StartTimestamp",
        "RallyEndTimestamp",
        "ShotTypes",
        "Player1Won",
        "ShotSeqJson",
        "LabelSeqJson",
        "ConfidenceSeqJson",
        "dominant_tactic",
        "attack_ratio",
        "defense_ratio",
        "neutral_ratio",
        "terminal_ratio",
        "rally_length",
        "non_terminal_length",
        "rally_duration_sec",
        "attack_count",
        "defense_count",
        "neutral_count",
        "terminal_count",
        "attack_streak_max",
        "defense_to_attack_transitions",
        "neutral_to_attack_transitions",
        "early_attack",
        "last_tactical_attack",
    ]

    export_df[rally_output_columns].to_csv(output_dir / "rally_analysis.csv", index=False)
    pattern_df.to_csv(output_dir / "pattern_stats.csv", index=False)
    shot_pattern_df.to_csv(output_dir / "shot_pattern_stats.csv", index=False)
    for legacy_path in [
        output_dir / "feature_importance.csv",
        output_dir / "model_summary.json",
        output_dir / "summary.txt",
    ]:
        if legacy_path.exists():
            legacy_path.unlink()


def run_analysis_pipeline(
    input_path: Path | None = None,
    output_dir: Path = Path("outputs"),
    min_pattern_count: int = 3,
    pattern_sizes: list[int] | None = None,
    write_files: bool = True,
) -> dict[str, object]:
    resolved_input_path = resolve_input_path(input_path)
    resolved_pattern_sizes = sorted(set(pattern_sizes or [2, 3]))
    confidence_map = normalize_confidence_map(DEFAULT_SHOT_CONFIDENCE)

    df = pd.read_csv(resolved_input_path)
    df["Player1Won"] = pd.to_numeric(df["Player1Won"], errors="coerce")

    enriched_df, unknown_counter = enrich_rallies(df, confidence_map)
    pattern_df = build_pattern_table(
        enriched_df,
        sequence_column="LabelSeq",
        pattern_sizes=resolved_pattern_sizes,
        min_pattern_count=max(1, min_pattern_count),
        exclude_values={"Terminal"},
    )
    shot_pattern_df = build_pattern_table(
        enriched_df,
        sequence_column="ShotSeq",
        pattern_sizes=resolved_pattern_sizes,
        min_pattern_count=max(1, min_pattern_count),
        exclude_values=None,
    )

    if write_files:
        write_outputs(
            enriched_df,
            pattern_df,
            shot_pattern_df,
            output_dir,
        )

    return {
        "input_path": resolved_input_path,
        "output_dir": output_dir,
        "confidence_map": confidence_map,
        "rally_df": enriched_df,
        "pattern_df": pattern_df,
        "shot_pattern_df": shot_pattern_df,
        "unknown_shots": unknown_counter,
    }


def main() -> int:
    args = parse_args()
    pipeline = run_analysis_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        min_pattern_count=args.min_pattern_count,
        pattern_sizes=args.pattern_sizes,
        write_files=True,
    )

    input_path = pipeline["input_path"]
    assert isinstance(input_path, Path)

    print(f"Analysis complete for {input_path.name}")
    print(f"Outputs written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
