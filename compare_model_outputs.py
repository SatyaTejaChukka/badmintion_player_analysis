from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved LSTM and BiLSTM sequence-model outputs."
    )
    parser.add_argument(
        "--lstm-dir",
        type=Path,
        default=Path("compare_outputs/lstm"),
        help="Directory containing the LSTM attention-model artifacts.",
    )
    parser.add_argument(
        "--bilstm-dir",
        type=Path,
        default=Path("compare_outputs/bilstm"),
        help="Directory containing the BiLSTM attention-model artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("compare_outputs/comparison"),
        help="Directory where comparison files will be written.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected file was not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file was not found: {path}")
    return pd.read_csv(path)


def build_metric_comparison(
    lstm_summary: dict,
    bilstm_summary: dict,
) -> pd.DataFrame:
    metrics = [
        ("test_accuracy", True),
        ("test_roc_auc", True),
        ("test_loss", False),
        ("test_prefix_accuracy", True),
        ("test_prefix_roc_auc", True),
        ("test_next_shot_accuracy", True),
        ("test_next_shot_top3_accuracy", True),
        ("best_val_accuracy", True),
        ("best_val_auc", True),
        ("best_val_next_shot_accuracy", True),
        ("best_val_next_shot_top3_accuracy", True),
        ("epochs_ran", True),
    ]

    rows: list[dict[str, object]] = []
    for metric_name, higher_is_better in metrics:
        lstm_value = lstm_summary.get(metric_name)
        bilstm_value = bilstm_summary.get(metric_name)
        delta = None
        winner = "tie"
        if lstm_value is not None and bilstm_value is not None:
            delta = round(float(bilstm_value) - float(lstm_value), 4)
            if float(lstm_value) != float(bilstm_value):
                if higher_is_better:
                    winner = "BiLSTM" if float(bilstm_value) > float(lstm_value) else "LSTM"
                else:
                    winner = "BiLSTM" if float(bilstm_value) < float(lstm_value) else "LSTM"
        rows.append(
            {
                "metric": metric_name,
                "higher_is_better": higher_is_better,
                "lstm_value": lstm_value,
                "bilstm_value": bilstm_value,
                "bilstm_minus_lstm": delta,
                "better_model": winner,
            }
        )
    return pd.DataFrame(rows)


def build_prediction_comparison(
    lstm_predictions: pd.DataFrame,
    bilstm_predictions: pd.DataFrame,
) -> pd.DataFrame:
    join_columns = ["RallyId", "Player1Won"]
    bilstm_columns = [
        "RallyId",
        "Player1Won",
        "predicted_probability",
        "predicted_label",
    ]
    if "ShotTypes" in bilstm_predictions.columns:
        bilstm_columns.insert(1, "ShotTypes")

    merged = lstm_predictions.merge(
        bilstm_predictions[bilstm_columns],
        on=join_columns,
        suffixes=("_lstm", "_bilstm"),
    )

    if "ShotTypes_lstm" in merged.columns and "ShotTypes_bilstm" in merged.columns:
        merged["ShotTypes"] = merged["ShotTypes_bilstm"].fillna(merged["ShotTypes_lstm"])
        merged = merged.drop(columns=["ShotTypes_lstm", "ShotTypes_bilstm"])

    merged["probability_delta"] = (
        merged["predicted_probability_bilstm"] - merged["predicted_probability_lstm"]
    ).round(4)
    merged["abs_probability_delta"] = merged["probability_delta"].abs().round(4)
    merged["label_changed"] = (
        merged["predicted_label_lstm"] != merged["predicted_label_bilstm"]
    )
    merged["lstm_correct"] = merged["predicted_label_lstm"] == merged["Player1Won"]
    merged["bilstm_correct"] = merged["predicted_label_bilstm"] == merged["Player1Won"]

    outcome_labels = []
    for _, row in merged.iterrows():
        if row["lstm_correct"] and row["bilstm_correct"]:
            outcome_labels.append("both_correct")
        elif (not row["lstm_correct"]) and (not row["bilstm_correct"]):
            outcome_labels.append("both_wrong")
        elif row["bilstm_correct"]:
            outcome_labels.append("bilstm_only_correct")
        else:
            outcome_labels.append("lstm_only_correct")
    merged["comparison_outcome"] = outcome_labels

    ordered_columns = [
        column
        for column in [
            "RallyId",
            "ShotTypes",
            "Player1Won",
            "predicted_probability_lstm",
            "predicted_probability_bilstm",
            "probability_delta",
            "abs_probability_delta",
            "predicted_label_lstm",
            "predicted_label_bilstm",
            "label_changed",
            "lstm_correct",
            "bilstm_correct",
            "comparison_outcome",
        ]
        if column in merged.columns
    ]
    return merged[ordered_columns].sort_values(
        by=["abs_probability_delta", "RallyId"],
        ascending=[False, True],
    )


def build_summary(
    metric_comparison_df: pd.DataFrame,
    prediction_comparison_df: pd.DataFrame,
    lstm_summary: dict,
    bilstm_summary: dict,
) -> dict[str, object]:
    metrics_by_name = metric_comparison_df.set_index("metric")
    return {
        "input_file": bilstm_summary.get("input_file") or lstm_summary.get("input_file"),
        "lstm_model_type": lstm_summary.get("model_type", "LSTM"),
        "bilstm_model_type": bilstm_summary.get("model_type", "BiLSTM"),
        "test_rows_compared": int(len(prediction_comparison_df)),
        "lstm_test_accuracy": lstm_summary.get("test_accuracy"),
        "bilstm_test_accuracy": bilstm_summary.get("test_accuracy"),
        "accuracy_gap_bilstm_minus_lstm": metrics_by_name.at["test_accuracy", "bilstm_minus_lstm"],
        "lstm_test_roc_auc": lstm_summary.get("test_roc_auc"),
        "bilstm_test_roc_auc": bilstm_summary.get("test_roc_auc"),
        "roc_auc_gap_bilstm_minus_lstm": metrics_by_name.at["test_roc_auc", "bilstm_minus_lstm"],
        "lstm_test_loss": lstm_summary.get("test_loss"),
        "bilstm_test_loss": bilstm_summary.get("test_loss"),
        "loss_gap_bilstm_minus_lstm": metrics_by_name.at["test_loss", "bilstm_minus_lstm"],
        "lstm_test_prefix_accuracy": lstm_summary.get("test_prefix_accuracy"),
        "bilstm_test_prefix_accuracy": bilstm_summary.get("test_prefix_accuracy"),
        "prefix_accuracy_gap_bilstm_minus_lstm": metrics_by_name.at["test_prefix_accuracy", "bilstm_minus_lstm"],
        "lstm_test_next_shot_accuracy": lstm_summary.get("test_next_shot_accuracy"),
        "bilstm_test_next_shot_accuracy": bilstm_summary.get("test_next_shot_accuracy"),
        "next_shot_accuracy_gap_bilstm_minus_lstm": metrics_by_name.at["test_next_shot_accuracy", "bilstm_minus_lstm"],
        "lstm_test_next_shot_top3_accuracy": lstm_summary.get("test_next_shot_top3_accuracy"),
        "bilstm_test_next_shot_top3_accuracy": bilstm_summary.get("test_next_shot_top3_accuracy"),
        "next_shot_top3_gap_bilstm_minus_lstm": metrics_by_name.at["test_next_shot_top3_accuracy", "bilstm_minus_lstm"],
        "rows_with_changed_label": int(prediction_comparison_df["label_changed"].sum()),
        "rows_with_same_label": int((~prediction_comparison_df["label_changed"]).sum()),
        "avg_abs_probability_delta": round(float(prediction_comparison_df["abs_probability_delta"].mean()), 4),
        "max_abs_probability_delta": round(float(prediction_comparison_df["abs_probability_delta"].max()), 4),
        "bilstm_only_correct_rows": int((prediction_comparison_df["comparison_outcome"] == "bilstm_only_correct").sum()),
        "lstm_only_correct_rows": int((prediction_comparison_df["comparison_outcome"] == "lstm_only_correct").sum()),
        "both_correct_rows": int((prediction_comparison_df["comparison_outcome"] == "both_correct").sum()),
        "both_wrong_rows": int((prediction_comparison_df["comparison_outcome"] == "both_wrong").sum()),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lstm_summary = load_json(args.lstm_dir / "attention_summary.json")
    bilstm_summary = load_json(args.bilstm_dir / "attention_summary.json")
    lstm_predictions = load_predictions(args.lstm_dir / "attention_test_predictions.csv")
    bilstm_predictions = load_predictions(args.bilstm_dir / "attention_test_predictions.csv")

    metric_comparison_df = build_metric_comparison(lstm_summary, bilstm_summary)
    prediction_comparison_df = build_prediction_comparison(lstm_predictions, bilstm_predictions)
    summary = build_summary(
        metric_comparison_df=metric_comparison_df,
        prediction_comparison_df=prediction_comparison_df,
        lstm_summary=lstm_summary,
        bilstm_summary=bilstm_summary,
    )

    metric_comparison_df.to_csv(args.output_dir / "model_metric_comparison.csv", index=False)
    prediction_comparison_df.to_csv(args.output_dir / "model_prediction_differences.csv", index=False)
    (args.output_dir / "model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
