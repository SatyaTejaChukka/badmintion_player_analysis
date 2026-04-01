from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed

from analyze_rallies import resolve_input_path, run_analysis_pipeline
from sequence_intelligence import (
    DEFAULT_CONTEXT_CANDIDATE_TOP_K,
    DEFAULT_MIN_RECOMMENDATION_DELTA,
    SequenceIntelligenceArtifacts,
    build_context_transition_statistics,
    build_attention_models,
    build_cluster_outputs,
    build_multitask_training_examples,
    build_next_shot_training_examples,
    build_prefix_training_examples,
    build_shot_impact_trajectory_dataset,
    build_vocabulary,
    calibrate_win_probabilities,
    encode_sequences,
    extract_high_impact_patterns,
    fit_binary_temperature_scaler,
    pad_encoded_sequences,
    unpack_model_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an interpretable attention-based sequence model for rally tactics."
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
        help="Directory where model artifacts will be written.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--test-size", type=float, default=0.25, help="Fraction reserved for test evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--clusters", type=int, default=3, help="Number of strategy clusters.")
    parser.add_argument(
        "--min-recommendation-delta",
        type=float,
        default=DEFAULT_MIN_RECOMMENDATION_DELTA,
        help="Minimum calibrated win-probability gain required before a recommendation is shown.",
    )
    parser.add_argument(
        "--context-candidate-top-k",
        type=int,
        default=DEFAULT_CONTEXT_CANDIDATE_TOP_K,
        help="Maximum number of context-aware candidate shots to surface per recommendation view.",
    )
    parser.add_argument(
        "--encoder",
        choices=["lstm", "bilstm"],
        default="bilstm",
        help="Recurrent encoder to use before attention.",
    )
    return parser.parse_args()


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0 or weights.size == 0 or float(weights.sum()) <= 0:
        return None
    return float(np.average(values, weights=weights))


def weighted_top_k_accuracy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> float | None:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if y_true.size == 0 or probabilities.size == 0 or float(weights.sum()) <= 0:
        return None

    top_k = np.argsort(probabilities, axis=1)[:, -k:]
    hits = np.asarray([int(label in row) for label, row in zip(y_true, top_k)], dtype=float)
    return float(np.average(hits, weights=weights))


def split_train_validation_indices(
    train_indices: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(train_indices) < 4:
        return train_indices, np.asarray([], dtype=int)

    train_labels = labels[train_indices]
    stratify_labels = None
    unique_values, counts = np.unique(train_labels, return_counts=True)
    if len(unique_values) > 1 and counts.min() >= 2:
        stratify_labels = train_labels

    try:
        core_train_indices, validation_indices = train_test_split(
            train_indices,
            test_size=0.2,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        core_train_indices, validation_indices = train_test_split(
            train_indices,
            test_size=0.2,
            random_state=seed,
            stratify=None,
        )

    return np.asarray(core_train_indices), np.asarray(validation_indices)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_name = args.encoder.strip().lower()
    model_type_name = "BiLSTM + Attention" if encoder_name == "bilstm" else "LSTM + Attention"
    min_recommendation_delta = max(0.0, float(args.min_recommendation_delta))
    context_candidate_top_k = max(1, int(args.context_candidate_top_k))

    set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    input_path = resolve_input_path(args.input)
    pipeline = run_analysis_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        min_pattern_count=3,
        pattern_sizes=[2, 3],
        write_files=False,
    )

    rally_df = pipeline["rally_df"]
    assert isinstance(rally_df, pd.DataFrame)
    labeled_df = rally_df.dropna(subset=["Player1Won"]).copy().reset_index(drop=True)
    labeled_df = labeled_df[labeled_df["ShotSeq"].map(bool)].reset_index(drop=True)
    if labeled_df.empty:
        raise ValueError("No labeled shot sequences are available for sequence-intelligence training.")

    sequences = labeled_df["ShotSeq"].tolist()
    y = labeled_df["Player1Won"].astype(int).to_numpy()
    shot_to_id = build_vocabulary(sequences)
    encoded_sequences = encode_sequences(sequences, shot_to_id)
    max_sequence_length = max(len(sequence) for sequence in encoded_sequences)
    X = pad_encoded_sequences(encoded_sequences, max_length=max_sequence_length)

    row_indices = np.arange(len(labeled_df))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    X_test = X[test_indices]
    y_test = y[test_indices]
    train_rally_indices, validation_rally_indices = split_train_validation_indices(train_indices, y, args.seed)
    X_train = X[train_rally_indices]
    y_train = y[train_rally_indices]
    X_validation = X[validation_rally_indices]
    y_validation = y[validation_rally_indices]
    X_train_multitask, y_train_multitask, train_multitask_weights, train_multitask_df = build_multitask_training_examples(
        rally_ids=labeled_df.iloc[train_rally_indices]["RallyId"].tolist(),
        sequences=[sequences[index] for index in train_rally_indices],
        outcomes=y[train_rally_indices],
        shot_to_id=shot_to_id,
        max_sequence_length=max_sequence_length,
        split_label="train",
    )
    X_validation_multitask, y_validation_multitask, validation_multitask_weights, validation_multitask_df = build_multitask_training_examples(
        rally_ids=labeled_df.iloc[validation_rally_indices]["RallyId"].tolist(),
        sequences=[sequences[index] for index in validation_rally_indices],
        outcomes=y[validation_rally_indices],
        shot_to_id=shot_to_id,
        max_sequence_length=max_sequence_length,
        split_label="validation",
    )

    X_test_prefix, y_test_prefix, _, test_prefix_df = build_prefix_training_examples(
        rally_ids=labeled_df.iloc[test_indices]["RallyId"].tolist(),
        sequences=[sequences[index] for index in test_indices],
        outcomes=y[test_indices],
        shot_to_id=shot_to_id,
        max_sequence_length=max_sequence_length,
        split_label="test",
    )
    X_test_next_shot, y_test_next_shot, test_next_shot_weights, test_next_shot_df = build_next_shot_training_examples(
        rally_ids=labeled_df.iloc[test_indices]["RallyId"].tolist(),
        sequences=[sequences[index] for index in test_indices],
        shot_to_id=shot_to_id,
        max_sequence_length=max_sequence_length,
        split_label="test",
    )

    training_model, analysis_model = build_attention_models(
        vocab_size=max(shot_to_id.values()),
        max_sequence_length=max_sequence_length,
        encoder_type=encoder_name,
    )
    validation_enabled = len(X_validation_multitask) > 0
    monitor_metric = "val_loss" if validation_enabled else "loss"
    callbacks = [EarlyStopping(monitor=monitor_metric, patience=8, restore_best_weights=True)]

    fit_kwargs: dict[str, object] = {
        "epochs": args.epochs,
        "batch_size": min(args.batch_size, max(1, len(X_train_multitask))),
        "verbose": 0,
        "callbacks": callbacks,
        "sample_weight": train_multitask_weights,
    }
    if validation_enabled:
        fit_kwargs["validation_data"] = (
            X_validation_multitask,
            y_validation_multitask,
            validation_multitask_weights,
        )

    history = training_model.fit(
        X_train_multitask,
        y_train_multitask,
        **fit_kwargs,
    )

    if validation_enabled:
        validation_probabilities_raw, _, _, _ = unpack_model_outputs(analysis_model.predict(X_validation, verbose=0))
        validation_probabilities_raw = validation_probabilities_raw.reshape(-1)
        win_probability_calibration = {
            "method": "temperature_scaling",
            **fit_binary_temperature_scaler(
                y_true=y_validation,
                probabilities=validation_probabilities_raw,
            ),
        }
    else:
        win_probability_calibration = {
            "method": "identity",
            "temperature": 1.0,
            "validation_log_loss_before": None,
            "validation_log_loss_after": None,
        }

    evaluation = training_model.evaluate(
        X_test,
        {
            "win_probability": y_test.astype(np.float32),
            "next_shot_distribution": np.zeros(len(X_test), dtype=np.int32),
        },
        sample_weight={
            "win_probability": np.ones(len(X_test), dtype=np.float32),
            "next_shot_distribution": np.zeros(len(X_test), dtype=np.float32),
        },
        verbose=0,
        return_dict=True,
    )
    test_probabilities, _, _, _ = unpack_model_outputs(analysis_model.predict(X_test, verbose=0))
    test_probabilities = calibrate_win_probabilities(
        test_probabilities.reshape(-1),
        {"win_probability_calibration": win_probability_calibration},
    )
    test_predictions = (test_probabilities >= 0.5).astype(int)
    if len(X_test_prefix):
        test_prefix_probabilities, _, _, _ = unpack_model_outputs(analysis_model.predict(X_test_prefix, verbose=0))
        test_prefix_probabilities = calibrate_win_probabilities(
            test_prefix_probabilities.reshape(-1),
            {"win_probability_calibration": win_probability_calibration},
        )
        test_prefix_predictions = (test_prefix_probabilities >= 0.5).astype(int)
    else:
        test_prefix_probabilities = np.empty((0,), dtype=float)
        test_prefix_predictions = np.empty((0,), dtype=int)

    if len(X_test_next_shot):
        _, _, _, test_next_shot_probabilities = unpack_model_outputs(analysis_model.predict(X_test_next_shot, verbose=0))
        if test_next_shot_probabilities is None:
            raise ValueError("The analysis model did not return next-shot probabilities.")
        test_next_shot_predictions = np.argmax(test_next_shot_probabilities, axis=1)
    else:
        test_next_shot_probabilities = np.empty((0, max(shot_to_id.values()) + 1), dtype=float)
        test_next_shot_predictions = np.empty((0,), dtype=int)

    overall_probabilities, overall_attention_weights, overall_context_embeddings, _ = unpack_model_outputs(
        analysis_model.predict(X, verbose=0)
    )
    overall_probabilities = calibrate_win_probabilities(
        overall_probabilities.reshape(-1),
        {"win_probability_calibration": win_probability_calibration},
    )

    cluster_assignments_df, cluster_summary_df, cluster_model = build_cluster_outputs(
        context_embeddings=np.asarray(overall_context_embeddings),
        rally_df=labeled_df,
        predicted_probabilities=overall_probabilities,
        cluster_count=args.clusters,
        random_state=args.seed,
    )
    cluster_name_map = {
        str(int(row["strategy_cluster"])): row["strategy_cluster_name"]
        for _, row in cluster_summary_df.iterrows()
    }

    sequence_lengths = [len(sequence) for sequence in sequences]
    trimmed_attention_rows = [
        np.asarray(attention_row[:sequence_length], dtype=float).tolist()
        for attention_row, sequence_length in zip(overall_attention_weights, sequence_lengths)
    ]

    attention_pattern_df = extract_high_impact_patterns(
        sequences=sequences,
        attention_weight_rows=trimmed_attention_rows,
        outcomes=y,
        predictions=overall_probabilities,
        min_attention_weight=0.12,
        pattern_sizes=(2, 3),
        min_count=2,
    )

    test_attention_rows = [
        trimmed_attention_rows[index]
        for index in test_indices
    ]
    top_attention_shots = [
        sequences[index][int(np.argmax(test_attention_rows[row_position]))]
        for row_position, index in enumerate(test_indices)
    ]

    test_predictions_df = labeled_df.iloc[test_indices][["RallyId", "ShotTypes", "Player1Won"]].copy()
    test_predictions_df["predicted_probability"] = np.round(test_probabilities, 4)
    test_predictions_df["predicted_label"] = test_predictions
    test_predictions_df["strategy_cluster"] = cluster_assignments_df.iloc[test_indices]["strategy_cluster"].to_numpy()
    test_predictions_df["strategy_cluster_name"] = cluster_assignments_df.iloc[test_indices]["strategy_cluster_name"].to_numpy()
    test_predictions_df["top_attention_shot"] = top_attention_shots
    test_predictions_df["attention_weights_json"] = [
        json.dumps([round(weight, 4) for weight in weights]) for weights in test_attention_rows
    ]

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(output_dir / "attention_history.csv", index=False)

    id_to_shot = {
        int(shot_id): shot
        for shot, shot_id in shot_to_id.items()
    }
    top3_indices = np.argsort(test_next_shot_probabilities, axis=1)[:, -3:][:, ::-1]
    top3_labels = [
        [id_to_shot.get(int(shot_id), "<UNK>") for shot_id in row]
        for row in top3_indices
    ]
    next_shot_hits = (test_next_shot_predictions == y_test_next_shot).astype(float)
    next_shot_top3_hits = np.asarray(
        [int(label in row) for label, row in zip(y_test_next_shot, top3_indices)],
        dtype=float,
    )
    test_next_shot_predictions_df = test_next_shot_df.copy()
    test_next_shot_predictions_df["predicted_next_shot_id"] = test_next_shot_predictions.astype(int)
    test_next_shot_predictions_df["predicted_next_shot"] = [
        id_to_shot.get(int(prediction), "<UNK>") for prediction in test_next_shot_predictions
    ]
    test_next_shot_predictions_df["predicted_next_shot_probability"] = np.round(
        test_next_shot_probabilities[np.arange(len(test_next_shot_predictions)), test_next_shot_predictions],
        4,
    )
    test_next_shot_predictions_df["true_next_shot_probability"] = np.round(
        test_next_shot_probabilities[np.arange(len(y_test_next_shot)), y_test_next_shot],
        4,
    )
    test_next_shot_predictions_df["top3_predicted_next_shots_json"] = [
        json.dumps(row) for row in top3_labels
    ]
    test_next_shot_predictions_df["next_shot_correct"] = next_shot_hits.astype(int)
    test_next_shot_predictions_df["next_shot_top3_correct"] = next_shot_top3_hits.astype(int)

    cluster_assignments_df.to_csv(output_dir / "strategy_cluster_assignments.csv", index=False)
    cluster_summary_df.to_csv(output_dir / "strategy_cluster_summary.csv", index=False)
    attention_pattern_df.to_csv(output_dir / "attention_pattern_stats.csv", index=False)
    context_transition_stats = build_context_transition_statistics(sequences, max_context_size=2)

    metadata = {
        "shot_to_id": shot_to_id,
        "id_to_shot": {str(shot_id): shot for shot_id, shot in id_to_shot.items()},
        "max_sequence_length": int(max_sequence_length),
        "seed": args.seed,
        "encoder_type": encoder_name,
        "next_shot_head_enabled": True,
        "win_probability_calibration": win_probability_calibration,
        "min_recommendation_delta": round(min_recommendation_delta, 4),
        "context_candidate_top_k": int(context_candidate_top_k),
        "max_context_size": 2,
        "context_transition_stats": context_transition_stats,
        "cluster_name_map": cluster_name_map,
        "candidate_shots": sorted([shot for shot in shot_to_id if not shot.startswith("<")]),
    }

    sequence_artifacts = SequenceIntelligenceArtifacts(
        output_dir=output_dir,
        model=analysis_model,
        metadata=metadata,
        summary={},
        cluster_model=cluster_model,
    )

    test_index_set = set(int(index) for index in test_indices)
    validation_index_set = set(int(index) for index in validation_rally_indices)
    split_labels = [
        "test" if index in test_index_set else "validation" if index in validation_index_set else "train"
        for index in range(len(labeled_df))
    ]
    training_dataset_df = labeled_df[["RallyId", "ShotTypes", "Player1Won", "ShotSeq"]].copy()
    training_dataset_df["ShotSeqJson"] = training_dataset_df["ShotSeq"].apply(json.dumps)
    training_dataset_df["EncodedSequenceJson"] = [json.dumps(sequence) for sequence in encoded_sequences]
    training_dataset_df["sequence_length"] = training_dataset_df["ShotSeq"].apply(len)
    training_dataset_df["dataset_split"] = split_labels
    training_dataset_df = training_dataset_df.drop(columns=["ShotSeq"])

    shot_impact_trajectory_df = build_shot_impact_trajectory_dataset(
        rally_ids=training_dataset_df["RallyId"].tolist(),
        sequences=sequences,
        outcomes=y,
        artifacts=sequence_artifacts,
        split_labels=split_labels,
    )

    test_roc_auc = safe_roc_auc(y_test, test_probabilities)
    test_prefix_roc_auc = safe_roc_auc(y_test_prefix, test_prefix_probabilities) if len(y_test_prefix) else None
    test_next_shot_accuracy = weighted_mean(next_shot_hits, test_next_shot_weights)
    test_next_shot_top3_accuracy = weighted_top_k_accuracy(
        y_true=y_test_next_shot,
        probabilities=test_next_shot_probabilities,
        weights=test_next_shot_weights,
        k=3,
    )
    summary = {
        "input_file": input_path.name,
        "model_type": model_type_name,
        "training_mode": "full_sequence_supervised + next_shot_auxiliary",
        "train_rows": int(len(train_rally_indices)),
        "validation_rows": int(len(validation_rally_indices)),
        "test_rows": int(len(X_test)),
        "vocab_size": int(max(shot_to_id.values())),
        "observed_shot_types": int(len(shot_to_id) - 1),
        "max_sequence_length": int(max_sequence_length),
        "min_recommendation_delta": round(min_recommendation_delta, 4),
        "context_candidate_top_k": int(context_candidate_top_k),
        "win_probability_calibration_method": win_probability_calibration.get("method"),
        "win_probability_temperature": win_probability_calibration.get("temperature"),
        "validation_log_loss_before_calibration": win_probability_calibration.get("validation_log_loss_before"),
        "validation_log_loss_after_calibration": win_probability_calibration.get("validation_log_loss_after"),
        "train_positive_rate": round(float(y[train_rally_indices].mean()), 4),
        "validation_positive_rate": round(float(y[validation_rally_indices].mean()), 4) if len(validation_rally_indices) else None,
        "test_positive_rate": round(float(y_test.mean()), 4),
        "test_loss": round(float(evaluation["loss"]), 4),
        "test_accuracy": round(float(accuracy_score(y_test, test_predictions)), 4),
        "test_roc_auc": round(test_roc_auc, 4) if test_roc_auc is not None else None,
        "test_prefix_accuracy": round(float(accuracy_score(y_test_prefix, test_prefix_predictions)), 4) if len(y_test_prefix) else None,
        "test_prefix_roc_auc": round(float(test_prefix_roc_auc), 4) if test_prefix_roc_auc is not None else None,
        "test_next_shot_rows": int(len(X_test_next_shot)),
        "test_next_shot_accuracy": round(float(test_next_shot_accuracy), 4) if test_next_shot_accuracy is not None else None,
        "test_next_shot_top3_accuracy": round(float(test_next_shot_top3_accuracy), 4) if test_next_shot_top3_accuracy is not None else None,
        "best_val_accuracy": round(float(max(history.history.get("val_win_probability_accuracy", [0.0]))), 4),
        "best_val_auc": round(float(max(history.history.get("val_win_probability_auc", [0.0]))), 4),
        "best_val_next_shot_accuracy": round(float(max(history.history.get("val_next_shot_distribution_next_shot_accuracy", [0.0]))), 4),
        "best_val_next_shot_top3_accuracy": round(float(max(history.history.get("val_next_shot_distribution_next_shot_top3_accuracy", [0.0]))), 4),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "top_attention_pattern": attention_pattern_df.iloc[0]["pattern"] if not attention_pattern_df.empty else None,
        "top_strategy_cluster": cluster_summary_df.sort_values(by="win_rate", ascending=False).iloc[0]["strategy_cluster_name"],
        "training_sequence_rows": int(len(training_dataset_df)),
        "train_multitask_rows": int(len(train_multitask_df)),
        "validation_multitask_rows": int(len(validation_multitask_df)),
        "test_prefix_rows": int(len(test_prefix_df)),
        "shot_impact_rows": int(len(shot_impact_trajectory_df)),
    }

    test_prefix_probability_map = {
        rally_id: json.dumps(group["win_probability_after_shot"].round(4).tolist())
        for rally_id, group in shot_impact_trajectory_df[shot_impact_trajectory_df["dataset_split"] == "test"].groupby("RallyId")
    }
    test_prefix_delta_map = {
        rally_id: json.dumps(group["delta_vs_previous_shot"].round(4).tolist())
        for rally_id, group in shot_impact_trajectory_df[shot_impact_trajectory_df["dataset_split"] == "test"].groupby("RallyId")
    }
    test_predictions_df["prefix_win_probabilities_json"] = test_predictions_df["RallyId"].map(test_prefix_probability_map)
    test_predictions_df["shot_impact_deltas_json"] = test_predictions_df["RallyId"].map(test_prefix_delta_map)

    (output_dir / "attention_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    analysis_model.save(output_dir / "attention_sequence_model.keras")
    joblib.dump(cluster_model, output_dir / "strategy_kmeans.joblib")

    training_dataset_df.to_csv(output_dir / "sequence_training_dataset.csv", index=False)
    legacy_prefix_dataset_path = output_dir / "sequence_prefix_training_dataset.csv"
    if legacy_prefix_dataset_path.exists():
        legacy_prefix_dataset_path.unlink()
    shot_impact_trajectory_df.to_csv(output_dir / "attention_shot_impact_trajectories.csv", index=False)
    test_predictions_df.to_csv(output_dir / "attention_test_predictions.csv", index=False)
    test_next_shot_predictions_df.to_csv(output_dir / "attention_next_shot_test_predictions.csv", index=False)
    print(f"Attention model training complete for {input_path.name}")
    print(f"Artifacts written to {output_dir.resolve()}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
