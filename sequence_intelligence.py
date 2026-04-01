from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Input, LSTM, Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from analyze_rallies import parse_shot_sequence


PAD_TOKEN_ID = 0
UNKNOWN_TOKEN = "<UNK>"
UNKNOWN_TOKEN_ID = 1
DEFAULT_AUXILIARY_LOSS_WEIGHT = 0.35
DEFAULT_MIN_RECOMMENDATION_DELTA = 0.01
DEFAULT_CONTEXT_CANDIDATE_TOP_K = 6


@tf.keras.utils.register_keras_serializable(package="player_analysis")
class TemporalAttention(Layer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape: tuple[int, ...]) -> None:
        feature_size = int(input_shape[-1])
        self.attention_kernel = self.add_weight(
            name="attention_kernel",
            shape=(feature_size, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.attention_bias = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        scores = tf.squeeze(tf.tanh(tf.matmul(inputs, self.attention_kernel) + self.attention_bias), axis=-1)
        if mask is not None:
            mask = tf.cast(mask, dtype=scores.dtype)
            scores += (1.0 - mask) * tf.constant(-1e9, dtype=scores.dtype)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * tf.expand_dims(weights, axis=-1), axis=1)
        return context, weights

    def compute_mask(self, inputs: tf.Tensor, mask: tf.Tensor | None = None) -> tuple[None, None]:
        return None, None

    def get_config(self) -> dict[str, Any]:
        return super().get_config()


@dataclass
class SequenceIntelligenceArtifacts:
    output_dir: Path
    model: Model
    metadata: dict[str, Any]
    summary: dict[str, Any]
    cluster_model: KMeans | None


def clip_probability(probabilities: np.ndarray | float, epsilon: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)


def binary_logit(probabilities: np.ndarray | float) -> np.ndarray:
    clipped = clip_probability(probabilities)
    return np.log(clipped / (1.0 - clipped))


def binary_sigmoid(logits: np.ndarray | float) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    return 1.0 / (1.0 + np.exp(-logits))


def apply_temperature_scaling(
    probabilities: np.ndarray | float,
    temperature: float,
) -> np.ndarray:
    resolved_temperature = max(float(temperature), 1e-3)
    return binary_sigmoid(binary_logit(probabilities) / resolved_temperature)


def binary_log_loss(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = clip_probability(probabilities)
    return float(-np.mean(y_true * np.log(probabilities) + (1.0 - y_true) * np.log(1.0 - probabilities)))


def fit_binary_temperature_scaler(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if y_true.size == 0 or probabilities.size == 0 or len(np.unique(y_true)) < 2:
        return {
            "temperature": 1.0,
            "validation_log_loss_before": binary_log_loss(y_true, probabilities) if y_true.size else 0.0,
            "validation_log_loss_after": binary_log_loss(y_true, probabilities) if y_true.size else 0.0,
        }

    candidate_temperatures = np.unique(
        np.concatenate(
            [
                np.array([1.0], dtype=float),
                np.linspace(0.5, 3.0, 101, dtype=float),
                np.logspace(np.log10(0.25), np.log10(4.0), 81, dtype=float),
            ]
        )
    )
    best_temperature = 1.0
    best_loss = binary_log_loss(y_true, probabilities)
    for temperature in candidate_temperatures:
        calibrated = apply_temperature_scaling(probabilities, float(temperature))
        loss = binary_log_loss(y_true, calibrated)
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)

    return {
        "temperature": round(best_temperature, 6),
        "validation_log_loss_before": round(binary_log_loss(y_true, probabilities), 6),
        "validation_log_loss_after": round(best_loss, 6),
    }


def calibrate_win_probabilities(
    probabilities: np.ndarray | float,
    metadata: dict[str, Any],
) -> np.ndarray:
    calibration = metadata.get("win_probability_calibration", {})
    method = calibration.get("method", "identity")
    if method != "temperature_scaling":
        return np.asarray(probabilities, dtype=float)
    temperature = float(calibration.get("temperature", 1.0))
    return apply_temperature_scaling(probabilities, temperature)


def build_context_key(shots: list[str]) -> str:
    return " -> ".join(shots) if shots else "<START>"


def build_context_transition_statistics(
    sequences: list[list[str]],
    max_context_size: int = 2,
) -> dict[str, dict[str, dict[str, int]]]:
    stats: dict[str, dict[str, dict[str, int]]] = {
        str(context_size): {}
        for context_size in range(max(0, max_context_size) + 1)
    }

    for sequence in sequences:
        for position, next_shot in enumerate(sequence):
            prefix = sequence[:position]
            for context_size in range(max(0, max_context_size) + 1):
                context = prefix[-context_size:] if context_size > 0 else []
                context_key = build_context_key(context)
                context_stats = stats[str(context_size)].setdefault(context_key, {})
                context_stats[next_shot] = int(context_stats.get(next_shot, 0)) + 1

    return stats


def invert_vocabulary(shot_to_id: dict[str, int]) -> dict[int, str]:
    return {shot_id: shot for shot, shot_id in shot_to_id.items()}


def unpack_model_outputs(
    outputs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if not isinstance(outputs, (list, tuple)):
        raise ValueError("Unexpected model output format.")

    if len(outputs) == 4:
        win_probability, attention_weights, context_vector, next_shot_distribution = outputs
        return (
            np.asarray(win_probability),
            np.asarray(attention_weights),
            np.asarray(context_vector),
            np.asarray(next_shot_distribution),
        )

    if len(outputs) == 3:
        win_probability, attention_weights, context_vector = outputs
        return (
            np.asarray(win_probability),
            np.asarray(attention_weights),
            np.asarray(context_vector),
            None,
        )

    raise ValueError(f"Unexpected number of model outputs: {len(outputs)}")


def build_vocabulary(sequences: list[list[str]]) -> dict[str, int]:
    shots = sorted({shot for sequence in sequences for shot in sequence})
    shot_to_id = {UNKNOWN_TOKEN: UNKNOWN_TOKEN_ID}
    for index, shot in enumerate(shots, start=UNKNOWN_TOKEN_ID + 1):
        shot_to_id[shot] = index
    return shot_to_id


def encode_sequences(sequences: list[list[str]], shot_to_id: dict[str, int]) -> list[list[int]]:
    return [
        [shot_to_id.get(shot, UNKNOWN_TOKEN_ID) for shot in sequence]
        for sequence in sequences
    ]


def pad_encoded_sequences(encoded_sequences: list[list[int]], max_length: int | None = None) -> np.ndarray:
    resolved_max_length = max_length or max((len(sequence) for sequence in encoded_sequences), default=1)
    return pad_sequences(
        encoded_sequences,
        maxlen=resolved_max_length,
        padding="post",
        truncating="post",
        value=PAD_TOKEN_ID,
    )


def build_attention_models(
    vocab_size: int,
    max_sequence_length: int,
    embedding_dim: int = 32,
    lstm_units: int = 48,
    encoder_type: str = "bilstm",
    auxiliary_loss_weight: float = DEFAULT_AUXILIARY_LOSS_WEIGHT,
) -> tuple[Model, Model]:
    sequence_input = Input(shape=(max_sequence_length,), name="shot_sequence")
    embedded = Embedding(
        input_dim=vocab_size + 1,
        output_dim=embedding_dim,
        mask_zero=True,
        name="shot_embedding",
    )(sequence_input)
    normalized_encoder_type = encoder_type.strip().lower()
    if normalized_encoder_type == "lstm":
        sequence_features = LSTM(
            lstm_units,
            return_sequences=True,
            name="shot_encoder",
        )(embedded)
    elif normalized_encoder_type == "bilstm":
        sequence_features = Bidirectional(
            LSTM(lstm_units, return_sequences=True),
            name="shot_encoder",
        )(embedded)
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")
    context_vector, attention_weights = TemporalAttention(name="temporal_attention")(sequence_features)
    hidden = Dense(32, activation="relu", name="tactic_projection")(context_vector)
    win_probability = Dense(1, activation="sigmoid", name="win_probability")(hidden)
    next_shot_distribution = Dense(
        vocab_size + 1,
        activation="softmax",
        name="next_shot_distribution",
    )(hidden)

    training_model = Model(
        sequence_input,
        {
            "win_probability": win_probability,
            "next_shot_distribution": next_shot_distribution,
        },
        name="attention_sequence_model",
    )
    training_model.compile(
        loss={
            "win_probability": "binary_crossentropy",
            "next_shot_distribution": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={
            "win_probability": 1.0,
            "next_shot_distribution": auxiliary_loss_weight,
        },
        optimizer="adam",
        metrics={
            "win_probability": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
            "next_shot_distribution": [
                tf.keras.metrics.SparseCategoricalAccuracy(name="next_shot_accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="next_shot_top3_accuracy"),
            ],
        },
    )

    analysis_model = Model(
        sequence_input,
        [win_probability, attention_weights, context_vector, next_shot_distribution],
        name="attention_analysis_model",
    )
    return training_model, analysis_model


def decode_shot_sequence(raw_sequence: str | list[str]) -> list[str]:
    if isinstance(raw_sequence, list):
        return raw_sequence
    return parse_shot_sequence(raw_sequence)


def encode_single_sequence(shots: list[str], shot_to_id: dict[str, int], max_length: int) -> np.ndarray:
    encoded = encode_sequences([shots], shot_to_id)
    return pad_encoded_sequences(encoded, max_length=max_length)


def load_sequence_intelligence_artifacts(output_dir: Path = Path("outputs")) -> SequenceIntelligenceArtifacts | None:
    metadata_path = output_dir / "attention_metadata.json"
    summary_path = output_dir / "attention_summary.json"
    model_path = output_dir / "attention_sequence_model.keras"
    cluster_model_path = output_dir / "strategy_kmeans.joblib"

    if not metadata_path.exists() or not summary_path.exists() or not model_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    model = load_model(model_path, compile=False)
    cluster_model = joblib.load(cluster_model_path) if cluster_model_path.exists() else None

    return SequenceIntelligenceArtifacts(
        output_dir=output_dir,
        model=model,
        metadata=metadata,
        summary=summary,
        cluster_model=cluster_model,
    )


def _build_sequence_intelligence_result(
    shots: list[str],
    shot_to_id: dict[str, int],
    metadata: dict[str, Any],
    raw_probability: float,
    calibrated_probability: float,
    attention_row: np.ndarray,
    context_vector: np.ndarray,
    next_shot_distribution_row: np.ndarray | None,
    strategy_cluster_id: int | None,
    strategy_cluster_name: str | None,
) -> dict[str, Any]:
    max_sequence_length = int(metadata["max_sequence_length"])
    actual_length = min(len(shots), max_sequence_length)
    weights = np.asarray(attention_row[:actual_length], dtype=float).tolist()
    context = np.asarray(context_vector, dtype=float).tolist()

    next_shot_probability_df = build_next_shot_probability_table(
        next_shot_distribution=next_shot_distribution_row,
        shot_to_id=shot_to_id,
        candidate_shots=metadata.get("candidate_shots"),
    )
    next_shot_probability_map = (
        dict(zip(next_shot_probability_df["candidate_shot"], next_shot_probability_df["next_shot_probability"]))
        if not next_shot_probability_df.empty
        else {}
    )

    shot_rows = []
    for index, (shot, weight) in enumerate(zip(shots[:actual_length], weights), start=1):
        shot_rows.append(
            {
                "position": index,
                "shot": shot,
                "attention_weight": round(float(weight), 4),
            }
        )
    if shot_rows:
        shot_importance_df = pd.DataFrame(shot_rows).sort_values(
            by=["attention_weight", "position"], ascending=[False, True]
        )
    else:
        shot_importance_df = pd.DataFrame(columns=["position", "shot", "attention_weight"])

    unknown_shots = [shot for shot in shots if shot not in shot_to_id]

    return {
        "win_probability": float(calibrated_probability),
        "raw_win_probability": float(raw_probability),
        "attention_weights": weights,
        "context_embedding": context,
        "shot_importance_df": shot_importance_df,
        "strategy_cluster_id": strategy_cluster_id,
        "strategy_cluster_name": strategy_cluster_name,
        "unknown_shots": unknown_shots,
        "next_shot_probability_df": next_shot_probability_df,
        "next_shot_probability_map": next_shot_probability_map,
    }


def build_next_shot_probability_table(
    next_shot_distribution: np.ndarray | None,
    shot_to_id: dict[str, int],
    candidate_shots: list[str] | None = None,
) -> pd.DataFrame:
    if next_shot_distribution is None:
        return pd.DataFrame(columns=["candidate_shot", "next_shot_probability"])

    resolved_candidates = candidate_shots or sorted(
        [shot for shot in shot_to_id if not shot.startswith("<")]
    )
    rows = []
    for shot in resolved_candidates:
        shot_id = shot_to_id.get(shot)
        if shot_id is None or shot_id >= len(next_shot_distribution):
            continue
        rows.append(
            {
                "candidate_shot": shot,
                "next_shot_probability": round(float(next_shot_distribution[shot_id]), 4),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["candidate_shot", "next_shot_probability"])

    return pd.DataFrame(rows).sort_values(
        by=["next_shot_probability", "candidate_shot"],
        ascending=[False, True],
    ).reset_index(drop=True)


def generate_context_aware_candidate_shots(
    shots: list[str],
    artifacts: SequenceIntelligenceArtifacts,
    top_k: int | None = None,
    exclude_shots: set[str] | None = None,
    base_result: dict[str, Any] | None = None,
) -> list[str]:
    resolved_top_k = int(
        top_k
        or artifacts.metadata.get("context_candidate_top_k", DEFAULT_CONTEXT_CANDIDATE_TOP_K)
    )
    resolved_top_k = max(1, resolved_top_k)
    excluded = set(exclude_shots or set())
    candidate_scores: dict[str, float] = {}

    context_stats = artifacts.metadata.get("context_transition_stats", {})
    max_context_size = int(artifacts.metadata.get("max_context_size", 2))
    for context_size in range(min(max_context_size, len(shots)), -1, -1):
        context_key = build_context_key(shots[-context_size:] if context_size > 0 else [])
        shot_counts = context_stats.get(str(context_size), {}).get(context_key, {})
        total_count = float(sum(int(count) for count in shot_counts.values()))
        if total_count <= 0:
            continue
        context_weight = float(context_size + 1)
        for shot, count in shot_counts.items():
            if shot in excluded:
                continue
            candidate_scores[shot] = candidate_scores.get(shot, 0.0) + context_weight * (float(count) / total_count)

    resolved_base_result = base_result if base_result is not None else predict_sequence_intelligence(shots, artifacts)
    next_shot_probability_map = resolved_base_result.get("next_shot_probability_map", {})
    for shot, probability in next_shot_probability_map.items():
        if shot in excluded:
            continue
        candidate_scores[shot] = candidate_scores.get(shot, 0.0) + (2.0 * float(probability))

    fallback_candidates = artifacts.metadata.get("candidate_shots", [])
    for fallback_index, shot in enumerate(fallback_candidates):
        if shot in excluded or shot.startswith("<"):
            continue
        candidate_scores.setdefault(shot, max(0.001, 0.01 - (fallback_index * 0.0001)))

    ordered_candidates = [
        shot
        for shot, _ in sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return ordered_candidates[:resolved_top_k]


def predict_sequence_intelligence(
    shots: list[str],
    artifacts: SequenceIntelligenceArtifacts,
) -> dict[str, Any]:
    return predict_sequence_intelligence_batch([shots], artifacts)[0]


def predict_sequence_intelligence_batch(
    sequences: list[list[str]],
    artifacts: SequenceIntelligenceArtifacts,
) -> list[dict[str, Any]]:
    if not sequences:
        return []

    shot_to_id = artifacts.metadata["shot_to_id"]
    max_sequence_length = int(artifacts.metadata["max_sequence_length"])
    encoded_sequences = encode_sequences(sequences, shot_to_id)
    sequence_inputs = pad_encoded_sequences(encoded_sequences, max_length=max_sequence_length)
    predictions, attention_weights, context_vectors, next_shot_distributions = unpack_model_outputs(
        artifacts.model.predict(sequence_inputs, verbose=0)
    )

    raw_probabilities = np.asarray(predictions, dtype=float).reshape(-1)
    calibrated_probabilities = np.asarray(
        calibrate_win_probabilities(raw_probabilities, artifacts.metadata),
        dtype=float,
    ).reshape(-1)

    strategy_cluster_ids: list[int | None] = [None] * len(sequences)
    strategy_cluster_names: list[str | None] = [None] * len(sequences)
    if artifacts.cluster_model is not None:
        cluster_name_map = artifacts.metadata.get("cluster_name_map", {})
        cluster_ids = artifacts.cluster_model.predict(np.asarray(context_vectors, dtype=np.float64))
        strategy_cluster_ids = [int(cluster_id) for cluster_id in cluster_ids]
        strategy_cluster_names = [
            cluster_name_map.get(str(cluster_id), f"Cluster {cluster_id}")
            for cluster_id in strategy_cluster_ids
        ]

    results: list[dict[str, Any]] = []
    for index, shots in enumerate(sequences):
        next_shot_distribution_row = (
            None
            if next_shot_distributions is None
            else np.asarray(next_shot_distributions[index], dtype=float)
        )
        results.append(
            _build_sequence_intelligence_result(
                shots=shots,
                shot_to_id=shot_to_id,
                metadata=artifacts.metadata,
                raw_probability=float(raw_probabilities[index]),
                calibrated_probability=float(calibrated_probabilities[index]),
                attention_row=np.asarray(attention_weights[index], dtype=float),
                context_vector=np.asarray(context_vectors[index], dtype=float),
                next_shot_distribution_row=next_shot_distribution_row,
                strategy_cluster_id=strategy_cluster_ids[index],
                strategy_cluster_name=strategy_cluster_names[index],
            )
        )

    return results


def predict_prefix_probability_trajectory(
    shots: list[str],
    artifacts: SequenceIntelligenceArtifacts,
) -> pd.DataFrame:
    if not shots:
        return pd.DataFrame(
            columns=[
                "position",
                "shot",
                "prefix_sequence",
                "win_probability_after_shot",
                "delta_vs_previous_shot",
                "cumulative_delta_vs_first_shot",
            ]
        )

    shot_to_id = artifacts.metadata["shot_to_id"]
    max_sequence_length = int(artifacts.metadata["max_sequence_length"])
    prefixes = [shots[:index] for index in range(1, len(shots) + 1)]
    encoded_prefixes = encode_sequences(prefixes, shot_to_id)
    prefix_inputs = pad_encoded_sequences(encoded_prefixes, max_length=max_sequence_length)
    predictions, _, _, _ = unpack_model_outputs(artifacts.model.predict(prefix_inputs, verbose=0))
    predictions = calibrate_win_probabilities(predictions.reshape(-1), artifacts.metadata)

    rows = []
    first_probability = float(predictions[0])
    previous_probability = None
    for index, (shot, prefix, probability) in enumerate(zip(shots, prefixes, predictions), start=1):
        probability = float(probability)
        delta_vs_previous = 0.0 if previous_probability is None else probability - previous_probability
        rows.append(
            {
                "position": index,
                "shot": shot,
                "prefix_sequence": " -> ".join(prefix),
                "win_probability_after_shot": round(probability, 4),
                "delta_vs_previous_shot": round(delta_vs_previous, 4),
                "cumulative_delta_vs_first_shot": round(probability - first_probability, 4),
            }
        )
        previous_probability = probability

    return pd.DataFrame(rows)


def build_shot_impact_trajectory_dataset(
    rally_ids: list[str],
    sequences: list[list[str]],
    outcomes: np.ndarray,
    artifacts: SequenceIntelligenceArtifacts,
    split_labels: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    resolved_split_labels = split_labels or ["all"] * len(sequences)

    for rally_id, sequence, outcome, split_label in zip(rally_ids, sequences, outcomes, resolved_split_labels):
        trajectory_df = predict_prefix_probability_trajectory(sequence, artifacts)
        if trajectory_df.empty:
            continue
        trajectory_df.insert(0, "dataset_split", split_label)
        trajectory_df.insert(0, "Player1Won", int(outcome))
        trajectory_df.insert(0, "RallyId", rally_id)
        rows.append(trajectory_df)

    if not rows:
        return pd.DataFrame(
            columns=[
                "RallyId",
                "Player1Won",
                "dataset_split",
                "position",
                "shot",
                "prefix_sequence",
                "win_probability_after_shot",
                "delta_vs_previous_shot",
                "cumulative_delta_vs_first_shot",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def build_prefix_training_examples(
    rally_ids: list[str],
    sequences: list[list[str]],
    outcomes: np.ndarray,
    shot_to_id: dict[str, int],
    max_sequence_length: int,
    split_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    encoded_prefixes: list[list[int]] = []
    labels: list[int] = []
    sample_weights: list[float] = []
    rows: list[dict[str, Any]] = []

    for rally_id, sequence, outcome in zip(rally_ids, sequences, outcomes):
        if not sequence:
            continue
        rally_weight = 1.0 / len(sequence)
        for position in range(1, len(sequence) + 1):
            prefix = sequence[:position]
            encoded_prefix = [shot_to_id.get(shot, UNKNOWN_TOKEN_ID) for shot in prefix]
            encoded_prefixes.append(encoded_prefix)
            labels.append(int(outcome))
            sample_weights.append(rally_weight)
            rows.append(
                {
                    "RallyId": rally_id,
                    "dataset_split": split_label,
                    "prefix_position": position,
                    "prefix_length": len(prefix),
                    "prefix_sequence": " -> ".join(prefix),
                    "PrefixSeqJson": json.dumps(prefix),
                    "EncodedPrefixJson": json.dumps(encoded_prefix),
                    "Player1Won": int(outcome),
                    "sample_weight": round(rally_weight, 6),
                }
            )

    if not encoded_prefixes:
        empty_df = pd.DataFrame(
            columns=[
                "RallyId",
                "dataset_split",
                "prefix_position",
                "prefix_length",
                "prefix_sequence",
                "PrefixSeqJson",
                "EncodedPrefixJson",
                "Player1Won",
                "sample_weight",
            ]
        )
        return (
            np.empty((0, max_sequence_length), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            empty_df,
        )

    X = pad_encoded_sequences(encoded_prefixes, max_length=max_sequence_length)
    y = np.asarray(labels, dtype=np.int32)
    weights = np.asarray(sample_weights, dtype=np.float32)
    prefix_df = pd.DataFrame(rows)
    return X, y, weights, prefix_df


def build_next_shot_training_examples(
    rally_ids: list[str],
    sequences: list[list[str]],
    shot_to_id: dict[str, int],
    max_sequence_length: int,
    split_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    encoded_prefixes: list[list[int]] = []
    next_shot_labels: list[int] = []
    sample_weights: list[float] = []
    rows: list[dict[str, Any]] = []

    for rally_id, sequence in zip(rally_ids, sequences):
        if not sequence:
            continue
        rally_weight = 1.0 / len(sequence)
        for position, next_shot in enumerate(sequence):
            prefix = sequence[:position]
            encoded_prefix = [shot_to_id.get(shot, UNKNOWN_TOKEN_ID) for shot in prefix]
            next_shot_id = shot_to_id.get(next_shot, UNKNOWN_TOKEN_ID)
            encoded_prefixes.append(encoded_prefix)
            next_shot_labels.append(next_shot_id)
            sample_weights.append(rally_weight)
            rows.append(
                {
                    "RallyId": rally_id,
                    "dataset_split": split_label,
                    "prefix_position": position,
                    "prefix_length": len(prefix),
                    "prefix_sequence": " -> ".join(prefix) if prefix else "<START>",
                    "PrefixSeqJson": json.dumps(prefix),
                    "EncodedPrefixJson": json.dumps(encoded_prefix),
                    "next_shot": next_shot,
                    "next_shot_id": int(next_shot_id),
                    "sample_weight": round(rally_weight, 6),
                }
            )

    if not encoded_prefixes:
        empty_df = pd.DataFrame(
            columns=[
                "RallyId",
                "dataset_split",
                "prefix_position",
                "prefix_length",
                "prefix_sequence",
                "PrefixSeqJson",
                "EncodedPrefixJson",
                "next_shot",
                "next_shot_id",
                "sample_weight",
            ]
        )
        return (
            np.empty((0, max_sequence_length), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            empty_df,
        )

    X = pad_encoded_sequences(encoded_prefixes, max_length=max_sequence_length)
    y_next = np.asarray(next_shot_labels, dtype=np.int32)
    weights = np.asarray(sample_weights, dtype=np.float32)
    next_shot_df = pd.DataFrame(rows)
    return X, y_next, weights, next_shot_df


def build_multitask_training_examples(
    rally_ids: list[str],
    sequences: list[list[str]],
    outcomes: np.ndarray,
    shot_to_id: dict[str, int],
    max_sequence_length: int,
    split_label: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    encoded_inputs: list[list[int]] = []
    win_labels: list[int] = []
    next_shot_labels: list[int] = []
    win_sample_weights: list[float] = []
    next_shot_sample_weights: list[float] = []
    rows: list[dict[str, Any]] = []

    for rally_id, sequence, outcome in zip(rally_ids, sequences, outcomes):
        if not sequence:
            continue

        encoded_sequence = [shot_to_id.get(shot, UNKNOWN_TOKEN_ID) for shot in sequence]
        encoded_inputs.append(encoded_sequence)
        win_labels.append(int(outcome))
        next_shot_labels.append(PAD_TOKEN_ID)
        win_sample_weights.append(1.0)
        next_shot_sample_weights.append(0.0)
        rows.append(
            {
                "RallyId": rally_id,
                "dataset_split": split_label,
                "example_type": "full_sequence",
                "prefix_position": len(sequence),
                "prefix_length": len(sequence),
                "prefix_sequence": " -> ".join(sequence),
                "PrefixSeqJson": json.dumps(sequence),
                "EncodedPrefixJson": json.dumps(encoded_sequence),
                "Player1Won": int(outcome),
                "next_shot": None,
                "next_shot_id": None,
                "win_sample_weight": 1.0,
                "next_shot_sample_weight": 0.0,
            }
        )

        next_shot_weight = 1.0 / len(sequence)
        for position, next_shot in enumerate(sequence):
            prefix = sequence[:position]
            encoded_prefix = [shot_to_id.get(shot, UNKNOWN_TOKEN_ID) for shot in prefix]
            next_shot_id = shot_to_id.get(next_shot, UNKNOWN_TOKEN_ID)
            encoded_inputs.append(encoded_prefix)
            win_labels.append(0)
            next_shot_labels.append(next_shot_id)
            win_sample_weights.append(0.0)
            next_shot_sample_weights.append(next_shot_weight)
            rows.append(
                {
                    "RallyId": rally_id,
                    "dataset_split": split_label,
                    "example_type": "next_shot_prefix",
                    "prefix_position": position,
                    "prefix_length": len(prefix),
                    "prefix_sequence": " -> ".join(prefix) if prefix else "<START>",
                    "PrefixSeqJson": json.dumps(prefix),
                    "EncodedPrefixJson": json.dumps(encoded_prefix),
                    "Player1Won": int(outcome),
                    "next_shot": next_shot,
                    "next_shot_id": int(next_shot_id),
                    "win_sample_weight": 0.0,
                    "next_shot_sample_weight": round(next_shot_weight, 6),
                }
            )

    if not encoded_inputs:
        empty_df = pd.DataFrame(
            columns=[
                "RallyId",
                "dataset_split",
                "example_type",
                "prefix_position",
                "prefix_length",
                "prefix_sequence",
                "PrefixSeqJson",
                "EncodedPrefixJson",
                "Player1Won",
                "next_shot",
                "next_shot_id",
                "win_sample_weight",
                "next_shot_sample_weight",
            ]
        )
        empty_targets = {
            "win_probability": np.empty((0,), dtype=np.float32),
            "next_shot_distribution": np.empty((0,), dtype=np.int32),
        }
        empty_weights = {
            "win_probability": np.empty((0,), dtype=np.float32),
            "next_shot_distribution": np.empty((0,), dtype=np.float32),
        }
        return (
            np.empty((0, max_sequence_length), dtype=np.int32),
            empty_targets,
            empty_weights,
            empty_df,
        )

    X = pad_encoded_sequences(encoded_inputs, max_length=max_sequence_length)
    targets = {
        "win_probability": np.asarray(win_labels, dtype=np.float32),
        "next_shot_distribution": np.asarray(next_shot_labels, dtype=np.int32),
    }
    sample_weights = {
        "win_probability": np.asarray(win_sample_weights, dtype=np.float32),
        "next_shot_distribution": np.asarray(next_shot_sample_weights, dtype=np.float32),
    }
    multitask_df = pd.DataFrame(rows)
    return X, targets, sample_weights, multitask_df


def simulate_shot_replacements(
    shots: list[str],
    artifacts: SequenceIntelligenceArtifacts,
    candidate_shots: list[str] | None = None,
    candidate_shots_by_position: dict[int, list[str]] | None = None,
) -> pd.DataFrame:
    if not shots:
        return pd.DataFrame(
            columns=[
                "position",
                "original_shot",
                "candidate_shot",
                "predicted_win_probability",
                "delta_vs_current",
                "candidate_shot_plausibility",
                "replacement_score",
            ]
        )

    baseline_result = predict_sequence_intelligence(shots, artifacts)
    baseline_probability = baseline_result["win_probability"]
    prefix_results = predict_sequence_intelligence_batch(
        [shots[:index] for index in range(len(shots))],
        artifacts,
    )

    simulation_rows: list[dict[str, Any]] = []
    simulated_sequences: list[list[str]] = []
    for index, original_shot in enumerate(shots):
        prefix_result = prefix_results[index]
        next_shot_probability_map = prefix_result.get("next_shot_probability_map", {})
        resolved_candidates = None
        if candidate_shots_by_position is not None:
            resolved_candidates = candidate_shots_by_position.get(index + 1)
        if resolved_candidates is None:
            resolved_candidates = candidate_shots
        if resolved_candidates is None:
            resolved_candidates = generate_context_aware_candidate_shots(
                shots=shots[:index],
                artifacts=artifacts,
                exclude_shots={"Fall", "Miss", "Out", original_shot},
                base_result=prefix_result,
            )
        for candidate_shot in resolved_candidates:
            if candidate_shot == original_shot:
                continue
            simulated = shots[:]
            simulated[index] = candidate_shot
            simulated_sequences.append(simulated)
            simulation_rows.append(
                {
                    "position": index + 1,
                    "original_shot": original_shot,
                    "candidate_shot": candidate_shot,
                    "candidate_shot_plausibility": round(float(next_shot_probability_map.get(candidate_shot, 0.0)), 4),
                }
            )

    if not simulation_rows:
        return pd.DataFrame(
            columns=[
                "position",
                "original_shot",
                "candidate_shot",
                "predicted_win_probability",
                "delta_vs_current",
                "candidate_shot_plausibility",
                "replacement_score",
            ]
        )

    simulated_results = predict_sequence_intelligence_batch(simulated_sequences, artifacts)
    rows = []
    for row, simulated_result in zip(simulation_rows, simulated_results):
        probability = float(simulated_result["win_probability"])
        candidate_plausibility = float(row["candidate_shot_plausibility"])
        rows.append(
            {
                **row,
                "predicted_win_probability": round(probability, 4),
                "delta_vs_current": round(probability - baseline_probability, 4),
                "replacement_score": round(probability * candidate_plausibility, 4),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["replacement_score", "delta_vs_current", "predicted_win_probability"],
        ascending=[False, False, False],
    )


def simulate_next_shot_candidates(
    shots: list[str],
    artifacts: SequenceIntelligenceArtifacts,
    candidate_shots: list[str] | None = None,
) -> pd.DataFrame:
    baseline_result = predict_sequence_intelligence(shots, artifacts)
    baseline_probability = baseline_result["win_probability"]
    next_shot_probability_map = baseline_result.get("next_shot_probability_map", {})
    resolved_candidate_shots = candidate_shots or generate_context_aware_candidate_shots(
        shots=shots,
        artifacts=artifacts,
        exclude_shots={"Fall", "Miss", "Out"},
        base_result=baseline_result,
    )

    if not resolved_candidate_shots:
        return pd.DataFrame(
            columns=[
                "candidate_shot",
                "predicted_win_probability",
                "delta_vs_current",
                "next_shot_probability",
                "recommendation_score",
            ]
        )

    simulated_results = predict_sequence_intelligence_batch(
        [shots + [candidate_shot] for candidate_shot in resolved_candidate_shots],
        artifacts,
    )
    rows = []
    for candidate_shot, simulated_result in zip(resolved_candidate_shots, simulated_results):
        probability = float(simulated_result["win_probability"])
        next_shot_probability = float(next_shot_probability_map.get(candidate_shot, 0.0))
        rows.append(
            {
                "candidate_shot": candidate_shot,
                "predicted_win_probability": round(probability, 4),
                "delta_vs_current": round(probability - baseline_probability, 4),
                "next_shot_probability": round(next_shot_probability, 4),
                "recommendation_score": round(probability * next_shot_probability, 4),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["recommendation_score", "delta_vs_current", "predicted_win_probability"],
        ascending=[False, False, False],
    )


def extract_high_impact_patterns(
    sequences: list[list[str]],
    attention_weight_rows: list[list[float]],
    outcomes: np.ndarray,
    predictions: np.ndarray,
    min_attention_weight: float = 0.12,
    pattern_sizes: tuple[int, ...] = (2, 3),
    min_count: int = 2,
) -> pd.DataFrame:
    pattern_stats: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    rally_count = len(sequences)

    for sequence, attention_weights, outcome, prediction in zip(
        sequences,
        attention_weight_rows,
        outcomes,
        predictions,
    ):
        if not sequence:
            continue

        attention = np.asarray(attention_weights[: len(sequence)], dtype=float)
        if attention.size == 0:
            continue
        dynamic_threshold = max(min_attention_weight, float(np.quantile(attention, 0.75)))
        important_positions = {index for index, weight in enumerate(attention) if float(weight) >= dynamic_threshold}
        seen_in_rally: set[tuple[int, tuple[str, ...]]] = set()

        for pattern_size in pattern_sizes:
            if len(sequence) < pattern_size:
                continue
            for start in range(len(sequence) - pattern_size + 1):
                indices = set(range(start, start + pattern_size))
                if not important_positions.intersection(indices):
                    continue
                pattern = tuple(sequence[start : start + pattern_size])
                key = (pattern_size, pattern)
                stats = pattern_stats.setdefault(
                    key,
                    {
                        "count": 0,
                        "rally_count": 0,
                        "wins": 0,
                        "prediction_sum": 0.0,
                        "attention_sum": 0.0,
                    },
                )
                stats["count"] += 1
                stats["wins"] += int(outcome)
                stats["prediction_sum"] += float(prediction)
                stats["attention_sum"] += float(attention[start : start + pattern_size].mean())
                if key not in seen_in_rally:
                    stats["rally_count"] += 1
                    seen_in_rally.add(key)

    rows = []
    for (pattern_size, pattern), stats in pattern_stats.items():
        if int(stats["count"]) < min_count:
            continue
        count = int(stats["count"])
        rally_support = int(stats["rally_count"])
        win_rate = float(stats["wins"]) / count
        rows.append(
            {
                "pattern_size": pattern_size,
                "pattern": " -> ".join(pattern),
                "count": count,
                "rally_count": rally_support,
                "win_rate": round(win_rate, 4),
                "avg_predicted_win_probability": round(float(stats["prediction_sum"]) / count, 4),
                "avg_attention_weight": round(float(stats["attention_sum"]) / count, 4),
                "rally_support_pct": round(rally_support / max(rally_count, 1), 4),
            }
        )

    pattern_df = pd.DataFrame(rows)
    if pattern_df.empty:
        return pd.DataFrame(
            columns=[
                "pattern_size",
                "pattern",
                "count",
                "rally_count",
                "win_rate",
                "avg_predicted_win_probability",
                "avg_attention_weight",
                "rally_support_pct",
            ]
        )

    return pattern_df.sort_values(
        by=["avg_attention_weight", "win_rate", "count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def assign_cluster_names(cluster_summary_df: pd.DataFrame) -> dict[int, str]:
    if cluster_summary_df.empty:
        return {}

    if len(cluster_summary_df) == 1:
        return {int(cluster_summary_df.iloc[0]["strategy_cluster"]): "Balanced"}

    ordered = cluster_summary_df.sort_values(
        by=["avg_attack_ratio", "avg_defense_ratio"],
        ascending=[False, True],
    )
    cluster_ids = ordered["strategy_cluster"].astype(int).tolist()

    name_map: dict[int, str] = {}
    if len(cluster_ids) >= 1:
        name_map[cluster_ids[0]] = "Aggressive"
    if len(cluster_ids) >= 2:
        name_map[cluster_ids[-1]] = "Defensive"
    middle_ids = [cluster_id for cluster_id in cluster_ids if cluster_id not in name_map]
    for middle_index, cluster_id in enumerate(middle_ids, start=1):
        name_map[cluster_id] = "Balanced" if len(middle_ids) == 1 else f"Balanced {middle_index}"
    return name_map


def build_cluster_outputs(
    context_embeddings: np.ndarray,
    rally_df: pd.DataFrame,
    predicted_probabilities: np.ndarray,
    cluster_count: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    context_embeddings = np.asarray(context_embeddings, dtype=np.float64)
    effective_cluster_count = max(1, min(cluster_count, len(rally_df)))
    cluster_model = KMeans(n_clusters=effective_cluster_count, random_state=random_state, n_init=20)
    cluster_ids = cluster_model.fit_predict(context_embeddings)

    cluster_df = rally_df.copy()
    cluster_df["strategy_cluster"] = cluster_ids.astype(int)
    cluster_df["sequence_model_probability"] = np.round(predicted_probabilities.astype(float), 4)

    summary_df = (
        cluster_df.groupby("strategy_cluster", as_index=False)
        .agg(
            rallies=("RallyId", "count"),
            wins=("Player1Won", "sum"),
            avg_attack_ratio=("attack_ratio", "mean"),
            avg_defense_ratio=("defense_ratio", "mean"),
            avg_neutral_ratio=("neutral_ratio", "mean"),
            avg_sequence_model_probability=("sequence_model_probability", "mean"),
        )
        .sort_values(by="rallies", ascending=False)
        .reset_index(drop=True)
    )
    summary_df["win_rate"] = (summary_df["wins"] / summary_df["rallies"]).round(4)
    for column in [
        "avg_attack_ratio",
        "avg_defense_ratio",
        "avg_neutral_ratio",
        "avg_sequence_model_probability",
    ]:
        summary_df[column] = summary_df[column].round(4)

    name_map = assign_cluster_names(summary_df)
    cluster_df["strategy_cluster_name"] = cluster_df["strategy_cluster"].map(name_map)
    summary_df["strategy_cluster_name"] = summary_df["strategy_cluster"].map(name_map)

    return cluster_df, summary_df, cluster_model
