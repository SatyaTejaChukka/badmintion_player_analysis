from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_support import (
    analyze_user_sequence,
    build_coach_insights,
    load_dashboard_bundle,
    shot_frequency_table,
)


st.set_page_config(
    page_title="Rally Intelligence Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_bundle(input_path: str | None = None):
    resolved_input = Path(input_path) if input_path else None
    return load_dashboard_bundle(input_path=resolved_input, output_dir=Path("outputs"))


@st.cache_data(show_spinner=False)
def load_sequence_artifacts(
    output_dir: str,
) -> tuple[dict | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    output_path = Path(output_dir)
    summary_path = output_path / "attention_summary.json"
    history_path = output_path / "attention_history.csv"
    patterns_path = output_path / "attention_pattern_stats.csv"
    clusters_path = output_path / "strategy_cluster_summary.csv"
    tests_path = output_path / "attention_test_predictions.csv"

    summary = None
    history_df = None
    pattern_df = None
    cluster_df = None
    test_df = None

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if history_path.exists():
        history_df = pd.read_csv(history_path)
    if patterns_path.exists():
        pattern_df = pd.read_csv(patterns_path)
    if clusters_path.exists():
        cluster_df = pd.read_csv(clusters_path)
    if tests_path.exists():
        test_df = pd.read_csv(tests_path)

    return summary, history_df, pattern_df, cluster_df, test_df


@st.cache_data(show_spinner=False)
def load_model_comparison_artifacts(
    output_dir: str = "compare_outputs/comparison",
) -> tuple[dict | None, pd.DataFrame | None, pd.DataFrame | None]:
    output_path = Path(output_dir)
    summary_path = output_path / "model_comparison_summary.json"
    metrics_path = output_path / "model_metric_comparison.csv"
    predictions_path = output_path / "model_prediction_differences.csv"

    summary = None
    metrics_df = None
    predictions_df = None

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    if predictions_path.exists():
        predictions_df = pd.read_csv(predictions_path)

    return summary, metrics_df, predictions_df


def render_plotly(fig) -> None:
    st.plotly_chart(fig, width="stretch")


def render_table(df: pd.DataFrame) -> None:
    st.dataframe(df, width="stretch", hide_index=True)


def build_history_chart_df(history_df: pd.DataFrame) -> pd.DataFrame:
    metric_label_map = {
        "loss": "Train Loss",
        "val_loss": "Validation Loss",
        "win_probability_accuracy": "Train Win Accuracy",
        "val_win_probability_accuracy": "Validation Win Accuracy",
        "win_probability_auc": "Train Win AUC",
        "val_win_probability_auc": "Validation Win AUC",
        "next_shot_distribution_next_shot_accuracy": "Train Next-Shot Accuracy",
        "val_next_shot_distribution_next_shot_accuracy": "Validation Next-Shot Accuracy",
        "next_shot_distribution_next_shot_top3_accuracy": "Train Next-Shot Top-3",
        "val_next_shot_distribution_next_shot_top3_accuracy": "Validation Next-Shot Top-3",
        "accuracy": "Train Accuracy",
        "val_accuracy": "Validation Accuracy",
        "auc": "Train AUC",
        "val_auc": "Validation AUC",
    }
    available_columns = [
        column
        for column in metric_label_map
        if column in history_df.columns
    ]
    if not available_columns:
        return pd.DataFrame(columns=["epoch", "metric", "value"])

    melted = history_df.melt(
        id_vars="epoch",
        value_vars=available_columns,
        var_name="raw_metric",
        value_name="value",
    )
    melted["metric"] = melted["raw_metric"].map(metric_label_map).fillna(melted["raw_metric"])
    return melted[["epoch", "metric", "value"]]


def build_model_comparison_chart_df(metric_df: pd.DataFrame) -> pd.DataFrame:
    metric_label_map = {
        "test_accuracy": "Test Accuracy",
        "test_roc_auc": "Test ROC AUC",
        "test_prefix_accuracy": "Prefix Accuracy",
        "test_next_shot_accuracy": "Next-Shot Accuracy",
        "test_next_shot_top3_accuracy": "Next-Shot Top-3",
    }
    selected_rows = metric_df[metric_df["metric"].isin(metric_label_map)].copy()
    if selected_rows.empty:
        return pd.DataFrame(columns=["metric", "model", "value"])

    chart_df = selected_rows.melt(
        id_vars="metric",
        value_vars=["lstm_value", "bilstm_value"],
        var_name="raw_model",
        value_name="value",
    )
    chart_df["metric"] = chart_df["metric"].map(metric_label_map).fillna(chart_df["metric"])
    chart_df["model"] = chart_df["raw_model"].map(
        {"lstm_value": "LSTM + Attention", "bilstm_value": "BiLSTM + Attention"}
    )
    return chart_df[["metric", "model", "value"]]


def format_comparison_value(metric_name: str, value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if metric_name in {"test_loss"}:
        return f"{float(value):.4f}"
    if metric_name in {"epochs_ran"}:
        return f"{int(float(value))}"
    return f"{float(value):.1%}"


def format_comparison_delta(metric_name: str, value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if metric_name in {"test_loss"}:
        return f"{float(value):+.4f}"
    if metric_name in {"epochs_ran"}:
        return f"{float(value):+.0f}"
    return f"{float(value):+.1%}"


def render_overview(bundle) -> None:
    st.subheader("Performance Snapshot")

    labeled_df = bundle.rally_df.dropna(subset=["Player1Won"]).copy()
    labeled_rallies = int(len(labeled_df))
    player1_win_rate = float(labeled_df["Player1Won"].mean()) if not labeled_df.empty else 0.0
    average_rally_length = float(bundle.rally_df["rally_length"].mean()) if not bundle.rally_df.empty else 0.0
    average_duration = float(bundle.rally_df["rally_duration_sec"].mean()) if not bundle.rally_df.empty else 0.0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rallies", f"{len(bundle.rally_df)}")
    metric_cols[1].metric("Labeled Rallies", f"{labeled_rallies}")
    metric_cols[2].metric("Player 1 Win Rate", f"{player1_win_rate:.1%}" if labeled_rallies else "N/A")
    metric_cols[3].metric("Avg Rally Length", f"{average_rally_length:.1f}", delta=f"{average_duration:.1f}s avg")

    st.subheader("Coach Insights")
    for insight in build_coach_insights(bundle):
        st.markdown(f"- {insight}")

    tactic_profile = pd.DataFrame(
        {
            "tactic": ["Attack", "Defense", "Neutral", "Terminal"],
            "mean_ratio": [
                bundle.rally_df["attack_ratio"].mean(),
                bundle.rally_df["defense_ratio"].mean(),
                bundle.rally_df["neutral_ratio"].mean(),
                bundle.rally_df["terminal_ratio"].mean(),
            ],
        }
    )

    shot_df = shot_frequency_table(bundle.rally_df).head(12)
    scatter_df = bundle.rally_df.copy()
    scatter_df["Outcome"] = scatter_df["Player1Won"].map({1.0: "Won", 0.0: "Lost"}).fillna("Unknown")
    strategy_df = bundle.strategy_stats_df.sort_values("win_rate", ascending=True).copy()

    left_col, right_col = st.columns(2)

    with left_col:
        fig = px.bar(
            tactic_profile,
            x="tactic",
            y="mean_ratio",
            color="tactic",
            title="Average Tactical Mix",
            color_discrete_sequence=["#b53c2f", "#315d8a", "#d79b2e", "#616161"],
        )
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        render_plotly(fig)

    with right_col:
        fig = px.bar(
            shot_df,
            x="count",
            y="shot",
            orientation="h",
            title="Most Common Shot Types",
            color="count",
            color_continuous_scale="Sunset",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        render_plotly(fig)

    left_col, right_col = st.columns(2)

    with left_col:
        fig = px.scatter(
            scatter_df,
            x="attack_ratio",
            y="defense_ratio",
            color="Outcome",
            size="rally_length",
            hover_data=[
                "RallyId",
                "neutral_ratio",
                "dominant_tactic",
            ],
            title="Win/Loss Map by Tactical Mix",
            color_discrete_map={"Won": "#228B22", "Lost": "#b53c2f", "Unknown": "#6c757d"},
        )
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        render_plotly(fig)

    with right_col:
        if strategy_df.empty:
            st.info("No labeled strategy buckets are available yet.")
        else:
            fig = px.bar(
                strategy_df,
                x="win_rate",
                y="strategy_bucket",
                orientation="h",
                title="Win Rate by Strategy Bucket",
                color="avg_attack_ratio",
                hover_data=["rallies", "avg_defense_ratio", "avg_neutral_ratio"],
                color_continuous_scale="Viridis",
            )
            fig.update_layout(xaxis_tickformat=".0%")
            render_plotly(fig)


def render_patterns(bundle) -> None:
    st.subheader("Winning Sequences")

    controls = st.columns(3)
    min_count = controls[0].slider("Minimum occurrences", min_value=3, max_value=20, value=5)
    min_support = controls[1].slider("Minimum rally support", min_value=0.05, max_value=0.60, value=0.10)
    top_n = controls[2].slider("Patterns to display", min_value=5, max_value=20, value=10)
    pattern_view = st.radio(
        "Pattern source",
        options=["Tactical Labels", "Raw Shot Names"],
        horizontal=True,
    )

    active_pattern_df = bundle.pattern_df if pattern_view == "Tactical Labels" else bundle.shot_pattern_df
    chart_title = (
        "Top Supported Winning Tactical Patterns"
        if pattern_view == "Tactical Labels"
        else "Top Supported Winning Raw Shot Patterns"
    )

    filtered_patterns = active_pattern_df[
        (active_pattern_df["count"] >= min_count)
        & (active_pattern_df["rally_support_pct"] >= min_support)
    ].copy()

    if filtered_patterns.empty:
        st.warning("No patterns meet the current filters. Lower the thresholds to see more sequences.")
    else:
        chart_df = filtered_patterns.head(top_n).sort_values("win_rate", ascending=True)
        fig = px.bar(
            chart_df,
            x="win_rate",
            y="pattern",
            orientation="h",
            color="delta_vs_baseline",
            title=chart_title,
            hover_data=["count", "rally_count", "rally_support_pct"],
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(xaxis_tickformat=".0%")
        render_plotly(fig)

        display_df = filtered_patterns.head(top_n).copy()
        display_df["win_rate"] = display_df["win_rate"].map(lambda value: f"{value:.1%}")
        display_df["rally_support_pct"] = display_df["rally_support_pct"].map(lambda value: f"{value:.1%}")
        display_df["delta_vs_baseline"] = display_df["delta_vs_baseline"].map(lambda value: f"{value:+.1%}")
        render_table(display_df)

    st.subheader("Strategy Buckets")
    strategy_df = bundle.strategy_stats_df.sort_values("win_rate", ascending=False).copy()
    if strategy_df.empty:
        st.info("No strategy stats available because there are no labeled rally outcomes.")
        return

    fig = px.bar(
        strategy_df,
        x="win_rate",
        y="strategy_bucket",
        orientation="h",
        color="avg_attack_ratio",
        title="Win Rate by Detected Strategy",
        hover_data=["rallies", "avg_attack_ratio", "avg_defense_ratio", "avg_neutral_ratio"],
        color_continuous_scale="Viridis",
    )
    fig.update_layout(xaxis_tickformat=".0%")
    render_plotly(fig)

    display_df = strategy_df.copy()
    for column in ["win_rate", "avg_attack_ratio", "avg_defense_ratio", "avg_neutral_ratio"]:
        display_df[column] = display_df[column].map(lambda value: f"{value:.1%}")
    render_table(display_df)


def render_analyzer(bundle) -> None:
    st.subheader("Rally Analyzer")
    st.caption("Enter a comma-separated rally to classify each shot, estimate the win chance, and get a tactical recommendation.")

    default_sequence = "Serve, Lift, Drop, Smash"
    input_key = "rally_analyzer_sequence_text"
    result_key = "rally_analyzer_result"
    source_key = "rally_analyzer_result_input"

    if st.session_state.get(source_key) != bundle.input_path.name:
        st.session_state.pop(result_key, None)
        st.session_state[source_key] = bundle.input_path.name

    if input_key not in st.session_state:
        st.session_state[input_key] = default_sequence

    with st.form("rally_analyzer_form"):
        st.text_area("Shot sequence", key=input_key, height=100)
        submitted = st.form_submit_button("Analyze Rally", type="primary")

    if submitted:
        sequence_text = st.session_state[input_key]
        if sequence_text.strip():
            st.session_state[result_key] = analyze_user_sequence(sequence_text, bundle)
        else:
            st.session_state.pop(result_key, None)

    result = st.session_state.get(result_key)
    if result is None:
        st.info("Enter a comma-separated rally and click `Analyze Rally`.")
        return

    if not result["shots"]:
        st.warning("The sequence could not be parsed. Use comma-separated shot names such as `Serve, Lift, Drop, Smash`.")
        return

    features = result["features"]
    probability = result["win_probability"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted Win Probability", f"{probability:.1%}" if probability is not None else "N/A")
    metric_cols[1].metric("Attack Ratio", f"{features['attack_ratio']:.1%}")
    metric_cols[2].metric("Defense Ratio", f"{features['defense_ratio']:.1%}")
    metric_cols[3].metric("Estimated Duration", f"{features['rally_duration_sec']:.1f}s")

    st.markdown(f"**Detected Strategy:** {result['strategy_bucket']}")
    st.success(result["recommendation"])

    if result["unknown_shots"]:
        unknown_summary = ", ".join(sorted(result["unknown_shots"].keys()))
        st.warning(f"Unknown shot labels were defaulted to Neutral: {unknown_summary}")

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**Shot Classification**")
        render_table(result["shot_table"])

    with right_col:
        if result["matching_patterns"].empty:
            st.info("No historical tactical pattern from the current table matched this exact rally sequence.")
        else:
            st.markdown("**Matching Historical Patterns**")
            display_df = result["matching_patterns"].copy()
            display_df["win_rate"] = display_df["win_rate"].map(lambda value: f"{value:.1%}")
            display_df["rally_support_pct"] = display_df["rally_support_pct"].map(lambda value: f"{value:.1%}")
            display_df["delta_vs_baseline"] = display_df["delta_vs_baseline"].map(lambda value: f"{value:+.1%}")
            render_table(display_df)

        if result["matching_shot_patterns"].empty:
            st.info("No raw shot-name pattern from the current table matched this exact rally sequence.")
        else:
            st.markdown("**Matching Raw Shot Patterns**")
            display_df = result["matching_shot_patterns"].copy()
            display_df["win_rate"] = display_df["win_rate"].map(lambda value: f"{value:.1%}")
            display_df["rally_support_pct"] = display_df["rally_support_pct"].map(lambda value: f"{value:.1%}")
            display_df["delta_vs_baseline"] = display_df["delta_vs_baseline"].map(lambda value: f"{value:+.1%}")
            render_table(display_df)

    st.markdown("**What If You Add One More Shot?**")
    what_if_df = result["what_if_df"].copy()
    min_recommendation_delta = (
        float(result.get("sequence_result", {}).get("min_recommendation_delta", 0.01))
        if result.get("sequence_result") is not None
        else 0.01
    )
    if what_if_df.empty:
        st.info(
            f"No context-aware next-shot option clears the minimum meaningful gain threshold of {min_recommendation_delta:.1%}."
        )
    else:
        st.caption(
            f"Ranked by a plausibility-aware score that blends the auxiliary next-shot model with resulting win probability. Suggestions must clear at least {min_recommendation_delta:.1%} gain."
        )
        what_if_df["predicted_win_probability"] = what_if_df["predicted_win_probability"].map(lambda value: f"{value:.2%}")
        if "next_shot_probability" in what_if_df.columns:
            what_if_df["next_shot_probability"] = what_if_df["next_shot_probability"].map(lambda value: f"{value:.2%}")
        if "recommendation_score" in what_if_df.columns:
            what_if_df["recommendation_score"] = what_if_df["recommendation_score"].map(lambda value: f"{value:.4f}")
        chart_df = result["what_if_df"].copy()
        fig = px.bar(
            chart_df.sort_values("recommendation_score", ascending=True),
            x="delta_vs_current",
            y="candidate_shot",
            orientation="h",
            color="delta_vs_current",
            title="Immediate Shot Recommendation Impact",
            hover_data=["predicted_win_probability", "recommendation_score"] if "recommendation_score" in chart_df.columns else None,
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(xaxis_tickformat="+.2%")
        render_plotly(fig)
        render_table(what_if_df)

    sequence_result = result.get("sequence_result")
    if sequence_result is not None:
        st.markdown("**Sequence Intelligence**")
        deep_cols = st.columns(3)
        deep_probability = sequence_result["win_probability"]
        deep_cols[0].metric("Attention Model Win Prob.", f"{deep_probability:.1%}")
        deep_cols[1].metric(
            "Strategy Cluster",
            sequence_result.get("strategy_cluster_name") or "Unknown",
        )
        deep_cols[2].metric(
            "Most Important Shot",
            sequence_result["shot_importance_df"].iloc[0]["shot"] if not sequence_result["shot_importance_df"].empty else "N/A",
        )

        next_shot_probability_df = sequence_result.get("next_shot_probability_df", pd.DataFrame()).copy()
        if not next_shot_probability_df.empty:
            st.markdown("**Most Plausible Next Shots**")
            fig = px.bar(
                next_shot_probability_df.head(8).sort_values("next_shot_probability", ascending=True),
                x="next_shot_probability",
                y="candidate_shot",
                orientation="h",
                color="next_shot_probability",
                title="Auxiliary Next-Shot Head",
                color_continuous_scale="Blues",
            )
            fig.update_layout(xaxis_tickformat=".0%")
            render_plotly(fig)
            display_next_shot_df = next_shot_probability_df.head(8).copy()
            display_next_shot_df["next_shot_probability"] = display_next_shot_df["next_shot_probability"].map(lambda value: f"{value:.2%}")
            render_table(display_next_shot_df)

        prefix_df = sequence_result["prefix_probability_df"].copy()
        if not prefix_df.empty:
            st.markdown("**Win Probability After Each Shot**")
            fig = px.line(
                prefix_df,
                x="position",
                y="win_probability_after_shot",
                markers=True,
                hover_data=["shot", "delta_vs_previous_shot", "prefix_sequence"],
                title="Prefix Win-Probability Trajectory",
            )
            fig.update_layout(xaxis_title="Shot Position", yaxis_tickformat=".0%")
            render_plotly(fig)
            display_prefix_df = prefix_df.copy()
            display_prefix_df["win_probability_after_shot"] = display_prefix_df["win_probability_after_shot"].map(lambda value: f"{value:.1%}")
            display_prefix_df["delta_vs_previous_shot"] = display_prefix_df["delta_vs_previous_shot"].map(lambda value: f"{value:+.1%}")
            display_prefix_df["cumulative_delta_vs_first_shot"] = display_prefix_df["cumulative_delta_vs_first_shot"].map(lambda value: f"{value:+.1%}")
            render_table(display_prefix_df)

        left_col, right_col = st.columns(2)

        with left_col:
            attention_df = sequence_result["shot_importance_df"].sort_values("position").copy()
            fig = px.bar(
                attention_df,
                x="position",
                y="attention_weight",
                hover_data=["shot"],
                title="Attention Weight by Shot Position",
                color="attention_weight",
                color_continuous_scale="YlOrRd",
            )
            fig.update_layout(xaxis_title="Shot Position", yaxis_tickformat=".0%")
            render_plotly(fig)
            display_attention_df = sequence_result["shot_importance_df"].copy()
            display_attention_df["attention_weight"] = display_attention_df["attention_weight"].map(lambda value: f"{value:.1%}")
            render_table(display_attention_df)

        with right_col:
            replacement_df = sequence_result["replacement_df"].copy()
            if replacement_df.empty:
                st.info(
                    f"No context-aware replacement option clears the minimum meaningful gain threshold of {float(sequence_result.get('min_recommendation_delta', 0.01)):.1%}."
                )
            else:
                chart_df = replacement_df.head(10).copy()
                chart_df["replacement_option"] = chart_df.apply(
                    lambda row: f"Shot {int(row['position'])}: {row['original_shot']} -> {row['candidate_shot']}",
                    axis=1,
                )
                st.caption(
                    f"Each bar is one separate single-shot swap, ranked by a plausibility-aware replacement score. Suggestions must clear at least {float(sequence_result.get('min_recommendation_delta', 0.01)):.1%} gain."
                )
                fig = px.bar(
                    chart_df.sort_values("replacement_score", ascending=True),
                    x="delta_vs_current",
                    y="replacement_option",
                    color="delta_vs_current",
                    orientation="h",
                    title="Best Shot Replacements",
                    hover_data=["position", "original_shot", "candidate_shot", "predicted_win_probability", "replacement_score"],
                    color_continuous_scale="RdYlGn",
                )
                fig.update_layout(xaxis_tickformat="+.2%", yaxis_title="Replacement")
                render_plotly(fig)
                display_replacement_df = replacement_df.copy()
                display_replacement_df["predicted_win_probability"] = display_replacement_df["predicted_win_probability"].map(lambda value: f"{value:.2%}")
                display_replacement_df["delta_vs_current"] = display_replacement_df["delta_vs_current"].map(lambda value: f"{value:+.2%}")
                if "candidate_shot_plausibility" in display_replacement_df.columns:
                    display_replacement_df["candidate_shot_plausibility"] = display_replacement_df["candidate_shot_plausibility"].map(lambda value: f"{value:.2%}")
                if "replacement_score" in display_replacement_df.columns:
                    display_replacement_df["replacement_score"] = display_replacement_df["replacement_score"].map(lambda value: f"{value:.4f}")
                render_table(display_replacement_df)


def render_deep_model(bundle) -> None:
    st.subheader("Sequence Intelligence")
    st.caption("This tab surfaces the attention-based BiLSTM model that explains which shots mattered, which subsequences carry signal, and which rally styles the embedding space discovers.")

    summary, history_df, pattern_df, cluster_df, test_df = load_sequence_artifacts(str(bundle.output_dir))

    if summary is None:
        st.info("No attention-model artifacts found yet. Run `python train_sequence_model.py` to train and export them.")
        return

    if summary.get("input_file") != bundle.input_path.name:
        st.warning(
            f"Attention artifacts were trained on `{summary.get('input_file')}`, but the active CSV is `{bundle.input_path.name}`. Re-run `python train_sequence_model.py` to refresh them."
        )
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Train Rallies", f"{int(summary.get('train_rows', 0))}")
    metric_cols[1].metric("Test Rallies", f"{int(summary.get('test_rows', 0))}")
    metric_cols[2].metric("Test Accuracy", f"{summary.get('test_accuracy', 0.0):.1%}")
    metric_cols[3].metric("Test ROC AUC", f"{summary.get('test_roc_auc', 0.0) if summary.get('test_roc_auc') is not None else 0.0:.1%}")

    if summary.get("test_prefix_accuracy") is not None or summary.get("test_prefix_roc_auc") is not None:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Training Mode", summary.get("training_mode", "unknown").replace("_", " ").title())
        metric_cols[1].metric("Test Prefix Rows", f"{int(summary.get('test_prefix_rows', 0))}")
        metric_cols[2].metric("Prefix Rollout Acc.", f"{summary.get('test_prefix_accuracy', 0.0):.1%}" if summary.get("test_prefix_accuracy") is not None else "N/A")
        metric_cols[3].metric("Prefix Rollout ROC AUC", f"{summary.get('test_prefix_roc_auc', 0.0):.1%}" if summary.get("test_prefix_roc_auc") is not None else "N/A")

    if summary.get("test_next_shot_accuracy") is not None or summary.get("test_next_shot_top3_accuracy") is not None:
        metric_cols = st.columns(3)
        metric_cols[0].metric("Next-Shot Rows", f"{int(summary.get('test_next_shot_rows', 0))}")
        metric_cols[1].metric("Next-Shot Acc.", f"{summary.get('test_next_shot_accuracy', 0.0):.1%}" if summary.get("test_next_shot_accuracy") is not None else "N/A")
        metric_cols[2].metric("Next-Shot Top-3", f"{summary.get('test_next_shot_top3_accuracy', 0.0):.1%}" if summary.get("test_next_shot_top3_accuracy") is not None else "N/A")

    st.markdown(
        "This model learns from the full rally shot sequence and final win/loss label, then rolls the same sequence forward shot by shot at inference so you can inspect probability swings, attention, and learned play styles."
    )

    if history_df is not None and not history_df.empty:
        history_chart_df = build_history_chart_df(history_df)
        if history_chart_df.empty:
            st.info("No supported training-history metrics are available in the exported history file.")
        else:
            fig = px.line(
                history_chart_df,
                x="epoch",
                y="value",
                color="metric",
                title="LSTM Training History",
            )
            render_plotly(fig)

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("**High-Impact Subsequences**")
        if pattern_df is None or pattern_df.empty:
            st.info("No attention-derived pattern table is available yet.")
        else:
            chart_df = pattern_df.head(12).sort_values("avg_attention_weight", ascending=True)
            fig = px.bar(
                chart_df,
                x="avg_attention_weight",
                y="pattern",
                orientation="h",
                color="win_rate",
                hover_data=["count", "rally_count", "avg_predicted_win_probability"],
                title="Patterns Emphasized by Attention",
                color_continuous_scale="Turbo",
            )
            fig.update_layout(xaxis_tickformat=".0%")
            render_plotly(fig)
            display_pattern_df = pattern_df.head(12).copy()
            for column in ["win_rate", "avg_predicted_win_probability", "avg_attention_weight", "rally_support_pct"]:
                display_pattern_df[column] = display_pattern_df[column].map(lambda value: f"{value:.1%}")
            render_table(display_pattern_df)

    with right_col:
        st.markdown("**Learned Strategy Clusters**")
        if cluster_df is None or cluster_df.empty:
            st.info("No strategy-cluster summary is available yet.")
        else:
            fig = px.bar(
                cluster_df.sort_values("win_rate", ascending=True),
                x="win_rate",
                y="strategy_cluster_name",
                orientation="h",
                color="avg_attack_ratio",
                hover_data=["rallies", "avg_defense_ratio", "avg_neutral_ratio", "avg_sequence_model_probability"],
                title="Embedding-Derived Play Styles",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(xaxis_tickformat=".0%")
            render_plotly(fig)
            display_cluster_df = cluster_df.copy()
            for column in ["win_rate", "avg_attack_ratio", "avg_defense_ratio", "avg_neutral_ratio", "avg_sequence_model_probability"]:
                display_cluster_df[column] = display_cluster_df[column].map(lambda value: f"{value:.1%}")
            render_table(display_cluster_df)

    st.markdown("**Attention-Test Predictions**")
    if test_df is None or test_df.empty:
        st.info("No held-out attention predictions are available yet.")
    else:
        display_test_df = test_df.copy()
        display_test_df["predicted_probability"] = display_test_df["predicted_probability"].map(lambda value: f"{value:.1%}")
        render_table(display_test_df.head(12))

    st.markdown("**LSTM vs BiLSTM Comparison**")
    comparison_summary, comparison_metrics_df, comparison_predictions_df = load_model_comparison_artifacts()

    if comparison_summary is None:
        st.info(
            "No encoder-comparison artifacts found yet. Run the matched training commands and then `python compare_model_outputs.py` to surface them here."
        )
    elif comparison_summary.get("input_file") != bundle.input_path.name:
        st.warning(
            f"Comparison artifacts were generated for `{comparison_summary.get('input_file')}`, but the active CSV is `{bundle.input_path.name}`."
        )
    else:
        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "LSTM Test Acc.",
            f"{comparison_summary.get('lstm_test_accuracy', 0.0):.1%}",
        )
        metric_cols[1].metric(
            "BiLSTM Test Acc.",
            f"{comparison_summary.get('bilstm_test_accuracy', 0.0):.1%}",
            delta=f"{comparison_summary.get('accuracy_gap_bilstm_minus_lstm', 0.0):+.1%} vs LSTM",
        )
        metric_cols[2].metric(
            "Changed Labels",
            f"{int(comparison_summary.get('rows_with_changed_label', 0))} / {int(comparison_summary.get('test_rows_compared', 0))}",
        )
        metric_cols[3].metric(
            "Avg Prob Shift",
            f"{comparison_summary.get('avg_abs_probability_delta', 0.0):.1%}",
            delta=f"{comparison_summary.get('max_abs_probability_delta', 0.0):.1%} max",
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "LSTM ROC AUC",
            f"{comparison_summary.get('lstm_test_roc_auc', 0.0):.1%}",
        )
        metric_cols[1].metric(
            "BiLSTM ROC AUC",
            f"{comparison_summary.get('bilstm_test_roc_auc', 0.0):.1%}",
            delta=f"{comparison_summary.get('roc_auc_gap_bilstm_minus_lstm', 0.0):+.1%} vs LSTM",
        )
        metric_cols[2].metric(
            "BiLSTM-Only Fixes",
            f"{int(comparison_summary.get('bilstm_only_correct_rows', 0))}",
        )
        metric_cols[3].metric(
            "LSTM-Only Fixes",
            f"{int(comparison_summary.get('lstm_only_correct_rows', 0))}",
        )

        if comparison_metrics_df is not None and not comparison_metrics_df.empty:
            chart_df = build_model_comparison_chart_df(comparison_metrics_df)
            if not chart_df.empty:
                fig = px.bar(
                    chart_df,
                    x="metric",
                    y="value",
                    color="model",
                    barmode="group",
                    title="Held-Out Metric Comparison",
                    color_discrete_sequence=["#5d6d7e", "#c0392b"],
                )
                fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None)
                render_plotly(fig)

            display_metric_df = comparison_metrics_df.copy()
            metric_label_map = {
                "test_accuracy": "Test Accuracy",
                "test_roc_auc": "Test ROC AUC",
                "test_loss": "Test Loss",
                "test_prefix_accuracy": "Prefix Accuracy",
                "test_prefix_roc_auc": "Prefix ROC AUC",
                "test_next_shot_accuracy": "Next-Shot Accuracy",
                "test_next_shot_top3_accuracy": "Next-Shot Top-3",
                "best_val_accuracy": "Best Val Accuracy",
                "best_val_auc": "Best Val ROC AUC",
                "best_val_next_shot_accuracy": "Best Val Next-Shot Accuracy",
                "best_val_next_shot_top3_accuracy": "Best Val Next-Shot Top-3",
                "epochs_ran": "Epochs Ran",
            }
            display_metric_df["metric"] = display_metric_df["metric"].map(metric_label_map).fillna(display_metric_df["metric"])
            display_metric_df["higher_is_better"] = display_metric_df["higher_is_better"].map({True: "Yes", False: "No"})
            display_metric_df["lstm_value"] = comparison_metrics_df.apply(
                lambda row: format_comparison_value(str(row["metric"]), row["lstm_value"]),
                axis=1,
            )
            display_metric_df["bilstm_value"] = comparison_metrics_df.apply(
                lambda row: format_comparison_value(str(row["metric"]), row["bilstm_value"]),
                axis=1,
            )
            display_metric_df["bilstm_minus_lstm"] = comparison_metrics_df.apply(
                lambda row: format_comparison_delta(str(row["metric"]), row["bilstm_minus_lstm"]),
                axis=1,
            )
            render_table(display_metric_df)

        if comparison_predictions_df is not None and not comparison_predictions_df.empty:
            st.markdown("**Rows Where Final Prediction Changed**")
            changed_df = comparison_predictions_df[comparison_predictions_df["label_changed"]].copy()
            if changed_df.empty:
                st.info("The two encoders made the same final label prediction on every held-out rally.")
            else:
                fig = px.bar(
                    changed_df.sort_values("probability_delta", ascending=True),
                    x="probability_delta",
                    y="RallyId",
                    orientation="h",
                    color="comparison_outcome",
                    hover_data=[
                        "Player1Won",
                        "predicted_probability_lstm",
                        "predicted_probability_bilstm",
                    ],
                    title="Rallies Where the Final Predicted Label Changed",
                    color_discrete_map={
                        "bilstm_only_correct": "#1f7a1f",
                        "lstm_only_correct": "#c0392b",
                        "both_correct": "#2e86de",
                        "both_wrong": "#7f8c8d",
                    },
                )
                fig.update_layout(xaxis_tickformat="+.1%", yaxis_title="RallyId")
                render_plotly(fig)

                display_changed_df = changed_df.copy()
                display_changed_df["Player1Won"] = display_changed_df["Player1Won"].map(lambda value: int(float(value)))
                display_changed_df["predicted_probability_lstm"] = display_changed_df["predicted_probability_lstm"].map(lambda value: f"{value:.1%}")
                display_changed_df["predicted_probability_bilstm"] = display_changed_df["predicted_probability_bilstm"].map(lambda value: f"{value:.1%}")
                display_changed_df["probability_delta"] = display_changed_df["probability_delta"].map(lambda value: f"{value:+.1%}")
                render_table(display_changed_df)

            st.markdown("**Largest Probability Shifts**")
            display_difference_df = comparison_predictions_df.head(10).copy()
            display_difference_df["Player1Won"] = display_difference_df["Player1Won"].map(lambda value: int(float(value)))
            display_difference_df["predicted_probability_lstm"] = display_difference_df["predicted_probability_lstm"].map(lambda value: f"{value:.1%}")
            display_difference_df["predicted_probability_bilstm"] = display_difference_df["predicted_probability_bilstm"].map(lambda value: f"{value:.1%}")
            display_difference_df["probability_delta"] = display_difference_df["probability_delta"].map(lambda value: f"{value:+.1%}")
            display_difference_df["abs_probability_delta"] = display_difference_df["abs_probability_delta"].map(lambda value: f"{value:.1%}")
            render_table(display_difference_df)

    st.json(summary)


def main() -> None:
    st.title("Rally Intelligence Dashboard")
    st.caption("Interactive tactical analysis for badminton-style rally data.")

    with st.sidebar:
        st.header("Controls")
        st.code("streamlit run dashboard.py", language="bash")
        if st.button("Refresh Analysis Outputs"):
            get_bundle.clear()
            load_sequence_artifacts.clear()
            load_model_comparison_artifacts.clear()
            st.rerun()

    bundle = get_bundle()

    with st.sidebar:
        st.markdown(f"**Source CSV:** `{bundle.input_path.name}`")
        st.markdown(f"**Estimated seconds per shot:** `{bundle.seconds_per_shot:.2f}`")
        st.markdown(f"**Output folder:** `{bundle.output_dir}`")

    overview_tab, patterns_tab, analyzer_tab, deep_model_tab = st.tabs(
        ["Overview", "Patterns", "Rally Analyzer", "Deep Model"]
    )

    with overview_tab:
        render_overview(bundle)

    with patterns_tab:
        render_patterns(bundle)

    with analyzer_tab:
        render_analyzer(bundle)

    with deep_model_tab:
        render_deep_model(bundle)


if __name__ == "__main__":
    main()
