"""GenAI agent that answers questions with grounded analytics and LLM rewrites."""

from __future__ import annotations

import logging
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:  # LangChain 0.2+
    from langchain_community.callbacks.manager import get_openai_callback  # type: ignore
except Exception:  # pragma: no cover - legacy import path
    try:
        from langchain.callbacks import get_openai_callback  # type: ignore
    except Exception:  # pragma: no cover - callback manager unavailable
        get_openai_callback = None  # type: ignore

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class _EmptyCallback:
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    total_tokens: float = 0.0
    total_cost: float = 0.0
    is_dummy: bool = True


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()


class QualiGenixAgent:
    """Serve deterministic answers backed by the historical dataset and ML models."""

    def __init__(
        self,
        data_path: str = "data/processed/Master.csv",
        models_dir: str = "data/processed/models",
        results_dir: str = "data/processed/ml_results",
    ) -> None:
        self.data_path = Path(data_path)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)

        self.master_df = self._load_master_dataset()
        self.numeric_columns = self.master_df.select_dtypes(include=[np.number]).columns
        self.label_encoders = self._build_label_encoders()
        self.default_values = self._build_default_values()
        self.metric_aliases = self._build_metric_aliases()
        self.metric_display_names = self._build_metric_display_names()
        self.parameter_alias_lookup = self._build_parameter_alias_lookup()
        self.column_lookup = self._build_column_lookup()
        self.dataset_summary = self._compute_dataset_summary()
        self.feature_importance = self._load_feature_importance()
        self.model_registry, self.model_metrics = self._load_models()
        self.summary_llm: Optional[ChatOpenAI] = None
        self.callback_factory = get_openai_callback
        self.usage_totals = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        self.last_llm_stats: Optional[Dict[str, float]] = None
        self._setup_summarizer()

        logger.info("QualiGenixAgent ready: %d rows, %d columns", len(self.master_df), len(self.master_df.columns))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def query(self, question: str) -> str:
        if not question or not question.strip():
            return "Please provide a question about the manufacturing data or predictions."

        normalized = question.lower()
        handlers = [
            self._handle_prediction,
            self._handle_batch_comparison,
            self._handle_metric_summary,
            self._handle_feature_influence,
            self._handle_recommendation,
        ]

        for handler in handlers:
            response = handler(question, normalized)
            if response:
                return self._format_final_response(question, response)

        return self._format_final_response(question, self._fallback_response())

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _load_master_dataset(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        df = df.copy()

        for column in df.columns:
            if df[column].dtype == object:
                try:
                    df[column] = pd.to_numeric(df[column])
                except (TypeError, ValueError):
                    continue

        if {"tbl_max_weight", "tbl_min_weight"}.issubset(df.columns):
            df["weight_consistency"] = df["tbl_max_weight"] - df["tbl_min_weight"]

        if {"batch_yield", "Batch Size (tablets)"}.issubset(df.columns):
            batch_size = df["Batch Size (tablets)"].replace(0, np.nan)
            df["normalized_yield"] = (df["batch_yield"] / batch_size) * 1000

        return df

    def _build_default_values(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        for column in self.master_df.columns:
            series = self.master_df[column].dropna()
            if series.empty:
                continue
            if column in self.label_encoders:
                mapping = self.label_encoders[column]
                encoded_series = series.astype(str).map(mapping)
                encoded_series = encoded_series.dropna()
                if encoded_series.empty:
                    continue
                defaults[column] = float(encoded_series.median())
                continue
            if pd.api.types.is_numeric_dtype(series):
                defaults[column] = float(series.median())
            else:
                try:
                    defaults[column] = series.mode().iloc[0]
                except IndexError:
                    defaults[column] = series.iloc[0]

        for engineered in ["weight_consistency", "normalized_yield"]:
            if engineered not in defaults and engineered in self.master_df.columns:
                series = self.master_df[engineered].dropna()
                if not series.empty:
                    defaults[engineered] = float(series.median())

        return defaults

    def _build_label_encoders(self) -> Dict[str, Dict[str, int]]:
        categorical_cols = [
            "api_batch",
            "smcc_batch",
            "lactose_batch",
            "starch_batch",
            "code",
            "strength",
            "size",
        ]
        encoders: Dict[str, Dict[str, int]] = {}
        for column in categorical_cols:
            if column not in self.master_df.columns:
                continue
            series = self.master_df[column].astype(str).fillna("missing")
            classes = sorted(series.unique())
            encoders[column] = {label: idx for idx, label in enumerate(classes)}
        return encoders

    def _build_metric_aliases(self) -> Dict[str, List[str]]:
        return {
            "dissolution_min": ["dissolution min", "minimum dissolution", "lowest dissolution"],
            "dissolution_av": [
                "average dissolution",
                "dissolution rate",
                "mean dissolution",
                "overall dissolution",
                "dissolution",
            ],
            "batch_yield": ["batch yield", "yield"],
            "impurities_total": ["total impurities", "impurities"],
            "impurity_o": ["impurity o"],
            "impurity_l": ["impurity l"],
            "resodual_solvent": ["residual solvent", "solvent"],
        }

    def _build_metric_display_names(self) -> Dict[str, str]:
        return {
            "dissolution_av": "Average dissolution (%)",
            "dissolution_min": "Minimum dissolution (%)",
            "batch_yield": "Batch yield (%)",
            "impurities_total": "Total impurities (%)",
            "impurity_o": "Impurity O (%)",
            "impurity_l": "Impurity L (%)",
            "resodual_solvent": "Residual solvent (%)",
        }

    def _build_parameter_alias_lookup(self) -> Dict[str, str]:
        alias_map = {
            "api_water": ["api water", "api_water"],
            "api_content": ["api content"],
            "api_total_impurities": ["api impurities", "api total impurities"],
            "main_CompForce mean": ["compression", "compression force", "main compforce", "main_compforce"],
            "main_CompForce_median": ["compression median", "compforce median"],
            "tbl_speed_mean": ["speed", "press speed", "tableting speed", "tbl speed"],
            "tbl_fill_mean": ["fill mean", "die fill"],
            "stiffness_mean": ["stiffness"],
            "Batch Size (tablets)": ["batch size", "tablet count"],
            "batch_yield": ["yield", "batch_yield"],
            "weight_consistency": ["weight consistency"],
            "normalized_yield": ["normalized yield"],
            "impurities_total": ["impurities"],
            "dissolution_av": ["dissolution"],
        }

        lookup: Dict[str, str] = {}
        for feature, aliases in alias_map.items():
            for alias in aliases:
                lookup[self._sanitize_text(alias)] = feature
        return lookup

    def _build_column_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for column in self.master_df.columns:
            lookup[self._sanitize_text(column)] = column
        return lookup

    def _compute_dataset_summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for metric in self.metric_display_names.keys():
            if metric in self.master_df.columns:
                series = self.master_df[metric].dropna()
                if series.empty:
                    continue
                summary[metric] = {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "p10": float(series.quantile(0.10)),
                    "p90": float(series.quantile(0.90)),
                }
        return summary

    def _load_feature_importance(self) -> Dict[str, List[Tuple[str, float]]]:
        importance: Dict[str, List[Tuple[str, float]]] = {}
        if not self.results_dir.exists():
            return importance

        for file in self.results_dir.glob("feature_importance_*.csv"):
            target = file.stem.replace("feature_importance_", "")
            try:
                df = pd.read_csv(file)
            except Exception as exc:
                logger.warning("Could not read %s: %s", file, exc)
                continue

            if df.empty or "feature" not in df.columns or "importance" not in df.columns:
                continue

            # Normalise within each model before aggregating
            df["normalised"] = df.groupby("model")["importance"].transform(
                lambda x: x / x.sum() if x.sum() else 0
            )
            ranking = (
                df.groupby("feature")["normalised"].sum().sort_values(ascending=False).reset_index()
            )
            importance[target] = list(zip(ranking["feature"], ranking["normalised"].round(4)))
        return importance

    def _load_models(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, float]]]:
        registry: Dict[str, Dict[str, Any]] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        summary_path = self.results_dir / "model_results_summary.csv"
        best_by_target: Dict[str, str] = {}

        if summary_path.exists():
            try:
                summary_df = pd.read_csv(summary_path)
                test_df = summary_df[summary_df["split"] == "test"]
                test_df = test_df.sort_values("r2", ascending=False)
                best_rows = test_df.drop_duplicates(subset=["target"], keep="first")
                for _, row in best_rows.iterrows():
                    best_by_target[str(row["target"])] = str(row["model"])
                    metrics[str(row["target"])] = {
                        "r2": float(row.get("r2", np.nan)),
                        "rmse": float(row.get("rmse", np.nan)),
                        "mae": float(row.get("mae", np.nan)),
                    }
            except Exception as exc:
                logger.warning("Could not parse %s: %s", summary_path, exc)

        targets = set(best_by_target.keys())
        available_models: Dict[str, List[str]] = {}
        if self.models_dir.exists():
            for file in self.models_dir.glob("*_*.joblib"):
                stem = file.stem
                if "_" not in stem:
                    continue
                target_name, model_name = stem.rsplit("_", 1)
                available_models.setdefault(target_name, []).append(model_name)

        for target, models in available_models.items():
            preferred = best_by_target.get(target)
            load_order = [preferred] if preferred else []
            if "CatBoost" in models and "CatBoost" not in load_order:
                load_order.append("CatBoost")
            for model_name in models:
                if model_name not in load_order:
                    load_order.append(model_name)

            for model_name in load_order:
                model_file = self.models_dir / f"{target}_{model_name}.joblib"
                if not model_file.exists():
                    continue
                try:
                    model = joblib.load(model_file)
                    features = list(getattr(model, "feature_name_", []))
                    if not features:
                        features = list(getattr(model, "feature_names_", []))
                    registry[target] = {
                        "model": model,
                        "name": model_name,
                        "features": features,
                        "path": model_file,
                    }
                    logger.info("Loaded model %s for %s", model_name, target)
                    break
                except Exception as exc:
                    logger.warning("Could not load model %s: %s", model_file, exc)

        return registry, metrics

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------
    def _setup_summarizer(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.info("OPENAI_API_KEY not set. Skipping LLM summarizer.")
            return

        model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")

        if get_openai_callback is None:
            logger.warning("LangChain callback helpers unavailable; skipping cost tracking.")

        try:
            self.summary_llm = ChatOpenAI(
                model=model_name,
                temperature=0.2,
                max_tokens=300,
                timeout=30,
            )
            logger.info("LLM summarizer enabled using model %s", model_name)
        except Exception as exc:
            logger.warning("Failed to initialise LLM summarizer: %s", exc)
            self.summary_llm = None

    def _format_final_response(self, question: str, base_response: str) -> str:
        summary = self._summarize_response(question, base_response)
        usage_line = self._format_usage_line()

        if summary:
            final_message = f"{summary}\n\n---\n{base_response.strip()}"
        else:
            final_message = base_response.strip()

        if usage_line:
            final_message = f"{final_message}\n\n{usage_line}"

        return final_message

    def _summarize_response(self, question: str, base_response: str) -> Optional[str]:
        if not self.summary_llm:
            return None

        context = (
            "You are a pharmaceutical manufacturing assistant. "
            "Rewrite the facts for an executive audience using only the supplied information. "
            "Never introduce new numbers or claims."
        )
        prompt = (
            f"Question:\n{question.strip()}\n\n"
            f"Facts to use verbatim:\n{base_response.strip()}\n\n"
            "Write a concise summary (<=120 words), highlight key metrics, and mention if data is missing."
        )

        callback_ctx = self.callback_factory() if (self.callback_factory and get_openai_callback is not None) else nullcontext(_EmptyCallback())

        try:
            with callback_ctx as cb:  # type: ignore[attr-defined]
                result = self.summary_llm.invoke(
                    [
                        SystemMessage(content=context),
                        HumanMessage(content=prompt),
                    ]
                )
            # callback manager may be the dummy placeholder
            if hasattr(cb, "total_tokens") and not getattr(cb, "is_dummy", False):
                self._update_usage(cb)  # type: ignore[arg-type]
            return result.content.strip()
        except Exception as exc:
            logger.warning("LLM summarization failed: %s", exc)
            return None

    def _update_usage(self, callback_stats: Any) -> None:
        try:
            prompt_tokens = float(getattr(callback_stats, "prompt_tokens", 0.0))
            completion_tokens = float(getattr(callback_stats, "completion_tokens", 0.0))
            total_tokens = float(getattr(callback_stats, "total_tokens", prompt_tokens + completion_tokens))
            total_cost = float(getattr(callback_stats, "total_cost", 0.0))
        except Exception:  # pragma: no cover - guard against unexpected callback payloads
            return

        self.usage_totals["requests"] += 1
        self.usage_totals["prompt_tokens"] += prompt_tokens
        self.usage_totals["completion_tokens"] += completion_tokens
        self.usage_totals["total_tokens"] += total_tokens
        self.usage_totals["total_cost"] += total_cost
        self.last_llm_stats = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost,
        }

    def _format_usage_line(self) -> Optional[str]:
        if not self.summary_llm or not self.last_llm_stats:
            return None

        last = self.last_llm_stats
        totals = self.usage_totals
        return (
            "LLM usage — last reply: "
            f"prompt {int(last['prompt_tokens'])} tok, completion {int(last['completion_tokens'])} tok, "
            f"cost ${last['cost']:.5f}. Session total: {int(totals['requests'])} call(s), "
            f"{int(totals['total_tokens'])} tok, cost ${totals['total_cost']:.5f}."
        )

    # ------------------------------------------------------------------
    # Query handlers
    # ------------------------------------------------------------------
    def _handle_prediction(self, question: str, normalized: str) -> Optional[str]:
        trigger_words = {"predict", "estimate", "expected", "forecast"}
        if not any(word in normalized for word in trigger_words):
            return None

        target = self._identify_target(normalized)
        if not target:
            # Try a lightweight guess based on available models
            for candidate in ["dissolution_av", "batch_yield", "impurities_total"]:
                if candidate in self.model_registry:
                    target = candidate
                    break

        if not target:
            return "I could not identify which metric to predict. Try specifying dissolution, yield, or impurities."

        provided_parameters = self._extract_parameters(question)
        if not provided_parameters:
            return (
                "I need parameter values (for example `api_water=1.6, compression=4.2, speed=105`) "
                "to run the prediction."
            )

        prediction, used_row, defaults_applied = self._predict_with_model(target, provided_parameters)
        if prediction is None:
            return "A trained model for that target is not available in this demo."

        metric_name = self.metric_display_names.get(target, target)
        model_info = self.model_registry[target]
        metrics = self.model_metrics.get(target, {})

        provided_summary = ", ".join(
            f"{self._prettify_feature(feature)}={self._format_number(value)}"
            for feature, value in provided_parameters.items()
        )

        if not provided_summary:
            provided_summary = "Parameters were filled with historical medians."

        defaults_note = ""
        if defaults_applied:
            defaults_note = (
                "Remaining model features were filled with historical medians to keep the input valid."
            )

        model_line = f"Model: {model_info['name']}"
        if metrics.get("r2") is not None:
            model_line += f" | Test R² {metrics['r2']:.3f}"
        if metrics.get("rmse") is not None and not np.isnan(metrics["rmse"]):
            model_line += f" | RMSE {metrics['rmse']:.2f}"

        response_lines = [
            f"**{metric_name} prediction:** {self._format_number(prediction)}",
            model_line,
            f"Inputs interpreted: {provided_summary}",
        ]

        if defaults_note:
            response_lines.append(defaults_note)

        top_diff_features = [f for f in defaults_applied if f in provided_parameters]
        if not top_diff_features and defaults_applied:
            response_lines.append(
                f"Key features filled from data: {', '.join(self._prettify_feature(f) for f in defaults_applied[:5])}."
            )

        paired_targets = {
            "dissolution_av": "batch_yield",
            "batch_yield": "dissolution_av",
            "impurities_total": "dissolution_av",
        }
        paired = paired_targets.get(target)
        if paired and paired in self.dataset_summary:
            summary = self.dataset_summary[paired]
            response_lines.append(
                f"Historical {self.metric_display_names.get(paired, paired)}: "
                f"mean {self._format_number(summary['mean'])}, best {self._format_number(summary['p90'])}."
            )

        logger.info("Prediction completed for %s", target)
        return "\n".join(response_lines)

    def _handle_batch_comparison(self, question: str, normalized: str) -> Optional[str]:
        if "batch" not in normalized or "compare" not in normalized:
            return None

        batch_numbers = re.findall(r"batch\s*(\d+)", normalized)
        if not batch_numbers:
            batch_numbers = re.findall(r"\b(\d{1,4})\b", normalized)
        batch_ids = sorted({int(num) for num in batch_numbers})
        if not batch_ids:
            return "Please specify which batch numbers to compare."

        if "batch" not in self.master_df.columns:
            return "Batch information is unavailable in the dataset."

        subset = self.master_df[self.master_df["batch"].isin(batch_ids)]
        if subset.empty:
            return f"Could not find batches {batch_ids} in the dataset."

        metrics = [
            "dissolution_av",
            "dissolution_min",
            "batch_yield",
            "impurities_total",
            "impurity_o",
            "impurity_l",
            "resodual_solvent",
            "tbl_speed_mean",
            "main_CompForce mean",
        ]
        available_metrics = [metric for metric in metrics if metric in subset.columns]
        comparison_df = subset[["batch"] + available_metrics].copy()
        comparison_df.sort_values("batch", inplace=True)

        table = comparison_df.set_index("batch").round(2).to_string()
        response_lines = ["Batch comparison:", "```", table, "```"]

        if len(batch_ids) >= 2:
            try:
                diff = comparison_df.set_index("batch").diff().dropna()
                if not diff.empty:
                    latest_diff = diff.iloc[-1]
                    deltas = [
                        f"{self._prettify_feature(col)} Δ {self._format_number(val)}"
                        for col, val in latest_diff.items()
                        if abs(val) > 0.05
                    ]
                    if deltas:
                        response_lines.append("Recent change: " + "; ".join(deltas))
            except Exception:
                pass

        return "\n".join(response_lines)

    def _handle_metric_summary(self, question: str, normalized: str) -> Optional[str]:
        keywords = ["average", "mean", "median", "trend", "distribution"]
        if not any(word in normalized for word in keywords):
            return None

        target = self._identify_target(normalized)
        if not target:
            return None

        summary = self.dataset_summary.get(target)
        if not summary:
            return f"I do not have summary statistics for {target}."

        metric_name = self.metric_display_names.get(target, target)
        response_lines = [
            f"{metric_name}: mean {self._format_number(summary['mean'])}, median {self._format_number(summary['median'])}.",
            f"Range observed: {self._format_number(summary['min'])} – {self._format_number(summary['max'])}.",
            f"Typical spread (10th-90th percentile): {self._format_number(summary['p10'])} – {self._format_number(summary['p90'])}.",
        ]

        if target == "dissolution_av" and "dissolution_min" in self.dataset_summary:
            response_lines.append(
                f"Minimum dissolution typically tracks at {self._format_number(self.dataset_summary['dissolution_min']['mean'])}."
            )

        top_batches = self.master_df.nlargest(3, target)[["batch", target]] if "batch" in self.master_df else None
        if top_batches is not None:
            values = ", ".join(
                f"Batch {int(row['batch'])}: {self._format_number(row[target])}"
                for _, row in top_batches.iterrows()
            )
            response_lines.append(f"Top recent batches: {values}.")

        return "\n".join(response_lines)

    def _handle_feature_influence(self, question: str, normalized: str) -> Optional[str]:
        influence_words = ["affect", "influence", "drivers", "important", "impact"]
        if not any(word in normalized for word in influence_words):
            return None

        target = self._identify_target(normalized)
        if not target:
            return None

        ranking = self.feature_importance.get(target)
        if not ranking:
            return f"I do not have feature importance data for {target}."

        filtered_entries: List[Tuple[str, float]] = []
        for feature, score in ranking:
            if self._sanitize_text(feature) == self._sanitize_text(target):
                continue
            if target == "dissolution_av" and "drug release" in feature.lower():
                continue
            filtered_entries.append((feature, score))
        top_entries = filtered_entries[:6]
        insights: List[str] = []
        for feature, score in top_entries:
            if feature in self.numeric_columns and target in self.numeric_columns:
                corr = self.master_df[feature].corr(self.master_df[target])
                corr_text = f"corr {corr:.2f}" if not np.isnan(corr) else "corr n/a"
            else:
                corr_text = "corr n/a"
            insights.append(
                f"{self._prettify_feature(feature)} (importance {score:.2f}, {corr_text})"
            )

        response_lines = [
            f"Key drivers for {self.metric_display_names.get(target, target)}:",
            "- " + "\n- ".join(insights[:5]),
        ]

        return "\n".join(response_lines)

    def _handle_recommendation(self, question: str, normalized: str) -> Optional[str]:
        action_words = ["improve", "increase", "reduce", "optimize", "decrease"]
        if not any(word in normalized for word in action_words):
            return None

        target = self._identify_target(normalized)
        if not target:
            if "yield" in normalized:
                target = "batch_yield"
            elif "impur" in normalized:
                target = "impurities_total"
            elif "dissolution" in normalized:
                target = "dissolution_av"

        if not target or target not in self.feature_importance:
            return None

        high = self.master_df
        low = self.master_df
        if target in self.numeric_columns:
            series = self.master_df[target].dropna()
            if not series.empty:
                threshold_high = series.quantile(0.75)
                threshold_low = series.quantile(0.25)
                high = self.master_df[self.master_df[target] >= threshold_high]
                low = self.master_df[self.master_df[target] <= threshold_low]

        suggestions: List[str] = []
        for feature, _ in self.feature_importance[target][:6]:
            if feature not in self.master_df.columns or feature not in self.numeric_columns:
                continue
            high_mean = high[feature].mean()
            low_mean = low[feature].mean()
            if np.isnan(high_mean) or np.isnan(low_mean):
                continue
            delta = high_mean - low_mean
            if abs(delta) < 1e-3:
                continue
            direction = "higher" if delta > 0 else "lower"
            suggestions.append(
                f"{self._prettify_feature(feature)} tends to be {direction} (Δ {self._format_number(abs(delta))}) in top-performing batches."
            )
            if len(suggestions) >= 3:
                break

        if not suggestions:
            return None

        guardrail = ""
        if target == "batch_yield" and "dissolution_av" in self.dataset_summary:
            guardrail = (
                f"Maintain dissolution around {self._format_number(self.dataset_summary['dissolution_av']['mean'])}% "
                "while applying these adjustments."
            )
        elif target == "impurities_total" and "resodual_solvent" in self.dataset_summary:
            guardrail = (
                f"Watch residual solvent (typical {self._format_number(self.dataset_summary['resodual_solvent']['mean'])}%) as you lower impurities."
            )

        response_lines = [
            f"Actionable levers for {self.metric_display_names.get(target, target)}:",
            "- " + "\n- ".join(suggestions),
        ]
        if guardrail:
            response_lines.append(guardrail)

        return "\n".join(response_lines)

    def _fallback_response(self) -> str:
        lines = [
            "I can help with:",
            "- Summaries (e.g., 'What is the average dissolution rate?')",
            "- Batch comparisons (e.g., 'Compare batch 5 vs 25')",
            "- Feature drivers (e.g., 'Which parameters influence batch yield?')",
            "- ML predictions (e.g., 'Predict dissolution for api_water=1.5, compression=4.2, speed=100')",
        ]

        if self.dataset_summary:
            key_metric = self.dataset_summary.get("dissolution_av")
            if key_metric:
                lines.append(
                    f"Current dissolution performance sits around {self._format_number(key_metric['mean'])}% on average."
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _predict_with_model(
        self, target: str, provided: Dict[str, Any]
    ) -> Tuple[Optional[float], Dict[str, Any], List[str]]:
        if target not in self.model_registry:
            return None, {}, []

        registry_entry = self.model_registry[target]
        model = registry_entry["model"]
        feature_names = registry_entry.get("features", [])
        if not feature_names:
            return None, {}, []

        used_row: Dict[str, Any] = {}
        defaults_applied: List[str] = []
        for feature in feature_names:
            value = self._resolve_feature_value(feature, provided)
            value = self._encode_if_needed(feature, value)
            if value is None:
                default_value = self.default_values.get(feature)
                value = self._encode_if_needed(feature, default_value)
                defaults_applied.append(feature)
            if value is None:
                value = 0.0
            used_row[feature] = value

        input_df = pd.DataFrame([used_row], columns=feature_names)
        try:
            prediction = float(model.predict(input_df)[0])
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", target, exc)
            return None, used_row, defaults_applied

        return prediction, used_row, defaults_applied

    def _resolve_feature_value(self, feature: str, provided: Dict[str, Any]) -> Optional[Any]:
        if feature in provided:
            return provided[feature]

        if feature == "weight_consistency":
            max_w = provided.get("tbl_max_weight")
            min_w = provided.get("tbl_min_weight")
            if max_w is not None and min_w is not None:
                return max_w - min_w

        if feature == "normalized_yield":
            yield_val = provided.get("batch_yield")
            batch_size = provided.get("Batch Size (tablets)")
            if yield_val is not None and batch_size not in (None, 0):
                return (yield_val / batch_size) * 1000

        if feature == "main_CompForce_median" and "main_CompForce mean" in provided:
            return provided["main_CompForce mean"]

        return None

    def _encode_if_needed(self, feature: str, value: Optional[Any]) -> Optional[float]:
        if value is None or feature not in self.label_encoders:
            return value

        mapping = self.label_encoders[feature]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if float(value).is_integer() and int(value) in mapping.values():
                return float(value)
            value_key = str(int(round(value))) if float(value).is_integer() else str(value)
        else:
            value_key = str(value)

        if value_key not in mapping:
            if mapping:
                return float(np.median(list(mapping.values())))
            return None
        return float(mapping[value_key])

    def _identify_target(self, normalized_question: str) -> Optional[str]:
        for metric, aliases in self.metric_aliases.items():
            for alias in aliases:
                if alias in normalized_question:
                    return metric
        return None

    def _extract_parameters(self, question: str) -> Dict[str, Any]:
        pattern = r"(?:^|[\s,;])([A-Za-z0-9_\(\)%]+)\s*[=:]\s*([-+]?\d*\.?\d+)"
        matches = re.findall(pattern, question)
        parameters: Dict[str, Any] = {}
        for raw_name, raw_value in matches:
            feature = self._resolve_feature_name(raw_name)
            if not feature:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            parameters[feature] = value
        return parameters

    def _resolve_feature_name(self, raw_name: str) -> Optional[str]:
        sanitized = self._sanitize_text(raw_name)
        if sanitized in self.column_lookup:
            return self.column_lookup[sanitized]
        return self.parameter_alias_lookup.get(sanitized)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def _prettify_feature(self, feature: str) -> str:
        return feature.replace("_", " ").replace("  ", " ").strip().capitalize()

    @staticmethod
    def _format_number(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        if isinstance(value, (int, np.integer)):
            return f"{int(value)}"
        try:
            if abs(float(value)) >= 100:
                return f"{float(value):.1f}"
            return f"{float(value):.2f}"
        except Exception:
            return str(value)


def main() -> None:
    agent = QualiGenixAgent()
    demo_questions = [
        "What is the average dissolution rate in our dataset?",
        "Compare batch 5 vs batch 25 quality metrics",
        "Predict dissolution for api_water=1.5, compression=4.2, speed=100",
        "Which process parameters most affect batch yield?",
        "How can we improve batch yield while maintaining quality?",
    ]
    for question in demo_questions:
        print("\n>", question)
        print(agent.query(question))


if __name__ == "__main__":  # pragma: no cover
    main()
