"""
reports.py — Auto-generate paper-ready tables and figures
=========================================================

Generates:
- Table 2 (main results) CSV
- Table 4 (literature comparison) CSV
- Confusion matrix figures
- Per-fold accuracy distribution
- Statistical comparison tables
- Markdown summary

"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generate paper-ready tables and figures from LOSO/LODO results.

    Usage
    -----
    >>> gen = ReportGenerator(output_dir='./loso_results')
    >>> gen.generate_tables({'XGBoost': loso_xgb.results_,
    ...                       'LDA': loso_lda.results_,
    ...                       'CNN1D': loso_cnn.results_})
    """

    def __init__(self, output_dir: Union[str, Path] = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_tables(self, results: Dict[str, Dict],
                         dataset_name: str = "DB2") -> Dict[str, Path]:
        """Generate all tables (CSV + Markdown)."""
        paths: Dict[str, Path] = {}

        # Main results table (Table 2)
        paths["main_results"] = self._table_main_results(results, dataset_name)
        # Per-fold table
        paths["per_fold"] = self._table_per_fold(results, dataset_name)
        # Per-class F1
        paths["per_class"] = self._table_per_class(results, dataset_name)
        # Statistical comparison
        paths["statistics"] = self._table_statistics(results, dataset_name)
        return paths

    def _table_main_results(self, results: Dict[str, Dict],
                              dataset_name: str) -> Path:
        rows = []
        for model_name, res in results.items():
            agg = res["aggregate"]
            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Accuracy_mean": agg["accuracy_mean"],
                "Accuracy_std": agg["accuracy_std"],
                "MacroF1_mean": agg["macro_f1_mean"],
                "MacroF1_std": agg["macro_f1_std"],
                "Inflation_pp": agg.get("inflation_mean_pp", 0.0),
                "N_folds": res["n_folds"],
                "N_samples": res["n_samples"],
                "N_features": res["n_features"],
            })
        df = pd.DataFrame(rows)
        path = self.output_dir / f"Table2_main_results_{dataset_name}.csv"
        df.to_csv(path, index=False, float_format="%.4f")
        logger.info(f"Main results table → {path}")
        # Also markdown
        md_path = path.with_suffix(".md")
        df.to_markdown(md_path, index=False, floatfmt=".4f")
        return path

    def _table_per_fold(self, results: Dict[str, Dict],
                          dataset_name: str) -> Path:
        rows = []
        for model_name, res in results.items():
            for fold in res["per_fold"]:
                rows.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Fold": fold["fold"],
                    "Test_subject": fold["test_subject"],
                    "Accuracy": fold["metrics"]["accuracy"],
                    "MacroF1": fold["metrics"]["macro_f1"],
                    "Inflation_pp": fold["metrics"]["inflation"],
                    "Elapsed_s": fold["elapsed_seconds"],
                })
        df = pd.DataFrame(rows)
        path = self.output_dir / f"per_fold_{dataset_name}.csv"
        df.to_csv(path, index=False, float_format="%.4f")
        return path

    def _table_per_class(self, results: Dict[str, Dict],
                          dataset_name: str) -> Path:
        rows = []
        for model_name, res in results.items():
            # Take first fold as representative (or average)
            for fold in res["per_fold"][:1]:
                for cls, m in fold["metrics"]["per_class"].items():
                    rows.append({
                        "Dataset": dataset_name,
                        "Model": model_name,
                        "Fold": fold["fold"],
                        "Class": cls,
                        "Precision": m["precision"],
                        "Recall": m["recall"],
                        "F1": m["f1"],
                        "Support": m["support"],
                    })
        df = pd.DataFrame(rows)
        path = self.output_dir / f"per_class_{dataset_name}.csv"
        df.to_csv(path, index=False, float_format="%.4f")
        return path

    def _table_statistics(self, results: Dict[str, Dict],
                            dataset_name: str) -> Path:
        from myoadapt.evaluation.statistics import (
            friedman_test,
            wilcoxon_pairwise,
        )
        per_fold = {n: r["aggregate"]["per_fold_accuracies"]
                    for n, r in results.items()}
        friedman = friedman_test(per_fold)
        wilcoxon = wilcoxon_pairwise(per_fold, reference=list(results.keys())[0])

        rows = []
        for t in wilcoxon["tests"]:
            rows.append({
                "Dataset": dataset_name,
                "Model_A": t["model_a"],
                "Model_B": t["model_b"],
                "Statistic": t["statistic"],
                "p_raw": t["p_value_raw"],
                "p_corrected": t["p_value_corrected"],
                "cohen_d": t.get("cohen_d", 0.0),
                "significant": t["significant"],
            })
        df = pd.DataFrame(rows)
        path = self.output_dir / f"statistics_{dataset_name}.csv"
        df.to_csv(path, index=False, float_format="%.4f")

        # Also save Friedman summary
        with open(self.output_dir / f"friedman_{dataset_name}.json", "w") as f:
            json.dump(friedman, f, indent=2, default=str)

        return path

    def generate_summary_markdown(self, results: Dict[str, Dict],
                                    dataset_name: str = "DB2") -> Path:
        """Generate a Markdown summary report."""
        path = self.output_dir / f"summary_{dataset_name}.md"
        with open(path, "w") as f:
            f.write(f"# LOSO Results — {dataset_name}\n\n")
            f.write("## Main Results\n\n")
            f.write("| Model | Accuracy | Macro-F1 | Inflation (pp) |\n")
            f.write("|-------|----------|----------|----------------|\n")
            for name, res in results.items():
                agg = res["aggregate"]
                f.write(f"| {name} | {agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f} "
                        f"| {agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f} "
                        f"| {agg.get('inflation_mean_pp', 0)*100:.2f} |\n")
            f.write("\n## Per-fold Accuracies\n\n")
            for name, res in results.items():
                accs = res["aggregate"]["per_fold_accuracies"]
                f.write(f"**{name}**: {', '.join(f'{a:.3f}' for a in accs)}\n\n")
        logger.info(f"Summary markdown → {path}")
        return path
