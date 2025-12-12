"""
Compare model metrics across training runs.
Generates comparison reports and visualizations.
"""
import json
from pathlib import Path
from datetime import datetime
import argparse


def load_metrics(metrics_file: Path = Path("training_logs/model_metrics.json")) -> list:
    """Load all saved metrics."""
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return []
    
    with open(metrics_file) as f:
        return json.load(f)


def print_comparison_report(metrics_list: list):
    """Print formatted comparison of all training runs."""
    if not metrics_list:
        print("No metrics found to compare.")
        return
    
    print("\n" + "="*100)
    print("TRAINING RUNS COMPARISON REPORT")
    print("="*100)
    
    for idx, entry in enumerate(metrics_list, 1):
        timestamp = entry["timestamp"]
        metrics = entry["metrics"]
        hparams = entry["hyperparameters"]
        
        print(f"\nRun #{idx}: {timestamp}")
        print(f"  Hyperparameters: epochs={hparams['epochs']}, batch_size={hparams['batch_size']}, lr={hparams['learning_rate']}")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Precision:        {metrics['precision']:.4f}")
        print(f"  Recall:           {metrics['recall']:.4f}")
        print(f"  F1-Score:         {metrics['f1_score']:.4f}")
        print(f"  Dice Coefficient: {metrics['dice_coefficient']:.4f}")
        print(f"  IoU (Jaccard):    {metrics['iou']:.4f}")
    
    # Show best run
    if len(metrics_list) > 1:
        print("\n" + "-"*100)
        best_f1_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['metrics']['f1_score'])
        best_entry = metrics_list[best_f1_idx]
        print(f"Best Run (by F1-Score): Run #{best_f1_idx + 1}")
        print(f"  Timestamp: {best_entry['timestamp']}")
        print(f"  F1-Score:  {best_entry['metrics']['f1_score']:.4f}")
    
    print("="*100 + "\n")


def get_latest_metrics(metrics_list: list) -> dict:
    """Get metrics from the latest training run."""
    if not metrics_list:
        return {}
    return metrics_list[-1]


def generate_markdown_table(metrics_list: list) -> str:
    """Generate markdown table of metrics for README."""
    if not metrics_list:
        return ""
    
    lines = ["| Run | Accuracy | Precision | Recall | F1-Score | Dice | IoU | Timestamp |"]
    lines.append("|-----|----------|-----------|--------|----------|------|-----|-----------|")
    
    for idx, entry in enumerate(metrics_list, 1):
        m = entry["metrics"]
        ts = entry["timestamp"].split("T")[0]  # Date only
        line = f"| {idx} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['dice_coefficient']:.4f} | {m['iou']:.4f} | {ts} |"
        lines.append(line)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare model metrics across training runs")
    parser.add_argument("--metrics_file", default="training_logs/model_metrics.json", help="Path to metrics JSON file")
    parser.add_argument("--format", choices=["text", "markdown"], default="text", help="Output format")
    
    args = parser.parse_args()
    
    metrics_list = load_metrics(Path(args.metrics_file))
    
    if args.format == "markdown":
        print(generate_markdown_table(metrics_list))
    else:
        print_comparison_report(metrics_list)


if __name__ == "__main__":
    main()
