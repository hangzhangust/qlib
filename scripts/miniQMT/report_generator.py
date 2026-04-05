"""Generate HTML backtest report with charts and risk metrics.

Usage:
    from scripts.miniQMT.report_generator import generate_html_report
    generate_html_report(report, analysis, output_dir="./output")
"""

import base64
import io
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_html_report(report, analysis, output_dir=".", benchmark_name="SH000300"):
    """Generate an HTML backtest report with embedded charts and metrics.

    Parameters
    ----------
    report : pd.DataFrame
        Backtest report with columns: return, bench, cost, etc.
    analysis : dict
        Risk analysis dict with "risk" key containing metric name-value pairs.
    output_dir : str
        Directory to save the report files.
    benchmark_name : str
        Benchmark display name.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")

    # Compute cumulative returns
    strategy_return = report["return"] - report["cost"]
    bench_return = report["bench"]
    excess_return = strategy_return - bench_return

    cum_strategy = (1 + strategy_return).cumprod() - 1
    cum_bench = (1 + bench_return).cumprod() - 1
    cum_excess = (1 + excess_return).cumprod() - 1

    # Chart 1: Cumulative Returns
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(cum_strategy.index, cum_strategy.values, label="Strategy (net)", linewidth=1.5)
    ax1.plot(cum_bench.index, cum_bench.values, label=f"Benchmark ({benchmark_name})", linewidth=1.5)
    ax1.plot(cum_excess.index, cum_excess.values, label="Excess Return", linewidth=1.5, linestyle="--")
    ax1.set_title("Cumulative Returns")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    chart1_b64 = _fig_to_base64(fig1)
    plt.close(fig1)

    # Chart 2: Drawdown
    cum_excess_total = (1 + excess_return).cumprod()
    running_max = cum_excess_total.cummax()
    drawdown = (cum_excess_total - running_max) / running_max

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red")
    ax2.set_title("Excess Return Drawdown")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    chart2_b64 = _fig_to_base64(fig2)
    plt.close(fig2)

    # Build metrics table
    risk_metrics = analysis.get("risk", {})
    metrics_rows = ""
    for name, value in risk_metrics.items():
        metrics_rows += f"<tr><td>{name}</td><td>{value:.6f}</td></tr>\n"

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Backtest Report {date_str}</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
h2 {{ color: #555; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 10px 0; }}
table {{ border-collapse: collapse; width: 60%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background-color: #4CAF50; color: white; }}
tr:nth-child(even) {{ background-color: #f2f2f2; }}
.meta {{ color: #888; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Backtest Report</h1>
<p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Benchmark: {benchmark_name}</p>

<h2>Cumulative Returns</h2>
<img src="data:image/png;base64,{chart1_b64}" alt="Cumulative Returns">

<h2>Excess Return Drawdown</h2>
<img src="data:image/png;base64,{chart2_b64}" alt="Drawdown">

<h2>Risk Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{metrics_rows}
</table>

</body>
</html>"""

    # Save files
    html_path = output_dir / f"backtest_report_{date_str}.html"
    html_path.write_text(html, encoding="utf-8")

    report_csv_path = output_dir / f"report_{date_str}.csv"
    report.to_csv(report_csv_path)

    analysis_csv_path = output_dir / f"analysis_{date_str}.csv"
    pd.DataFrame.from_dict(risk_metrics, orient="index", columns=["value"]).to_csv(analysis_csv_path)

    return str(html_path)


def _fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
