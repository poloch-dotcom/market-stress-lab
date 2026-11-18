"""Run the whole lab on the synthetic market and write figures + a results file.

    python scripts/run_demo.py            # synthetic data, seed 42
    python scripts/run_demo.py --csv my_market.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stresslab import FACTORS, Portfolio, load_csv, simulate_market  # noqa: E402
from stresslab import backtest as bt  # noqa: E402
from stresslab import calibration as cal  # noqa: E402
from stresslab import stress as st  # noqa: E402
from stresslab import var as v  # noqa: E402
from stresslab.data import rolling_moves  # noqa: E402

FIG = ROOT / "reports" / "figures"
ALPHA, H = 0.99, 10

# Colours: categorical slots 1-3 of a colour-blind-checked palette, plus neutrals.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29d", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 1.6,
})


def meur(x: float) -> str:
    return f"{x / 1e6:,.2f} M€"


def shade_stress(ax, market):
    reg = market["regime"].to_numpy()
    idx = market.index
    starts = np.flatnonzero(np.diff(np.r_[0, reg]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[reg, 0]) == -1)
    for s, e in zip(starts, ends):
        ax.axvspan(idx[s], idx[e], color=GRID, lw=0, zorder=0)


def fig_market(market):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    axes[0].plot(market.index, market["SPOT"], color=BLUE)
    axes[0].set_title("Synthetic market: equity index (top) and implied volatility (bottom)")
    axes[0].set_ylabel("index level")
    axes[1].plot(market.index, market["VOL_LEVEL"], color=BLUE)
    axes[1].set_ylabel("implied vol (pts)")
    if "regime" in market:
        for ax in axes:
            shade_stress(ax, market)
        axes[1].text(0.01, 0.92, "grey bands = stress regime", transform=axes[1].transAxes, color=INK2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "market.png", dpi=150)
    plt.close(fig)


def fig_distribution(pnl, lines):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.hist(pnl / 1e6, bins=120, color=MUTED, alpha=0.6, edgecolor=SURFACE, linewidth=0.3)
    for label, value, colour in lines:
        ax.axvline(-value / 1e6, color=colour, lw=1.6, label=f"{label}: {meur(value)}")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlim(np.quantile(pnl, 0.0005) / 1e6 * 1.1, np.quantile(pnl, 0.999) / 1e6)
    ax.set_title("One-day P&L of today's book replayed on history (full revaluation)")
    ax.set_xlabel("P&L (M€)")
    ax.set_ylabel("days")
    fig.tight_layout()
    fig.savefig(FIG / "pnl_distribution.png", dpi=150)
    plt.close(fig)


def fig_backtest(res):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(res.index, res["pnl"] / 1e6, color=MUTED, lw=0.6, label="daily P&L")
    ax.plot(res.index, -res["var_historical"] / 1e6, color=BLUE, label="99% VaR, historical (500d)")
    ax.plot(res.index, -res["var_fhs"] / 1e6, color=ORANGE, label="99% VaR, filtered historical (GARCH)")
    for col, colour, marker in (("var_historical", BLUE, "o"), ("var_fhs", ORANGE, "s")):
        hits = res["pnl"] < -res[col]
        ax.scatter(res.index[hits], res["pnl"][hits] / 1e6, s=22, color=colour, marker=marker,
                   edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.set_title("Rolling backtest: markers are the days the loss exceeded the VaR")
    ax.set_ylabel("M€")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8.5, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "backtest.png", dpi=150)
    plt.close(fig)


def fig_stress(table):
    t = table.sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_axisbelow(True)
    ax.barh(t.index, t.values / 1e6, color=BLUE, height=0.6)
    for y, val in enumerate(t.values):
        ax.text(val / 1e6, y, f"  {meur(val)}", va="center", ha="left", color=SURFACE, fontsize=8.5,
                fontweight="bold")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_title("10-day stress losses on today's book")
    ax.set_xlabel("P&L (M€)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "stress.png", dpi=150)
    plt.close(fig)


def fig_calibration(info, core):
    test = info["test"]
    models = info["models"]
    eq_grid = np.linspace(test["EQ"].min() * 2.5, test["EQ"].quantile(0.9), 200)
    cs_fixed = test["CS"].median()
    grid = pd.DataFrame({"EQ": eq_grid, "CS": cs_fixed})[core]
    train_min = info["train_min_eq"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(test["EQ"] * 100, test["VOL"], s=8, color=MUTED, alpha=0.55, lw=0, label="test-period tail days")
    for name, colour, label in (("gaussian", BLUE, "Gaussian (linear)"),
                                ("gbm", AQUA, "Gradient boosting"),
                                ("hybrid", ORANGE, "Hybrid: linear + boosting on residuals")):
        j = models[name].columns.index("VOL")
        ax.plot(eq_grid * 100, models[name].predict(grid)[:, j], color=colour, label=label)
    ax.axvspan(eq_grid[0] * 100, train_min * 100, color=GRID, lw=0, zorder=0)
    ax.text(0.01, 0.04, "grey band: moves larger than the worst day in the training sample",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    ax.set_title("Implied-vol shock implied by an equity shock (credit spread held at its median)")
    ax.set_xlabel("one-day equity move (%)")
    ax.set_ylabel("implied vol change (pts)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.0f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(x) if isinstance(x, (int, float, np.floating)) and not isinstance(x, bool) else str(x)
                 for x in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="real market levels, see stresslab.data.load_csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)

    market = load_csv(args.csv) if args.csv else simulate_market(seed=args.seed)
    book = Portfolio().at_market(market)
    x = market[FACTORS].to_numpy()
    pnl = pd.Series(book.pnl(x), index=market.index)
    out = [f"# Results ({'real data: ' + args.csv if args.csv else f'synthetic market, seed {args.seed}'})", ""]
    out += [f"Book priced at spot {book.state.spot:,.0f}, implied vol {book.state.implied_vol:.1%}. "
            f"Short puts struck at {book.put_strike:,.0f}.", ""]

    # 1. One-day 99% VaR and ES, several methods
    cov1 = market[FACTORS].cov().to_numpy()
    mc_n = v.monte_carlo(book, cov1, ALPHA, dof=None)
    mc_t = v.monte_carlo(book, cov1, ALPHA, dof=5.0)
    fhs = v.filtered_historical(pnl.to_numpy(), ALPHA)
    var_table = pd.DataFrame({
        "VaR 99%": {
            "Historical (full reval)": v.historical_var(pnl, ALPHA),
            "Normal on P&L": v.normal_var(pnl, ALPHA),
            "Cornish-Fisher": v.cornish_fisher_var(pnl, ALPHA),
            "Delta-normal (linearized)": v.delta_normal_var(book.sensitivities(), cov1, ALPHA),
            "Monte Carlo, Gaussian": mc_n["var"],
            "Monte Carlo, Student-t (5 dof)": mc_t["var"],
            "Filtered historical (GARCH)": fhs["var"],
        },
        "ES 97.5%": {
            "Historical (full reval)": v.historical_es(pnl, 0.975),
            "Normal on P&L": v.normal_es(pnl, 0.975),
            "Cornish-Fisher": np.nan,
            "Delta-normal (linearized)": np.nan,
            "Monte Carlo, Gaussian": v.historical_es(mc_n["pnl"], 0.975),
            "Monte Carlo, Student-t (5 dof)": v.historical_es(mc_t["pnl"], 0.975),
            "Filtered historical (GARCH)": v.filtered_historical(pnl.to_numpy(), 0.975, fit=fhs["fit"])["es"],
        },
    })
    var_table.index.name = "Method (1 day)"
    out += ["## 1. One-day risk measures (EUR)", "", md_table(var_table.fillna(0).round(0)).replace("| 0 |", "| - |"),
            "", f"GARCH on the P&L: alpha = {fhs['fit'].alpha:.3f}, beta = {fhs['fit'].beta:.3f}, "
            f"persistence = {fhs['fit'].persistence:.3f}.", ""]
    fig_market(market)
    fig_distribution(pnl.to_numpy(), [
        ("Historical VaR 99%", var_table.loc["Historical (full reval)", "VaR 99%"], BLUE),
        ("Historical ES 97.5%", var_table.loc["Historical (full reval)", "ES 97.5%"], ORANGE),
        ("Delta-normal VaR 99%", var_table.loc["Delta-normal (linearized)", "VaR 99%"], AQUA),
    ])

    # 2. Rolling backtest
    res = bt.rolling_backtest(pnl, window=500, alpha=ALPHA)
    summary = bt.summarize(res, ALPHA)
    summary.index.name = "VaR model"
    out += ["## 2. Backtest of the 99% one-day VaR", "",
            f"{len(res)} out-of-sample days, 500-day rolling window, GARCH refitted every 20 days.", "",
            md_table(summary, "{}"), ""]
    fig_backtest(res)

    # 3. Stress tests
    scen = st.run_scenarios(book, horizon_days=H)
    worst = st.worst_historical_windows(market, book, horizon=H, top=3)
    start, end = st.stressed_period(pnl)
    moves = rolling_moves(market, H)
    cov_stressed = moves.loc[start:end].cov().to_numpy()
    targets = [2e6, 4e6, 6e6]
    reverse = []
    for L in targets:
        r = st.reverse_stress_test(book, cov_stressed, L, horizon_days=H)
        reverse.append({"loss target": meur(L), **{k: round(val, 3) for k, val in r.as_series().items()},
                        "distance (sigmas)": round(r.mahalanobis, 2), "tail prob.": f"{r.tail_probability:.2%}"})
    reverse = pd.DataFrame(reverse).set_index("loss target")

    scen_out = scen.copy()
    scen_out.index.name = "Scenario (10 days)"
    worst_out = worst.copy()
    worst_out.index = [f"window ending {d}" for d in worst["window_end"]]
    worst_out.index.name = "Historical window (10 days)"
    worst_out = worst_out.drop(columns="window_end")
    worst_out[FACTORS] = worst_out[FACTORS].round(3)
    worst_out["pnl"] = worst_out["pnl"].map(lambda z: f"{z:,.0f}")
    out += ["## 3. Stress tests (10-day horizon)", "", "### Hypothetical scenarios, P&L by position (EUR)", "",
            md_table(scen_out.round(0)), "",
            "### Worst historical windows replayed on today's book", "",
            md_table(worst_out, "{:g}"), "",
            f"### Reverse stress test (stressed covariance: {start.date()} to {end.date()})", "",
            "Most plausible 10-day move producing each loss. EQ and FX in log-return, IR and CS in bp, VOL in vol points.",
            "", md_table(reverse, "{}"), ""]
    all_losses = pd.concat([scen["total"], pd.Series(worst["pnl"].to_numpy(),
                            index=[f"History: 10d to {d}" for d in worst["window_end"]])])
    fig_stress(all_losses)

    # 4. Shock calibration
    core, secondary = ["EQ", "CS"], ["IR", "FX", "VOL"]
    rows = {}
    info_daily = None
    for h in (1, H):
        mv = rolling_moves(market, h)
        table, info = cal.evaluate(mv, core, secondary, horizon=h)
        if h == 1:
            info_daily = info
            split = int(len(mv) * 0.7)
            info_daily["train_min_eq"] = mv["EQ"].iloc[:split - h].min()
        for model in table.index:
            rows[(f"{h}-day", model)] = table.loc[model]
    calib = pd.DataFrame(rows).T
    calib.index = [f"{h}, {m}" for h, m in calib.index]
    calib.index.name = "Horizon, model"
    out += ["## 4. Shock calibration: RMSE on test-period tail moves", "",
            "Core shocks given: EQ and CS. Predicted: IR (bp), FX (log-return), VOL (vol pts). Lower is better.", "",
            md_table(calib, "{:.3f}"), ""]
    fig_calibration(info_daily, core)

    narrative = {"EQ": -0.25, "CS": 150.0}
    filled = cal.complete_scenario(narrative, info["models"]["hybrid"], secondary)
    out += ["Example: the narrative 'equities -25%, spreads +150 bp' completed by the hybrid model (10-day): "
            + ", ".join(f"{k} = {val:+.3g}" for k, val in filled.items()) + ".", ""]

    (ROOT / "reports" / "results.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
