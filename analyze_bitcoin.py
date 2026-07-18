import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
SENTIMENT_PATH = ROOT / "fear_greed_index.csv"
TRADER_PATH = ROOT / "historical_data.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def resolve_path(path_value, fallback):
    if path_value is None:
        return fallback
    path = Path(path_value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def normalize_sentiment_columns(sentiment):
    sentiment = sentiment.copy()
    rename_map = {}
    for col in sentiment.columns:
        low = col.lower()
        if low == "date":
            rename_map[col] = "Date"
        elif low in {"classification", "sentiment", "label", "mood"}:
            rename_map[col] = "Classification"
        elif low == "value":
            rename_map[col] = "value"
    if rename_map:
        sentiment = sentiment.rename(columns=rename_map)

    if "Date" not in sentiment.columns:
        raise ValueError("Sentiment data must contain a Date column.")
    if "Classification" not in sentiment.columns:
        raise ValueError("Sentiment data must contain a Classification column.")

    sentiment = sentiment[["Date", "Classification", "value"]].copy()
    sentiment["Date"] = pd.to_datetime(sentiment["Date"], errors="coerce").dt.normalize()
    sentiment = sentiment.dropna(subset=["Date"])
    return sentiment


def normalize_trader_columns(trader):
    trader = trader.copy()
    rename_map = {}
    for col in trader.columns:
        low = col.lower()
        if low in {"time", "date", "datetime", "timestamp"}:
            rename_map[col] = "Date"
        elif low in {"account", "account_id", "trader", "trader_id"}:
            rename_map[col] = "account"
        elif low in {"closedpnl", "closed_pnl", "pnl", "profit_loss"}:
            rename_map[col] = "closedPnL"
        elif low in {"size", "quantity", "position_size", "qty", "size tokens", "size_tokens"}:
            rename_map[col] = "size"
        elif low in {"price", "execution price", "execution_price"}:
            rename_map[col] = "execution_price"
        elif low in {"leverage", "lev"}:
            rename_map[col] = "leverage"
        elif low in {"side", "direction"}:
            rename_map[col] = "side"
        elif low in {"start position", "start_position"}:
            rename_map[col] = "start_position"
    if rename_map:
        trader = trader.rename(columns=rename_map)

    if "Date" not in trader.columns:
        raise ValueError("Trader data must contain a Date or time column.")

    if "Timestamp IST" in trader.columns:
        trader["Date"] = pd.to_datetime(trader["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce")
    else:
        trader["Date"] = pd.to_datetime(trader["Date"], errors="coerce")
    trader["Date"] = trader["Date"].dt.normalize()

    if "account" not in trader.columns:
        trader["account"] = trader.index.astype(str)
    if "closedPnL" not in trader.columns:
        trader["closedPnL"] = 0.0
    if "size" not in trader.columns:
        trader["size"] = 1.0
    if "leverage" not in trader.columns:
        trader["leverage"] = 1.0
    if "side" not in trader.columns:
        trader["side"] = "UNKNOWN"
    return trader


def load_data(sentiment_path: Path, trader_path: Path):
    sentiment = pd.read_csv(sentiment_path)
    trader = pd.read_csv(trader_path)
    sentiment = normalize_sentiment_columns(sentiment)
    trader = normalize_trader_columns(trader)
    return sentiment, trader


def prepare_data(sentiment, trader):
    sentiment = sentiment.copy()
    trader = trader.copy()

    trader["Date"] = pd.to_datetime(trader["Date"], errors="coerce").dt.normalize()
    sentiment["Date"] = pd.to_datetime(sentiment["Date"], errors="coerce").dt.normalize()

    daily = (
        trader.groupby(["Date", "account"])
        .agg(
            total_closed_pnl=("closedPnL", "sum"),
            total_size=("size", "sum"),
            avg_leverage=("leverage", "mean"),
            trade_count=("account", "size"),
            buy_trades=("side", lambda s: (s.astype(str).str.upper() == "BUY").sum()),
            sell_trades=("side", lambda s: (s.astype(str).str.upper() == "SELL").sum()),
        )
        .reset_index()
    )

    sentiment = sentiment[["Date", "Classification", "value"]].copy()
    merged = daily.merge(sentiment, on="Date", how="left")
    merged["Classification"] = merged["Classification"].fillna("Unknown")
    merged["value"] = merged["value"].fillna(0)
    return merged


def summarize_results(merged):
    summary = (
        merged.groupby("Classification")
        .agg(
            accounts=("account", "nunique"),
            trades=("trade_count", "sum"),
            avg_pnl=("total_closed_pnl", "mean"),
            median_pnl=("total_closed_pnl", "median"),
            total_pnl=("total_closed_pnl", "sum"),
            avg_size=("total_size", "mean"),
            avg_leverage=("avg_leverage", "mean"),
            avg_sentiment_score=("value", "mean"),
        )
        .reset_index()
    )
    return summary


def plot_results(summary):
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(data=summary, x="Classification", y="avg_pnl", ax=axes[0])
    axes[0].set_title("Average PnL by Market Sentiment")
    axes[0].set_ylabel("Avg closed PnL")
    axes[0].set_xlabel("Sentiment")

    sns.barplot(data=summary, x="Classification", y="trades", ax=axes[1])
    axes[1].set_title("Trading Activity by Market Sentiment")
    axes[1].set_ylabel("Total trades")
    axes[1].set_xlabel("Sentiment")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sentiment_vs_trading.png", dpi=200)
    plt.close(fig)


def write_report(summary, output_path: Path):
    lines = [
        "Bitcoin sentiment vs trader performance summary",
        "============================================",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['Classification']}: avg pnl={row['avg_pnl']:.2f}, total pnl={row['total_pnl']}, trades={row['trades']}, avg leverage={row['avg_leverage']:.2f}, avg sentiment={row['avg_sentiment_score']:.1f}"
        )

    best_class = summary.loc[summary["avg_pnl"].idxmax(), "Classification"] if not summary.empty else "N/A"
    worst_class = summary.loc[summary["avg_pnl"].idxmin(), "Classification"] if not summary.empty else "N/A"

    lines.extend(
        [
            "",
            "Interpretation:",
            "--------------",
            f"- The strongest average performance appears in {best_class}.",
            f"- The weakest average performance appears in {worst_class}.",
            "- Use these results as a starting point for testing sentiment-aware position sizing and risk controls.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze bitcoin sentiment versus trader performance")
    parser.add_argument("--sentiment", type=Path, default=SENTIMENT_PATH, help="Path to sentiment CSV")
    parser.add_argument("--trader", type=Path, default=TRADER_PATH, help="Path to trader CSV")
    args = parser.parse_args()

    sentiment_path = resolve_path(args.sentiment, SENTIMENT_PATH)
    trader_path = resolve_path(args.trader, TRADER_PATH)

    if not sentiment_path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_path}")
    if not trader_path.exists():
        raise FileNotFoundError(f"Trader file not found: {trader_path}")

    sentiment, trader = load_data(sentiment_path, trader_path)
    merged = prepare_data(sentiment, trader)
    summary = summarize_results(merged)

    print("Merged analysis sample:")
    print(merged.head().to_string(index=False))
    print("\nSummary by sentiment:")
    print(summary.to_string(index=False))

    plot_results(summary)
    write_report(summary, OUTPUT_DIR / "summary_report.txt")
    print(f"\nSaved chart to {OUTPUT_DIR / 'sentiment_vs_trading.png'}")
    print(f"Saved report to {OUTPUT_DIR / 'summary_report.txt'}")


if __name__ == "__main__":
    main()
