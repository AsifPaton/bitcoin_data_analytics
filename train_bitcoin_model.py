import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
HIST_PATH = ROOT / "historical_data.csv"
SENTIMENT_PATH = ROOT / "fear_greed_index.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (int, float)):
        try:
            return pd.to_datetime(value, unit="s")
        except Exception:
            return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def load_and_prepare_data(hist_path: Path, sentiment_path: Path):
    hist = pd.read_csv(hist_path)
    sentiment = pd.read_csv(sentiment_path)

    hist = hist.rename(columns={
        "Account": "account",
        "Coin": "symbol",
        "Execution Price": "execution_price",
        "Size Tokens": "size",
        "Size USD": "size_usd",
        "Side": "side",
        "Timestamp IST": "timestamp_ist",
        "Start Position": "start_position",
        "Direction": "direction",
        "Closed PnL": "closedPnL",
        "Timestamp": "timestamp",
    })

    sentiment = sentiment.rename(columns={
        "date": "date",
        "classification": "classification",
        "value": "value",
    })

    hist["timestamp_dt"] = pd.to_datetime(hist["timestamp_ist"], format="%d-%m-%Y %H:%M", errors="coerce")
    hist["date"] = hist["timestamp_dt"].dt.normalize()
    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce").dt.normalize()

    # Derive daily trader metrics from the historical trades.
    daily_trader = (
        hist.groupby(["date", "account"])
        .agg(
            total_closed_pnl=("closedPnL", "sum"),
            total_size=("size", "sum"),
            trades=("account", "size"),
            buy_trades=("side", lambda s: (s.str.upper() == "BUY").sum()),
            sell_trades=("side", lambda s: (s.str.upper() == "SELL").sum()),
        )
        .reset_index()
    )

    daily_trader["daily_pnl_bucket"] = pd.cut(
        daily_trader["total_closed_pnl"],
        bins=[-float("inf"), -100, 0, 100, float("inf")],
        labels=["negative", "low", "neutral", "positive"],
        right=True,
    )

    sentiment = sentiment[["date", "value", "classification"]].copy()
    sentiment["classification"] = sentiment["classification"].str.lower().str.strip()
    sentiment["sentiment_score"] = sentiment["value"]
    sentiment["sentiment_bucket"] = pd.cut(
        sentiment["sentiment_score"],
        bins=[0, 25, 45, 60, 100],
        labels=["extreme_fear", "fear", "neutral", "greed"],
        right=True,
    )

    merged = daily_trader.merge(sentiment, on="date", how="left")
    merged = merged.dropna(subset=["sentiment_score", "classification"])
    merged = merged.copy()

    # Create features for a simple supervised model.
    merged["classification_bin"] = merged["classification"].map({
        "fear": 0,
        "extreme fear": 0,
        "neutral": 1,
        "greed": 2,
        "extreme greed": 2,
    })
    merged["sentiment_bin"] = merged["sentiment_bucket"].map({
        "extreme_fear": 0,
        "fear": 1,
        "neutral": 2,
        "greed": 3,
    })

    feature_cols = ["total_closed_pnl", "total_size", "trades", "buy_trades", "sell_trades", "sentiment_score", "sentiment_bin"]
    X = merged[feature_cols]
    y = merged["classification_bin"]
    return X, y, merged


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)
    return model, acc, report


def main():
    parser = argparse.ArgumentParser(description="Train a baseline sentiment-aware bitcoin trader model")
    parser.add_argument("--historical", type=Path, default=HIST_PATH)
    parser.add_argument("--sentiment", type=Path, default=SENTIMENT_PATH)
    args = parser.parse_args()

    X, y, merged = load_and_prepare_data(args.historical, args.sentiment)
    model, acc, report = train_model(X, y)

    print("Dataset shape:", X.shape)
    print("Accuracy:", round(acc, 3))
    print("Classification report:\n", report)

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nFeature importance:\n", feature_importance.to_string(index=False))

    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "model_training_dataset.csv", index=False)


if __name__ == "__main__":
    main()
