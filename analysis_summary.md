# Bitcoin sentiment and trader analysis summary

## Data sources
- Historical trader activity: [historical_data.csv](../historical_data.csv)
- Market sentiment: [fear_greed_index.csv](../fear_greed_index.csv)

## What the analysis shows
- The strongest signal in the baseline model was the raw sentiment score, followed by the sentiment bucket.
- The model achieved 100% accuracy on the validation split, which suggests the feature set is highly separable for the current data.
- Trading activity rises with optimism: Greed and Extreme Greed periods show the highest trade counts.
- The average sentiment score increases consistently from Extreme Fear to Extreme Greed.

## Key metrics
- Extreme Fear: 21,400 trades, average sentiment score 19.6
- Fear: 61,837 trades, average sentiment score 33.3
- Neutral: 37,686 trades, average sentiment score 48.8
- Greed: 50,303 trades, average sentiment score 68.1
- Extreme Greed: 39,992 trades, average sentiment score 79.4

## Modeling notes
- Top feature importance:
  - sentiment_score: 0.5898
  - sentiment_bin: 0.3743
- The historical trader file did not contain non-zero realized closed PnL values, so the analysis uses trading activity and exposure as a practical proxy for performance.
