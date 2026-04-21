def build_features(df):
    features = []

    for _, row in df.iterrows():
        features.append([
            row.get("fb_polarity", 0),
            row.get("fb_score", 0),
            row.get("sent_rolling_3", 0),
            row.get("sent_rolling_7", 0),
            row.get("sent_rolling_20", 0),

            row.get("known_pumper", 0),
            row.get("price_region", 0),
            row.get("inflection_point", 0),

            row.get("hour", 12),
            row.get("dayofweek", 0),
            row.get("is_weekend", 0),
            row.get("is_market_hours", 1),

            row.get("rsi_proxy", 50),
            row.get("macd_proxy", 0),
            row.get("sma_ratio_proxy", 1),
            row.get("volume_proxy", 0),
        ])

    return features