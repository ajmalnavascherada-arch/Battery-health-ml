def create_labels(df):
    def label_health(cap):
        if cap > 1.8:
            return "Healthy"
        elif cap > 1.5:
            return "Aged"
        else:
            return "Degraded"

    df["label"] = df["capacity"].apply(label_health)
    return df


def create_features(df):
    df["voltage_range"] = df["max_voltage"] - df["min_voltage"]
    df["power_proxy"] = df["avg_voltage"] * df["avg_current"]
    return df