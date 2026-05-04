import scipy.io
import pandas as pd
import os


def load_data(file_path):
    data = scipy.io.loadmat(file_path)

    # Dynamically extract battery name from file
    battery_name = os.path.basename(file_path).replace(".mat", "")

    # Access correct battery key
    if battery_name in data:
        battery = data[battery_name]
    else:
        # fallback: pick first non-meta key
        battery_key = [k for k in data.keys() if not k.startswith("__")][0]
        battery = data[battery_key]
        battery_name = battery_key

    cycles = battery[0][0]['cycle'][0]

    records = []

    for i, cycle in enumerate(cycles):
        if cycle['type'][0] == 'discharge':

            capacity = cycle['data'][0][0]['Capacity'][0][0]
            voltage = cycle['data'][0][0]['Voltage_measured'][0]
            current = cycle['data'][0][0]['Current_measured'][0]

            records.append({
                "battery_id": battery_name,   # ✅ critical for group split
                "cycle_index": i,             # ✅ useful for time-based analysis
                "capacity": capacity,
                "avg_voltage": voltage.mean(),
                "max_voltage": voltage.max(),
                "min_voltage": voltage.min(),
                "avg_current": current.mean()
            })

    df = pd.DataFrame(records)

    return df