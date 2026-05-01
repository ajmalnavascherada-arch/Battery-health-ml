import scipy.io
import pandas as pd


def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    battery = data['B0005']
    cycles = battery[0][0]['cycle'][0]

    records = []

    for cycle in cycles:
        if cycle['type'][0] == 'discharge':
            capacity = cycle['data'][0][0]['Capacity'][0][0]
            voltage = cycle['data'][0][0]['Voltage_measured'][0]
            current = cycle['data'][0][0]['Current_measured'][0]

            records.append({
                "capacity": capacity,
                "avg_voltage": voltage.mean(),
                "max_voltage": voltage.max(),
                "min_voltage": voltage.min(),
                "avg_current": current.mean()
            })

    df = pd.DataFrame(records)
    return df