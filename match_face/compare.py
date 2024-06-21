import pandas as pd

# Cargar el archivo Excel
df = pd.read_excel('All_quick.xlsx')

# Calcular la precisión del reconocimiento para las filas con verified igual a VERDADERO
df['precision'] = df.apply(lambda row: (row['distance'] / row['threshold']) * 100 if row['verified'] == True else 0, axis=1)

# Agrupar por modelo, detector_backend y distance_metric y calcular métricas
grouped = df.groupby(['model', 'detector_backend', 'distance_metric']).agg(
    precision_mean=('precision', 'mean'),
    true_count=('verified', lambda x: (x == True).sum()),
    false_count=('verified', lambda x: (x == False).sum())
).reset_index()

# Calcular la tasa de fallos
grouped['failure_rate'] = grouped['false_count'] / (grouped['true_count'] + grouped['false_count'])

# Ordenar por precisión y tasa de fallos
grouped = grouped.sort_values(by=['precision_mean', 'failure_rate'], ascending=[False, True])

print(grouped)