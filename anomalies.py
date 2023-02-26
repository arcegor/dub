import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns


# преобразую переменные используя boxcox метод
def boxcox_df(x):
    x_boxcox, _ = stats.boxcox(x)
    return x_boxcox


def preprocess(path='SKAB/anomaly-free/anomaly-free.csv'):
    rfm_data = pd.read_csv(path, delimiter=';')
    data = rfm_data.iloc[:, 1:]
    # rfm_data_boxcox = data.apply(boxcox_df, axis=0)
    # нормализую данные используя StandardScaler()
    scaler = StandardScaler()
    scaler.fit(data)
    rfm_norm = scaler.transform(data)
    rfm_norm = pd.DataFrame(rfm_norm, index=data.index, columns=data.columns)
    return rfm_norm


def num_clusters(df):
    # Рассчитываю оптимальное количество кластеров используя Elbow criterion метод
    sse = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        sse[k] = kmeans.inertia_
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()


def get_clusters(df, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    kmeans.fit(df)
    cluster_labels = kmeans.labels_
    # добавляю колонку 'Кластеры' в датасет с оригинальными данными
    rfm_data = df.assign(Cluster=cluster_labels)
    # рассчитываю средние значения переменных и размер каждого кластера
    rfm_data_grouped = rfm_data.groupby('Cluster').agg(
        {'Accelerometer1RMS': 'mean', 'Accelerometer2RMS': 'mean',
         'Current': 'mean', 'Pressure': 'mean',
         'Temperature': 'mean', 'Thermocouple': 'mean',
         'Voltage': 'mean', 'Volume Flow RateRMS': 'mean'})
    # добавляю колонку 'Кластер'
    rfm_norm_clusters = df.assign(Cluster=cluster_labels)
    sns.heatmap(rfm_data_grouped)
    df = pd.DataFrame(df,
                      index=rfm_data.index,
                      columns=rfm_data.columns)
    df['Cluster'] = rfm_data_grouped['Cluster']
    data_melt = pd.melt(df.reset_index(),
                        id_vars=['CustomerID', 'Cluster'],
                        value_vars=['Accelerometer1RMS', 'Accelerometer2RMS',
                                    'Current', 'Pressure', 'Temperature', 'Thermocouple',
                                    'Voltage', 'Volume Flow RateRMS'],
                        var_name='Attribute', value_name='Value')
    plt.title('Snake plot of standardized variables')
    sns.lineplot(x="Attribute",
                 y="Value",
                 hue='Cluster',
                 data=data_melt)
