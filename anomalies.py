import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns


def preprocess(path='SKAB/anomaly-free/anomaly-free.csv'):
    rfm_data = pd.read_csv(path, delimiter=';')
    data = rfm_data.iloc[:, 1:]
    orig_data = data
    rfm_norm = standart(data)
    return rfm_norm, orig_data


def standart(data):
    scaler = StandardScaler()
    scaler.fit(data)
    rfm_norm = scaler.transform(data)
    rfm_norm = pd.DataFrame(rfm_norm, index=data.index, columns=data.columns)
    return rfm_norm


def norm(data):
    scaler = preprocessing.MinMaxScaler()
    names = data.columns
    res = scaler.fit_transform(data)
    res = pd.DataFrame(res, index=data.index, columns=names)
    return res


def num_clusters(df):
    # Рассчитываю оптимальное количество кластеров используя Elbow criterion метод
    sse = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        sse[k] = kmeans.inertia_
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()


def get_clusters(rfm_data, df, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    kmeans.fit(df)
    cluster_labels = kmeans.labels_
    # добавляю колонку 'Кластеры' в датасет с оригинальными данными
    rfm_data = rfm_data.assign(Cluster=cluster_labels)
    # рассчитываю средние значения переменных и размер каждого кластера
    rfm_data_grouped = rfm_data.groupby(['Cluster']).agg(
        {'Accelerometer1RMS': 'mean', 'Accelerometer2RMS': 'mean',
         'Current': 'mean', 'Pressure': 'mean',
         'Temperature': 'mean', 'Thermocouple': 'mean',
         'Voltage': 'mean', 'Volume Flow RateRMS': 'mean'})
    df = pd.DataFrame(df,
                      index=rfm_data.index,
                      columns=rfm_data.columns)
    df['Cluster'] = rfm_data['Cluster']
    return df, rfm_data, kmeans


def snake_plot(data):
    data_melt = pd.melt(data.reset_index(),
                        id_vars=['Cluster'],
                        value_vars=['Accelerometer1RMS', 'Accelerometer2RMS',
                                    'Current', 'Pressure', 'Temperature', 'Thermocouple',
                                    'Voltage', 'Volume Flow RateRMS'],
                        var_name='Attribute',
                        value_name='Value')
    plt.title('Snake plot of standardized variables')
    sns.lineplot(x="Attribute",
                 y="Value",
                 hue="Cluster",
                 data=data_melt)
    plt.show()


def heatmap(data, orig_data):
    cluster_avg = data.groupby(['Cluster']).mean()
    orig_data = orig_data.iloc[:, :-1]
    population_avg = orig_data.mean()
    relative_imp = cluster_avg / population_avg - 1
    relative_imp.round(2)
    plt.figure(figsize=(8, 2))
    plt.title('Relative importance of attributes')
    sns.heatmap(data=relative_imp, annot=True, fmt='.2f'
                , cmap='RdYlGn')
    plt.show()


def anomaly(data, kmeans):
    x_cluster_centers = kmeans.cluster_centers_
    res = []
    data_nc = data.iloc[:, :-1]
    tmp = data.values.tolist()
    tmp_nc = data_nc.values.tolist()
    dists = []
    for num, item in enumerate(tmp):
        cluster = int(item[8])
        dist = abs(tmp_nc[num] - x_cluster_centers[cluster])
        dists.append(dist)
    dists = pd.DataFrame(dists, index=data.index, columns=data_nc.columns)
    dists = dists.assign(Cluster=data["Cluster"])
    for num, item in enumerate(tmp):
        cluster = int(item[8])
        dist = abs(tmp_nc[num] - x_cluster_centers[cluster])
        km_y_pred = []
        for stat, value in enumerate(dist):
            temp = dists[dists["Cluster"] == cluster].iloc[:, stat].tolist()
            p = np.percentile(temp, 97)
            if value >= p:
                km_y_pred.append(1)
            else:
                km_y_pred.append(0)
        res.append(km_y_pred)
    df = pd.DataFrame(res, index=data.index, columns=data_nc.columns)
    return df


def anomaly_params(data, kmeans):
    anomaly_matrix = anomaly(data, kmeans)
    anomaly_matrix = anomaly_matrix.assign(Fraud_score=anomaly_matrix.sum(axis=1))
    # изолирую нормализованные данные с аномальными значениями
    anomaly_anomaly = anomaly_matrix[(anomaly_matrix['Accelerometer1RMS'] == 1)
                                     | (anomaly_matrix['Accelerometer2RMS'] == 1)
                                     | (anomaly_matrix['Current'] == 1)
                                     | (anomaly_matrix['Pressure'] == 1)
                                     | (anomaly_matrix['Temperature'] == 1)
                                     | (anomaly_matrix['Thermocouple'] == 1)
                                     | (anomaly_matrix['Voltage'] == 1)
                                     | (anomaly_matrix['Volume Flow RateRMS'] == 1)]
    rfm_norm_anomaly = data[data.index.isin(anomaly_anomaly.index)]
    rfm_norm_not_anomaly = data[~data.index.isin(anomaly_anomaly.index)]
    # рассчитываю количество измерений, по которым 2 и более аномальных значения данных
    anomaly_2 = anomaly_anomaly[(anomaly_anomaly['Fraud_score'] >= 2)]
    print('Количество измерений с аномальным поведением (1 и более аномалий): ', len(anomaly_anomaly))
    print('Количество измерений с аномальным поведением (2 и более аномалии): ', anomaly_2.shape[0])

    # # Getting the Centroids
    # centroids = kmeans.cluster_centers_
    # u_labels = np.unique(label)
    #
    # # plotting the results:
    #
    # for i in u_labels:
    #     plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    # plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k)
    # plt.legend()
    # plt.show()
