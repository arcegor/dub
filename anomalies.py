import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler


def preprocess(path='SKAB/other/9.csv'):
    rfm_data = pd.read_csv(path, delimiter=';')
    data = rfm_data.iloc[:, 1:-2]
    y = rfm_data.iloc[:, -2]
    orig_data = data
    rfm_norm = standart(data)
    return rfm_norm, orig_data, y


def standart(data):
    scaler = StandardScaler()
    scaler.fit(data)
    rfm_norm = scaler.transform(data)
    rfm_norm = pd.DataFrame(rfm_norm, index=data.index, columns=data.columns)
    return rfm_norm


# def norm(data):
#     scaler = preprocessing.MinMaxScaler()
#     names = data.columns
#     res = scaler.fit_transform(data)
#     res = pd.DataFrame(res, index=data.index, columns=names)
#     return res


def num_clusters(df):
    # Рассчитываю оптимальное количество кластеров используя Elbow criterion метод
    sse = {}
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        sse[k] = kmeans.inertia_
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()


# def get_clusters(rfm_data, df, n_clusters=5):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=1)
#     kmeans.fit(df)
#     cluster_labels = kmeans.labels_
#     # добавляю колонку 'Кластеры' в датасет с оригинальными данными
#     rfm_data = rfm_data.assign(Cluster=cluster_labels)
#     # рассчитываю средние значения переменных и размер каждого кластера
#     rfm_data_grouped = rfm_data.groupby(['Cluster']).agg(
#         {'Accelerometer1RMS': 'mean', 'Accelerometer2RMS': 'mean',
#          'Current': 'mean', 'Pressure': 'mean',
#          'Temperature': 'mean', 'Thermocouple': 'mean',
#          'Voltage': 'mean', 'Volume Flow RateRMS': 'mean'})
#     df = pd.DataFrame(df,
#                       index=rfm_data.index,
#                       columns=rfm_data.columns)
#     df['Cluster'] = rfm_data['Cluster']
#     return df, rfm_data, kmeans


# def snake_plot(data):
#     data_melt = pd.melt(data.reset_index(),
#                         id_vars=['Cluster'],
#                         value_vars=['Accelerometer1RMS', 'Accelerometer2RMS',
#                                     'Current', 'Pressure', 'Temperature', 'Thermocouple',
#                                     'Voltage', 'Volume Flow RateRMS'],
#                         var_name='Attribute',
#                         value_name='Value')
#     plt.title('Snake plot of standardized variables')
#     sns.lineplot(x="Attribute",
#                  y="Value",
#                  hue="Cluster",
#                  data=data_melt)
#     plt.show()
#
#
# def heatmap(data, orig_data):
#     cluster_avg = data.groupby(['Cluster']).mean()
#     orig_data = orig_data.iloc[:, :-1]
#     population_avg = orig_data.mean()
#     relative_imp = cluster_avg / population_avg - 1
#     relative_imp.round(2)
#     plt.figure(figsize=(8, 2))
#     plt.title('Relative importance of attributes')
#     sns.heatmap(data=relative_imp, annot=True, fmt='.2f'
#                 , cmap='RdYlGn')
#     plt.show()


def anomaly(data, kmeans):
    cluster_labels = kmeans.labels_
    data = data.assign(Cluster=cluster_labels)
    x_cluster_centers = kmeans.cluster_centers_
    res = []
    data_nc = data.iloc[:, :-1]
    tmp = data.values.tolist()
    tmp_nc = data_nc.values.tolist()
    dists = []
    for num, item in enumerate(tmp):
        cluster = int(item[-1])
        dist = abs(tmp_nc[num] - x_cluster_centers[cluster])
        dists.append(dist)
    dists = pd.DataFrame(dists, index=data.index, columns=data_nc.columns)
    dists = dists.assign(Cluster=data["Cluster"])
    for num, item in enumerate(tmp):
        cluster = int(item[-1])
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
    # # изолирую нормализованные данные с аномальными значениями
    # anomaly_anomaly = anomaly_matrix[(anomaly_matrix['Accelerometer1RMS'] == 1)
    #                                  | (anomaly_matrix['Accelerometer2RMS'] == 1)
    #                                  | (anomaly_matrix['Current'] == 1)
    #                                  | (anomaly_matrix['Pressure'] == 1)
    #                                  | (anomaly_matrix['Temperature'] == 1)
    #                                  | (anomaly_matrix['Thermocouple'] == 1)
    #                                  | (anomaly_matrix['Voltage'] == 1)
    #                                  | (anomaly_matrix['Volume Flow RateRMS'] == 1)]
    # rfm_norm_anomaly = data[data.index.isin(anomaly_anomaly.index)]
    # rfm_norm_not_anomaly = data[~data.index.isin(anomaly_anomaly.index)]
    # # рассчитываю количество измерений, по которым 2 и более аномальных значения данных
    # anomaly_2 = anomaly_anomaly[(anomaly_anomaly['Fraud_score'] >= 2)]
    # print('Количество измерений с аномальным поведением (1 и более аномалий): ', len(anomaly_anomaly))
    # print('Количество измерений с аномальным поведением (2 и более аномалии): ', anomaly_2.shape[0])
    # graphic(data.iloc[:, :-1], kmeans)
    return anomaly_matrix


#
# def graphic(data, kmeans):
#     pca = PCA(2)
#     label = kmeans.predict(data)
#     u_labels = np.unique(label)
#     df = pca.fit_transform(data)
#     centroids = kmeans.cluster_centers_
#     for i in u_labels:
#         plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
#     plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
#     plt.legend()
#     plt.show()


def f(x):
    if x >= 1:
        return 1
    else:
        return 0


def main(path='SKAB/other/9.csv', n_clusters=5):
    data, _, y = preprocess(path)
    x = data
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    kmeans.fit(x)
    res = anomaly_params(x, kmeans)
    res = res.iloc[:, -1].apply(f)
    accuracy = accuracy_score(y, res)
    precision = precision_score(y, res)
    recall = recall_score(y, res)
    f1 = 2 * (precision * recall) / (precision + recall)
    roc = roc_auc_score(y, res)
    # print('precision = ', precision)
    # print('recall = ', recall)
    # print('f1 = ', f1)
    # print('roc = ', roc)
    result = [accuracy, precision, recall, f1, roc]
    return result, y, res


def best_clusters():
    result = []
    max = 0
    num_clusters = 0
    y = 0
    res_best = 0
    for i in range(1, 11):
        score, y, res = main(n_clusters=i)
        if score[0] > max:
            max = score[0]
            res_best = res
            num_clusters = i
        result.append(score)
    fpr, tpr, thresholds = metrics.roc_curve(y, res_best)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()
    return result[num_clusters], num_clusters

