import anomalies


if __name__ == '__main__':
    path = 'SKAB/other/11.csv'

    # Находим оптимальное число кластеров с помощью метода Elbow criterion (по графику)
    #anomalies.lokot(path)


    # Считаем метрики с подобранным числом кластеров методом Elbow criterion
    # result, num_clusters = anomalies.best_clusters_prefind(path, n_clusters=3)
    # print(result)
    # print(num_clusters)

    # Перебираем число кластеров от 1 до 10 и выбираем лучший результат
    result, num_clusters = anomalies.best_clusters(path)
    print(result)
    print(num_clusters)

