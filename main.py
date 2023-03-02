import anomalies


if __name__ == '__main__':
    # df, orig = anomalies.preprocess()
    #anomalies.num_clusters(df)
    # df, orig, kmeans = anomalies.get_clusters(orig, df)
    #anomalies.snake_plot(df)
    #anomalies.heatmap(df, orig)
    # anomalies.anomaly_params(df, kmeans)
    result, num_clusters = anomalies.best_clusters()
    print(result)
    print(num_clusters)

