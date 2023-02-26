import anomalies


if __name__ == '__main__':
    df = anomalies.preprocess()
    #anomalies.num_clusters(df)
    anomalies.get_clusters(df)

