import anomalies


if __name__ == '__main__':
    df, orig = anomalies.preprocess()
    #anomalies.num_clusters(df)
    anomalies.get_clusters(orig, df)

