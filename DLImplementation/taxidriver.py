import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt

def simulate_ride_distances():
    logging.info('Simulating ride distances!')
    ride_dists = np.concatenate(
        (
            10 * np.random.random(size=370),
            30 * np.random.random(size=10),  # long distances
            10 * np.random.random(size=10),  # same distance
            10 * np.random.random(size=10)  # same distance
        )
    )
    return ride_dists

def simulate_ride_speeds():
    logging.info('simulating ride speeds')
    ride_speeds = np.concatenate(
        (
            np.random.normal(loc=30, scale=5, size=370),
            np.random.normal(loc=30, scale=5, size=10), # same speed
            np.random.normal(loc=50, scale=10, size=10), # high speed
            np.random.normal(loc=15, scale=4, size=10) #low speed
        )
    )
    return ride_speeds

def simulate_ride_data():
    ride_dists = simulate_ride_distances()
    ride_speeds = simulate_ride_speeds()
    ride_times = ride_dists / ride_speeds

    # assemble into data frame 
    df = pd.DataFrame(
        {
            'ride_dist': ride_dists,
            'ride_time': ride_times,
            'ride_speed': ride_speeds
        }
    )

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn import metrics

def plot_cluster_results(data, labels, core_samples_mask, n_clusters_):
    fig = plt.figure(figsize=(10,10))
    unique_labels = set(labels)
    colors = [plt.cm.cool(each) 
            for each in np.linspace(0,1,len(unique_labels))]


def cluster_and_label(data, create_and_show_plot=True):
    data = StandardScaler().fit_transform(data)
    db = DBSCAN(eps=0.3, min_samples=10).fit(data)

    # find labels from the clustering
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_]=True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    run_metadata = {

    }


if __name__ == "__main__":
    import os