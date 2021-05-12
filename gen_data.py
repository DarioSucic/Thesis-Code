import numpy as np

def generate_data(seed, n_clusters, n_points_per_cluster):

    np.random.seed(seed)
    medoids = np.random.randint(-128, 128, size=(n_clusters, 2))

    clusters = []

    for medoid in medoids:
        theta = np.random.uniform(0, 2*np.pi, size=(n_points_per_cluster, 1))
        
        xs, ys = np.cos(theta), np.sin(theta)
        xs[0], ys[0] = 0, 0

        cluster = medoid + np.hstack((xs, ys))
        clusters.append(cluster)

    data = np.vstack(clusters)

    return data


if __name__ == "__main__":
    SEED = 69
    
    sizes = [
        ("tiny"  ,  2,    4),
        ("small" ,  4,   32),
        ("medium",  8,  256),
        ("large" , 16, 2048),
        ("huge"  , 32, 4096)
    ]

    for (size_name, n_clusters, n_points_per_cluster) in sizes:
        data = generate_data(SEED, n_clusters, n_points_per_cluster)
        with open(f"benches/test_data/{size_name}", "w") as file:
            np.savetxt(file, data)