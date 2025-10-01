import numpy as np
from proposed.utils import distance

def kmeans(nodes_pos, k, init_points=None, max_iter=100):
    # nodes_pos: list of (x,y)
    n = len(nodes_pos)
    # init centroids by sampling k nodes
    if init_points is not None:
        centroids = init_points
    else:
        centroids = [nodes_pos[i] for i in np.random.choice(range(n), k, replace=False)]

    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for i, p in enumerate(nodes_pos):
            dists = [distance(p, c) for c in centroids]
            idx = int(np.argmin(dists))
            clusters[idx].append(i)
        new_centroids = []
        for c in clusters:
            if len(c)==0:
                # reinitialize empty centroid
                new_centroids.append(nodes_pos[np.random.randint(0,n)])
            else:
                xs = [nodes_pos[i][0] for i in c]
                ys = [nodes_pos[i][1] for i in c]
                new_centroids.append((np.mean(xs), np.mean(ys)))
        # check convergence
        if all(distance(centroids[i], new_centroids[i]) < 1e-3 for i in range(k)):
            break
        centroids = new_centroids
    return clusters, centroids