from . import features
import numpy as np
import os
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib
from annoy import AnnoyIndex
from sklearn.decomposition import PCA
import joblib
matplotlib.use('Agg')

np.random.seed(0)

import time


def plot_multiple_img(image_paths, output_path, rows=1, cols=6):
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 2))
    for idx, path in enumerate(image_paths):
        im = Image.open(path)
        ax.ravel()[idx].imshow(np.asarray(im))
        ax.ravel()[idx].set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path)
    print(output_path)


def knn(input_path, output_path, features_path, n=6):
    print("[INFO] Instantiating Model")
    model = features.get_model()

    print("[INFO] Instantiating Preprocessing Pipeline")
    preprocess = features.get_preprocess_pipeline()

    print("[INFO] Loading Features")
    feature_list = np.load(os.path.join(features_path, "features.npy"))
    filename_list = np.load(os.path.join(features_path, "filenames.npy"))

    print("[INFO] Training KNN Model")
    neighbors = NearestNeighbors(
        n_neighbors=6, algorithm="brute", metric="euclidean"
    ).fit(feature_list)

    print("[INFO] Extracting Feature Vector")
    im = Image.open(input_path)
    im = preprocess(im)
    im = im.unsqueeze(0)
    with torch.no_grad():
        input_features = model(im).numpy()

    print("[INFO] Finding Similar Images")
    import time
    tic = time.perf_counter()    
    distances, indices = neighbors.kneighbors([input_features[0]], n)
    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
    similar_image_paths = filename_list[indices[0]]

    print("[INFO] Saving Similar Images to {0}".format(output_path))
    plot_multiple_img(similar_image_paths, output_path, 1, n)


def ann_index(features_path, output_path):
    print("[INFO] Loading Features")
    feature_list = np.load(os.path.join(features_path, "features.npy"))

    print("[INFO] Running PCA")
    n_components = 128
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(feature_list)
    joblib.dump(pca, os.path.join(output_path, "pca.joblib"))

    print("[INFO] Building Index")
    feature_length = n_components
    index = AnnoyIndex(feature_length, 'angular')
    for i, j in enumerate(components):
        index.add_item(i, j)

    index.build(15)
    index.save(os.path.join(output_path, "index.annoy"))


def ann(input_path, output_path, features_path, index_path, n=6):
    print("[INFO] Instantiating Model")
    model = features.get_model()

    print("[INFO] Instantiating Preprocessing Pipeline")
    preprocess = features.get_preprocess_pipeline()

    print("[INFO] Loading Image Filename Mapping")
    filename_list = np.load(os.path.join(features_path, "filenames.npy"))

    print("[INFO] Extracting Feature Vector")
    im = Image.open(input_path)
    im = preprocess(im)
    im = im.unsqueeze(0)
    with torch.no_grad():
        input_features = model(im).numpy()

    print("[INFO] Applying PCA")
    pca = joblib.load(os.path.join(features_path, "pca.joblib"))
    components = pca.transform(input_features)[0]

    print("[INFO] Loading ANN Index")
    ann_index = AnnoyIndex(components.shape[0], 'angular')
    ann_index.load(os.path.join(index_path, "index.annoy"))

    print("[INFO] Finding Similar Images")
    indices = ann_index.get_nns_by_vector(components, n, search_k=-1, include_distances=False)
    indices = np.array(indices)
    similar_image_paths = filename_list[indices]

    print("[INFO] Saving Similar Images to {0}".format(output_path))
    plot_multiple_img(similar_image_paths, output_path, 1, n)
