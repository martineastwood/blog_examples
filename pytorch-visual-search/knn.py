import argparse
import components.search


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-n", "--n_items", default=6, type=int, help="Number of items to return")
    ap.add_argument(
        "-i", "--image", help="Path to the input image")
    ap.add_argument(
        "-f", "--features", help="Directory containing the features")
    ap.add_argument(
        "-o", "--output", help="Path to store the output in")

    args = vars(ap.parse_args())

    components.search.knn(
        args["image"], args["output"], args["features"], args["n_items"])
