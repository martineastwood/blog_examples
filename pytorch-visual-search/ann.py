import argparse
import components.search


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-mi", "--make_index", action='store_true')
    ap.add_argument("-s", "--search", action='store_true')

    ap.add_argument(
        "-i", "--image", help="Path to the input image")
    ap.add_argument(
        "-f", "--features", help="Directory containing the features")
    ap.add_argument(
        "-o", "--output", help="Path to store the output in")
    ap.add_argument(
        "-i_path", "--index", help="Path to store the output in")

    args = vars(ap.parse_args())

    if args["make_index"]:
        components.search.ann_index(args["features"], args["output"])

    if args["search"]:
        components.search.ann(args["image"], args["output"], args["features"], args["index"])
