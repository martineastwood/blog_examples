import argparse
import components.features


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i", "--images", help="Directory containing the images")
    ap.add_argument(
        "-o", "--output", help="Directory to store the output in")

    args = vars(ap.parse_args())

    components.features.extract_features(args["images"], args["output"])
