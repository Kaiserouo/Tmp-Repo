import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess original jsonl data into ")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    args = parser.parse_args()

def main(args):
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)