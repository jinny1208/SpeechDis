import argparse

import yaml
import sys

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    sys.argv = ['preprocess.py', 'config/LibriTTS/preprocess.yaml']
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
