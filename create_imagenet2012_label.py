
"""create_imagenet2012_label"""
import os
import json
import argparse

parser = argparse.ArgumentParser(description="ghostnet imagenet2012 label")
parser.add_argument("--img_path", type=str, required=True, help="imagenet2012 file path.")
args = parser.parse_args()


def create_label(file_path):
    """
    create_imagenet2012_label
    Args:
        file_path:

    Returns:

    """
    print("[WARNING] Create imagenet label. Currently only use for Imagenet2012!")
    dirs = os.listdir(file_path)
    file_list = []
    for file in dirs:
        file_list.append(file)
    file_list = sorted(file_list)

    total = 0
    img_label = {}
    for i, file_dir in enumerate(file_list):
        files = os.listdir(os.path.join(file_path, file_dir))
        for f in files:
            img_label[f] = i
        total += len(files)

    with open("imagenet_label.json", "w+") as label:
        json.dump(img_label, label)

    print("[INFO] Completed! Total {} data.".format(total))


if __name__ == '__main__':
    create_label(args.img_path)
