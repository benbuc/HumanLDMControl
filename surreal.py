import datasets
import json
import os
import cv2
import numpy as np
import scipy.io as sio
from pathlib import Path
from PIL import Image
import random
from datasets import load_dataset


_CITATION = ""
_DESCRIPTION = ""
_URL = ""
_HOMEPAGE = ""
_LICENSE = ""


class Surreal(datasets.GeneratorBasedBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_path = Path("./SURREAL/")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "video_file": datasets.Value("string"),
                    "pose_video": datasets.Value("string"),
                    "frame_index": datasets.Value("int32"),
                    "prompt": datasets.Value("string"),
                    "smpl_shape": datasets.Sequence(datasets.Value("float32")),
                    "smpl_pose": datasets.Sequence(datasets.Value("float32")),
                }
            ),
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_path": self.base_path / "cmu" / "train" / "run0"},
            ),
        ]

    def _generate_examples(self, data_path):
        info_files = sorted(list((data_path).rglob("*_info.mat")))

        with open(self.base_path / "surreal_humanldm.json", "rt") as f:
            caption_data = json.load(f)

        caption_data = {
            (x["target"].split("/")[-1].replace(".mp4", "_info.mat"), x["frame_index"]): x for x in caption_data
        }

        idx = 0
        for info_file in info_files:
            info_data = sio.loadmat(info_file)

            smpl_shapes = info_data["shape"]
            smpl_poses = info_data["pose"]

            for frame_id in range(smpl_shapes.shape[1]):
                video_filename = info_file.name.replace("_info.mat", ".mp4")
                pose_filename = video_filename.replace(".mp4", "_pose.mp4")
                yield idx, {
                    "video_file": info_file.parent / video_filename,
                    "pose_video": data_path / "out_pose" / pose_filename,
                    "frame_index": frame_id,
                    "prompt": caption_data[(info_file.name, frame_id)]["prompt"],
                    "smpl_shape": smpl_shapes[:, frame_id],
                    "smpl_pose": smpl_poses[:, frame_id],
                }
                idx += 1


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("surreal.py")
    print(dataset)
    print("Dataset loaded successfully.")