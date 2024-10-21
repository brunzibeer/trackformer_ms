# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT dataset with tracking training augmentations.
"""
import bisect
import copy
import csv
import os
import random

from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from . import transforms as T
from .coco import make_coco_transforms
from .coco import build as build_coco


class MOTSynth(Dataset):
    SEQ_FRAMES = 1800

    def __init__(self, img_folder,
                ann_folder,
                split_file,
                transforms,
                norm_transforms,
                prev_frame_range=None,
                return_masks=False,
                overflow_boxes=False,
                remove_no_obj_imgs=False,
                prev_frame_rnd_augs=None,
                num_frames=3,
        ):
        super(MOTSynth, self).__init__()

        self._img_folder = img_folder
        self._ann_folder = ann_folder
        self._split_file = split_file
        self._transforms = transforms
        self._norm_transforms = norm_transforms
        self._prev_frame_range = prev_frame_range
        self._return_masks = return_masks
        self._overflow_boxes = overflow_boxes
        self._remove_no_obj_imgs = remove_no_obj_imgs
        self._prev_frame_rnd_augs = prev_frame_rnd_augs
        self._num_frames = num_frames
        
        self._split_sqeuences = Path(self._split_file).read_text().splitlines()

        self._vid_id_map = {}
        self._indices = []
        self._labels = {}
        for i, vid_ann_path in enumerate(sorted(Path(self._ann_folder).iterdir())):
            if vid_ann_path.stem not in self._split_sqeuences:
                continue
            self._vid_id_map[vid_ann_path.stem] = i
            self._indices.extend((vid_ann_path.stem, frame) for frame in range(self.SEQ_FRAMES - (self._num_frames - 1)))
            self._labels[vid_ann_path.stem] = torch.load(vid_ann_path)

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    @property
    def frame_range(self):
        if 'frame_range' in self.coco.dataset:
            return self.coco.dataset['frame_range']
        else:
            return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)
    
    def _load_single_frame(self, vid, idx: int):
        # Load the actual frame
        img_path = self._img_folder / vid / 'rgb' / f"{idx:04d}.png"
        img = Image.open(img_path).convert("RGB")

        w, h = img._size
        assert w > 0 and h > 0, f"invalid image {img_path} with shape {w} {h}"
        obj_idx_offset = self._vid_id_map[vid] * 100000  # 100000 unique ids is enough for a video.

        targets = {
            'boxes': [],
            'labels': [],
            'image_id': torch.as_tensor(idx),
            'track_ids': [],
            # 'area': None,
            # 'iscrowd': None,
            'orig_size': None,
            'size': None,
        }

        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        video_annotations = self._labels[vid]

        for frame_annotations in video_annotations[video_annotations[:, 0] == idx]:  # frame_id,x1,y1,x2,y2,valid,tracking_id,dist,visibility,x,y,z
            x1, y1, x2, y2, valid, track_id, _, _, _, _, _ = frame_annotations[1:]
            if valid == 0:
                continue

            targets['boxes'].append([x1, y1, x2, y2])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)
            targets['track_ids'].append(track_id + obj_idx_offset)
            targets['scores'].append(1.)
            targets['noisy'].append(0)
        
        return img, targets

    def _load_sampled_frames(self, vid, indices):
        imgs = []
        targets = []
        for idx in indices:
            img, target = self._load_single_frame(vid, idx)
            imgs.append(img)
            targets.append(target)

        to_return = {}
        if len(imgs) == 2:
            to_return = {
                **targets[0],
                'prev_image': imgs[1],
                'prev_target': targets[1]
            }
        elif len(imgs) == 3:
            to_return = {
                **targets[0],
                'prev_image': imgs[1],
                'prev_target': targets[1],
                'prev_prev_image': imgs[2],
                'prev_prev_target': targets[2]
            }

        return imgs[0], to_return


    def __getitem__(self, idx):
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()
        }
        
        vid, frame_idx = self._indices[idx]

        indices = [frame_idx]
        if self._num_frames > 1:
            prev_frame_id = random.randint(
                max(0, frame_idx - self._prev_frame_range),
                min(frame_idx + self._prev_frame_range, self.SEQ_FRAMES - (self._num_frames - 1))
            )
            indices.append(prev_frame_id)
            if self._num_frames > 2:
                assert self._num_frames == 3
                
                prev_prev_frame_id = min(
                    max(0, prev_frame_id + prev_frame_id - frame_idx),
                    self.SEQ_FRAMES - (self._num_frames - 1)
                )
                indices.append(prev_prev_frame_id)

        images, targets = self._load_sampled_frames(vid, indices)


    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):

    def sample_weight(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if hasattr(self.datasets[dataset_idx], 'sample_weight'):
            return self.datasets[dataset_idx].sample_weight(sample_idx)
        else:
            return 1 / len(self.datasets[dataset_idx])


def build_mot(image_set, args):
    root = Path()

    if image_set == 'train':
        args.mot_path_train = f'/homes/mbernardi/git/trackformer_ms/{args.mot_path_train}'
        root = Path(args.mot_path_train)
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range=args.track_prev_frame_range
    elif image_set == 'val':
        args.mot_path_val = f'/homes/mbernardi/git/trackformer_ms/{args.mot_path_val}'
        root = Path(args.mot_path_val)
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided MOT17Det path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = root / split
    ann_file = root / f"annotations/{split}.json"

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform, args.overflow_boxes)

    dataset = MOT(
        img_folder, ann_file, transforms, norm_transforms,
        prev_frame_range=prev_frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame=args.tracking,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
        prev_prev_frame=args.track_prev_prev_frame,
        )

    return dataset

def build_motsynth(image_set, args):
    root = Path(args.motsynth_path)
    split_files = Path(args.motsynth_split_files)
    prev_frame_rnd_augs = None
    prev_frame_range = None

    if image_set == 'train':
        prev_frame_rnd_augs = args.track_prev_frame_rnd_augs
        prev_frame_range=args.track_prev_frame_range
    elif image_set == 'val':
        prev_frame_rnd_augs = 0.0
        prev_frame_range = 1
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided MOTSynth path {root} does not exist'

    split = getattr(args, f"{image_set}_split")

    img_folder = root / 'frames'
    ann_folder = root / 'npy_annotations/annotations_clean'
    split_file = split_files / f"{split}.txt"

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.img_transform, args.overflow_boxes)
    
    dataset = MOTSynth(
        img_folder,
        ann_folder,
        split_file,
        transforms,
        norm_transforms,
        prev_frame_range=prev_frame_range,
        return_masks=args.masks,
        overflow_boxes=args.overflow_boxes,
        remove_no_obj_imgs=False,
        prev_frame_rnd_augs=prev_frame_rnd_augs,
    )

    return dataset