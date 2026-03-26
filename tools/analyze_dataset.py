import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate RCS stats files for segmentation labels')
    parser.add_argument(
        '--data-root',
        required=True,
        help='Dataset root, where output json files will be written.')
    parser.add_argument(
        '--ann-dir',
        default='ann_dir/train',
        help='Relative or absolute annotation directory.')
    parser.add_argument(
        '--suffix',
        default='.png',
        help='Annotation file suffix.')
    parser.add_argument(
        '--reduce-zero-label',
        action='store_true',
        help='Apply reduce_zero_label transform before counting.')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Optional class count limit after preprocessing (0..num_classes-1).'
    )
    return parser.parse_args()


def preprocess_label(label, reduce_zero_label=False):
    label = label.astype(np.int32)
    if reduce_zero_label:
        # Same convention as MMSeg LoadAnnotations(reduce_zero_label=True):
        # 0 -> 255(ignore), 1..K -> 0..K-1
        zero_mask = label == 0
        label = label - 1
        label[zero_mask] = 255
    return label


def collect_stats(ann_dir, suffix, reduce_zero_label=False, num_classes=None):
    ann_files = sorted(
        f for f in os.listdir(ann_dir)
        if osp.isfile(osp.join(ann_dir, f)) and f.endswith(suffix))
    if not ann_files:
        raise FileNotFoundError(f'No annotation files found in: {ann_dir}')

    sample_class_stats = []
    samples_with_class = defaultdict(list)

    for fname in ann_files:
        path = osp.join(ann_dir, fname)
        label = mmcv.imread(path, flag='unchanged')
        if label is None:
            continue
        if label.ndim == 3:
            # Safety fallback: if image is RGB, use first channel.
            label = label[..., 0]
        label = preprocess_label(label, reduce_zero_label=reduce_zero_label)

        classes, counts = np.unique(label, return_counts=True)
        file_stats = {'file': fname}
        for c, n in zip(classes.tolist(), counts.tolist()):
            if c == 255:
                continue
            if num_classes is not None and (c < 0 or c >= num_classes):
                continue
            if c < 0:
                continue
            file_stats[int(c)] = int(n)
            samples_with_class[int(c)].append((fname, int(n)))
        sample_class_stats.append(file_stats)

    return sample_class_stats, samples_with_class


def main():
    args = parse_args()
    ann_dir = args.ann_dir
    if not osp.isabs(ann_dir):
        ann_dir = osp.join(args.data_root, ann_dir)

    sample_class_stats, samples_with_class = collect_stats(
        ann_dir=ann_dir,
        suffix=args.suffix,
        reduce_zero_label=args.reduce_zero_label,
        num_classes=args.num_classes)

    out_stats = osp.join(args.data_root, 'sample_class_stats.json')
    out_samples = osp.join(args.data_root, 'samples_with_class.json')

    with open(out_stats, 'w', encoding='utf-8') as f:
        json.dump(sample_class_stats, f, indent=2)

    with open(out_samples, 'w', encoding='utf-8') as f:
        json.dump(samples_with_class, f, indent=2)

    print(f'Processed: {len(sample_class_stats)} files')
    print(f'Wrote: {out_stats}')
    print(f'Wrote: {out_samples}')


if __name__ == '__main__':
    main()
