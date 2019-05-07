from os.path import dirname, realpath
import copy
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
from onconet.utils.parsing import parse_args, parse_transformers
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import onconet.datasets.factory as dataset_factory
from onconet.learn.utils import ignore_None_collate
import onconet.transformers.factory as transformer_factory
from onconet.utils.generic import normalize_dictionary
import json
import pdb

IMAGE_RIGHT_ALIGNED_PATH = "/home/administrator/Mounts/Isilon/metadata/image_path_to_right_aligned_aug22_2018.json"


def modify_args(args):
    # Set transformer to resize image img_size before computing stats
    # to improve computation speed and memory overhead
    args.num_chan = 3
    args.img_size = (1664, 2048)
    args.batch_size = 1
    args.num_workers = 80
    dim = '3d' if args.video else '2d'
    args.image_transformers = parse_transformers(['scale_{}'.format(dim)])
    args.tensor_transformers = parse_transformers(['force_num_chan_{}'.format(dim)])

def get_image_to_right_side(args):
    args = copy.deepcopy(args)
    modify_args(args)

    transformers = transformer_factory.get_transformers(args.image_transformers, args.tensor_transformers, args)

    train, dev, test = dataset_factory.get_dataset(args, transformers, transformers)

    image_to_side = {}
    for dataset in [train,dev,test]:
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=ignore_None_collate)


        for batch in tqdm(data_loader):
            img = batch['x']
            paths = batch['path']
            if args.cuda:
                img = img.cuda()

            B, C, H, W = img.size()

            left_half = img[:, :, :, :W//2].contiguous().view(B,-1)
            right_half = img[:, :, :, W//2:].contiguous().view(B,-1)

            is_right_aligned = right_half.sum(dim=-1) > left_half.sum(dim=-1)

            for indx, path in enumerate(paths):
                image_to_side[path] = bool(is_right_aligned[indx])

    return image_to_side

if __name__ == "__main__":

    args = parse_args()
    image_is_right_side = get_image_to_right_side(args)
    json.dump(image_is_right_side, open(IMAGE_RIGHT_ALIGNED_PATH,'w'))
