import numpy as np
from abc import ABCMeta, abstractmethod
import torch
from torch.utils import data
import os
import warnings
import json
import traceback
from collections import Counter
from onconet.datasets.loader.image import image_loader
from onconet.utils.risk_factors import parse_risk_factors, RiskFactorVectorizer
import pdb

METAFILE_NOTFOUND_ERR = "Metadata file {} could not be parsed! Exception: {}!"
LOAD_FAIL_MSG = "Failed to load image: {}\nException: {}"


class Abstract_Onco_Dataset(data.Dataset):
    """
    Abstract Object for all Onco Datasets. All datasets have some metadata
    property associated with them, a create_dataset method, a task, and a check
    label and get label function.
    """
    __metaclass__ = ABCMeta

    def __init__(self, args, transformers, split_group):
        '''
        params: args - config.
        params: transformer - A transformer object, takes in a PIL image, performs some transforms and returns a Tensor
        params: split_group - ['train'|'dev'|'test'].

        constructs: standard pytorch Dataset obj, which can be fed in a DataLoader for batching
        '''
        super(Abstract_Onco_Dataset, self).__init__()
        args.metadata_path = os.path.join(args.metadata_dir,
                                          self.METADATA_FILENAME)

        self.args = args
        self.image_loader = image_loader(args.cache_path,
                                                      transformers)
        try:
            self.metadata_json = json.load(open(args.metadata_path, 'r'))
        except Exception as e:
            raise Exception(METAFILE_NOTFOUND_ERR.format(args.metadata_path, e))

        self.dataset = self.create_dataset(split_group, args.img_dir)
        if split_group == 'train' and self.args.data_fraction < 1.0:
            self.dataset = np.random.choice(self.dataset, int(len(self.dataset)*self.args.data_fraction), replace=False)
        self.risk_factor_vectorizer = RiskFactorVectorizer(args)
        if self.args.use_risk_factors:
            self.add_risk_factors_to_dataset()

        if 'dist_key' in self.dataset[0] and args.year_weighted_class_bal:
            dist_key = 'dist_key'
        else:
            dist_key = 'y'

        label_dist = [d[dist_key] for d in self.dataset]
        label_counts = Counter(label_dist)
        weight_per_label = 1./ len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
            }
        if args.year_weighted_class_bal or args.class_bal:
            print("Label weights are {}".format(label_weights))
        self.weights = [ label_weights[d[dist_key]] for d in self.dataset]

    @property
    @abstractmethod
    def task(self):
        pass

    @property
    @abstractmethod
    def METADATA_FILENAME(self):
        pass

    @abstractmethod
    def check_label(self, row):
        '''
        Return True if the row contains a valid label for the task
        :row: - metadata row
        '''
        pass

    @abstractmethod
    def get_label(self, row):
        '''
        Get task specific label for a given metadata row
        :row: - metadata row with contains label information
        '''
        pass

    @abstractmethod
    def create_dataset(self, split_group, img_dir):
        """
        Creating the dataset from the paths and labels in the json.

        :split_group: - ['train'|'dev'|'test'].
        :img_dir: - The path to the dir containing the images.

        """
        pass


    @staticmethod
    def set_args(args):
        """Sets any args particular to the dataset."""
        pass

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        try:
            additional = {} if sample['additional'] is None else sample['additional']
            x = self.image_loader.get_image(sample['path'], additional)

            item = {
                'x': x,
                'path': sample['path'],
                'y': sample['y']
            }

            if 'exam' in sample:
                item['exam'] = sample['exam']
            if self.args.use_risk_factors:
                # Note, risk factors not supported for target objects
                item['risk_factors'] = sample['risk_factors']

            return item

        except Exception:
            warnings.warn(LOAD_FAIL_MSG.format(sample['path'], traceback.print_exc()))

    def add_risk_factors_to_dataset(self):
        for sample in self.dataset:
            sample['risk_factors'] = self.risk_factor_vectorizer.get_risk_factors_for_sample(sample)

    def image_paths_by_views(self, exam):
        '''
        Determine images of left and right CCs and MLO.
        Args:
        exam - a dictionary with views and files sorted relatively.

        returns:
        4 lists of image paths of each view by this order: left_ccs, left_mlos, right_ccs, right_mlos. Force max 1 image per view.

        Note: Validation of cancer side is performed in the query scripts/from_db/cancer.py in OncoQueries
        '''
        left_ccs = [image_path for view, image_path in zip(exam['views'], exam['files']) if view.startswith('L CC')][:1]
        left_mlos = [image_path for view, image_path in zip(exam['views'], exam['files']) if view.startswith('L MLO')][:1]
        right_ccs = [image_path for view, image_path in zip(exam['views'], exam['files']) if view.startswith('R CC')][:1]
        right_mlos = [image_path for view, image_path in zip(exam['views'], exam['files']) if view.startswith('R MLO')][:1]
        return left_ccs, left_mlos, right_ccs, right_mlos
