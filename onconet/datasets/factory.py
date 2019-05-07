import pickle

NO_DATASET_ERR = "Dataset {} not in DATASET_REGISTRY! Available datasets are {}"
DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
    """Registers a dataset."""

    def decorator(f):
        DATASET_REGISTRY[dataset_name] = f
        return f

    return decorator


def get_dataset_class(args):
    if args.dataset not in DATASET_REGISTRY:
        raise Exception(
            NO_DATASET_ERR.format(args.dataset, DATASET_REGISTRY.keys()))

    return DATASET_REGISTRY[args.dataset]

# Depending on arg, build dataset
def get_dataset(args, transformers, test_transformers):
    dataset_class = get_dataset_class(args)
    train = dataset_class(args, transformers, 'train')
    dev = dataset_class(args, test_transformers, 'dev')
    test = dataset_class(args, test_transformers, 'test')

    return train, dev, test
