"""Runs models on the MNIST dataset to ensure that they compile and train correctly."""

from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

from onconet.train import train
from onconet.datasets import factory as dataset_factory
from onconet.models import factory as model_factory
from onconet.transformers import factory as transformer_factory
from onconet.utils import parsing

MNIST_DATASETS = ['mnist', 'mnist_binary']
NOT_MNIST_EXCEPTION = 'Running MNIST test with dataset: {}. Dataset must be in {}'

def get_mnist_parsed_args():
    args = parsing.parse_args()

    if not args.dataset in MNIST_DATASETS:
        raise Exception(NOT_MNIST_EXCEPTION.format(args.dataset,
                                                   MNIST_DATASETS))

    args.image_transformers = parsing.parse_transformers(['scale_2d', 'grayscale'])
    args.tensor_transformers = parsing.parse_transformers(['normalize_2d'])
    args.test_image_transformers = parsing.parse_transformers(['scale_2d', 'grayscale'])
    args.test_tensor_transformers = parsing.parse_transformers(['normalize_2d'])

    args.epochs = 1
    args.max_batches_per_epoch = 100
    args.num_classes = 10
    args.wrap_model = True
    args.num_images = 2

    return args

if __name__ == '__main__':
    # Get arguments
    args = get_mnist_parsed_args()

    # Data transformers
    transformers = transformer_factory.get_transformers(
        args.image_transformers, args.tensor_transformers, args)
    test_transformers = transformer_factory.get_transformers(
        args.test_image_transformers, args.test_tensor_transformers, args)

    # Data loaders
    train_data, dev_data, test_data = dataset_factory.get_dataset(args, transformers, test_transformers)

    # Model
    model = model_factory.get_model(args)

    print(model)
    if args.cuda:
        model.cuda()

    # Train
    print('Training')
    train.train_model(train_data, dev_data, model, args)

    # Evaluate
    print('Evaluating')
    if args.threshold is None:
        print (' Computing probs threshold based on Dev set... (Provide a threshold as arg to prevent this computation)')
        args.dev_stats = train.compute_threshold_and_dev_stats(dev_data, model, args)

    train.eval_model(test_data, model, args)
