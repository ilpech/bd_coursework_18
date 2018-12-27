import argparse
import mxnet as mx
import os, math, time, sys
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from mxnet import autograd as ag
import pathlib


class SingleClassAccuracy(mx.metric.EvalMetric):
    def __init__(self, name='single_class_accuracy',
                 output_names=None, label_names=None):
        super(SingleClassAccuracy, self).__init__(
            name,
            output_names=output_names, label_names=label_names)


    def update(self, label, preds):
        pred_label = nd.argmax(preds, axis=0)

        pred_label = pred_label.asnumpy().astype('int32')

        self.sum_metric += (pred_label.flat == label).sum()
        self.num_inst += len(pred_label.flat)


def get_data_raw(dataset_path, batch_size, num_workers):
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')
    # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    jitter_param = 0.4
    lighting_param = 0.1
    input_size = (16, 48)

    transform_train = transforms.Compose([
        # transforms.RandomResizedCrop(input_size,scale=(0.95, 1.0)),
        transforms.Resize(input_size, keep_ratio = True),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),

    ])

    transform_test = transforms.Compose([
        transforms.Resize(input_size, keep_ratio = True),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),

    ])

    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    test_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    return train_data, val_data, test_data


def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()


def test_on_single_class(net, val_data, ctx, class_ind):
    metric = SingleClassAccuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        for label_ind in range(len(label[0])):
            if label[0][label_ind] == class_ind:
                # print(label[0][label_ind].asnumpy()[0], outputs[0][label_ind])
                metric.update(label[0][label_ind].asnumpy()[0], outputs[0][label_ind])

    return metric.get()


def ensure_folder(dir_fname):
    if not os.path.exists(dir_fname):
        try:
            pathlib.Path(dir_fname).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            print('Unable to create {} directory. Permission denied'.format(dir_fname))


def get_train_argparse():
    parser = argparse.ArgumentParser(
                                    description=('Train a mxnet model'
                                                 'for image classification.')
                                    )
    parser.add_argument(
                        '--train', type=bool, default=True,
                        help='set train mode (or only test)'
                        )
    parser.add_argument(
                        '--model-name', type=str,
                        help='set model name'
                        )
    parser.add_argument(
                        '--net-name', type=str,
                        help='set network name'
                        )
    parser.add_argument(
                        '--params-dir', type=str,
                        help='set path to dir with params & struct'
                        )
    parser.add_argument(
                        '--resume-epoch', type=int, default=0,
                        help='set epoch to finetune from, def=0'
                        )
    parser.add_argument(
                        '--classes', type=int,
                        help='set output class number for model'
                        )

    parser.add_argument(
                        '--dataset-path', type=str,
                        help=('path to folder with sorted data:'
                              'into test, train, val with classes inside'
                              )
                        )
    parser.add_argument(
                        '--ctx', type=str, default='gpu',
                        help='device to use for training, def = gpu'
                        )
    parser.add_argument(
                        '--device-index', type=int, default=0,
                        help='device index to use for training, def = 0'
                        )
    parser.add_argument(
                        '--batch-size', type=int, default='16',
                        help='set batch size per device for training, def = 16'
                        )
    parser.add_argument(
                        '--epochs', type=int, default='240',
                        help='set number of epochs for training, def = 240'
                        )
    parser.add_argument(
                        '--optimizer', type=str, default='sgd',
                        help='set optimizer type, def = sgd'
                        )
    parser.add_argument(
                        '--lr', type=float, default='0.001',
                        help='set learning rate for optimizer, def=0.001'
                        )
    parser.add_argument(
                        '--lr-decay', type=float, default='0.1',
                        help='set learning rate for optimizer, def=0.1'
                        )
    parser.add_argument(
                        '--lr-decay-interval', type=int, default=20,
                        help='interval of epochs to decay lr'
                        )
    parser.add_argument(
                        '--momentum', type=float, default='0.9',
                        help='momentum value for optimizer, def=0.9'
                        )
    parser.add_argument(
                        '--wd', type=float, default='0.0001',
                        help='set lr, def=0.0001'
                        )
    parser.add_argument(
                        '--save', type=bool,
                        help='bool to save plots and params every log interval'
                        )
    parser.add_argument(
                        '--ssh', type=bool, default=False,
                        help='bool for remote access (no plots saving), def=False'
                        )
    parser.add_argument(
                        '--log-interval', type=int, default=10,
                        help='number of batches to wait before each logging and model saving'
                        )
    parser.add_argument(
                        '--note', type=str,
                        help='help note to add in top of log file if necessary'
                        )

    opt = parser.parse_args()

    return opt
