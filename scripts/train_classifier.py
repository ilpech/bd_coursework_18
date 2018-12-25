#brief: скрипт обучения классификатора столбиков на предобработанных сниппетах 16х48
#author: pichugin
# usage: python3 train_classifier.py --train True --model-name resblocks_v5 --net-name pillars_detector.008_test
#       --params-dir /datasets/models --classes 5 --dataset-path /datasets/dataset_pvsp.010_splitted_cropped_autocontrasted
#       --ctx gpu --batch-size 16 --epochs 140 --save True --ssh False --note test_try
import argparse
import mxnet as mx
import numpy as np
import os, time

import pathlib

from mxnet import gluon, image, init
from mxnet.gluon import nn
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from mxnet import autograd as ag

from mxnet.gluon.model_zoo import vision as models

import sys
sys.path.append("..")

from models.resblocks import get_resblocks
from models.cifar_wide_resnet import cifar_wideresnet16_2

from tools import *


opt = get_train_argparse()

train = opt.train

model_name = opt.model_name
net_name = opt.net_name
classes = opt.classes

params_dir = opt.params_dir
dataset_path = opt.dataset_path

ctx_type = opt.ctx
device_index = opt.device_index
batch_size = opt.batch_size

epochs = opt.epochs
resume_epoch = opt.resume_epoch
optimizer = opt.optimizer
lr = opt.lr
lr_decay = opt.lr_decay
lr_decay_interval = opt.lr_decay_interval
momentum = opt.momentum
wd = opt.wd

save = opt.save
log_interval = opt.log_interval
ssh = opt.ssh
note = opt.note

num_gpus = 1
num_workers = 8

if ctx_type == 'cpu':
    ctx = [mx.cpu()]
elif ctx_type == 'gpu':
    ctx = [mx.gpu(device_index)]

# net = get_resblocks(classes)
net = cifar_wideresnet16_2(classes=classes)

net.collect_params().initialize(init.Xavier(magnitude=2.24), ctx = ctx)
net.collect_params().reset_ctx(ctx)
net.hybridize()

train_data, val_data, test_data = get_data_raw(dataset_path, batch_size, num_workers)
num_batch = len(train_data)

params_path = os.path.join(params_dir, net_name)

ensure_folder(params_path)

if train:
    log_file = os.path.join(params_path, net_name + '_' + model_name + '_logs.txt')
    log = open(log_file, 'w')
    if note != None:
        log.write(note + '\n' )
    log.write(dataset_path + '\n')

if resume_epoch > 0:
    net.load_parameters(os.path.join(params_path, '{}_{:03d}__{}.params'.format(
        net_name,resume_epoch,model_name))
                        )

optimizer_params = {'wd': wd, 'momentum': momentum, 'learning_rate': lr}
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

train_metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()


print("Batch size", batch_size)
print('Workon dataset_path: ', dataset_path)
print('Model Name: ', model_name)
print('Params saving in: ', params_path)
print('Start training loop')

best_val_acc = 0
save_best_val_acc = False
lr_decay_count = 0
if train:
    assert resume_epoch < epochs, ('Error in finetune resume_epoch < epochs')
    for epoch in range(resume_epoch, epochs):
        if epoch % lr_decay_interval == 0 and epoch != 0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        tic = time.time()
        train_loss = 0
        train_metric.reset()

        for i, batch in enumerate(train_data):
            # Extract data and label
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            # AutoGrad
            with ag.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            # Backpropagation
            for l in loss:
                l.backward()

            # Optimize
            trainer.step(batch_size)

            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
            train_metric.update(label, outputs)


        name, train_acc = train_metric.get()
        train_loss /= num_batch


        name, val_acc = test(net, val_data, ctx)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_best_val_acc = True
        else:
            save_best_val_acc = False

        name, val_pillars_acc = test_on_single_class(net, val_data, ctx, 2)
        name, val_person_acc = test_on_single_class(net, val_data, ctx, 1)
        name, val_trash_acc = test_on_single_class(net, val_data, ctx, 3)
        name, val_vertical_acc = test_on_single_class(net, val_data, ctx, 4)

        scores = ('[Epoch {:d}] Train-acc: {:.3f}, loss: {:.3f} | Val-acc: {:.3f}'
                  '| Pillars-acc: {:.3f} | Persons-acc: {:.3f} | Vertical-acc: {:.3f} | Trash-acc: {:.3f} | time: {:.1f}').format(
                epoch, train_acc, train_loss, val_acc, val_pillars_acc, val_person_acc, val_vertical_acc, val_trash_acc, time.time() - tic)

        log.write(scores + '\n' )
        print(scores)

        if save:
            if (epoch+1) % log_interval == 0 or save_best_val_acc:
                if save_best_val_acc:
                    print('Params saved on epoch {}, new best val acc founded'.format(epoch+1))
                else:
                    print('Params saved on epoch {}'.format(epoch+1))
                net.save_parameters(os.path.join(
                        params_path,
                        '{:s}_{:03d}__{}.params'.format(net_name, epoch+1, model_name))
                        )
    if not ssh:
        train_history.plot()

name, test_acc = test(net, test_data, ctx)
name, test_half_p_acc = test_on_single_class(net, test_data, ctx, 0)
name, test_pillars_acc = test_on_single_class(net, test_data, ctx, 2)
name, test_person_acc = test_on_single_class(net, test_data, ctx, 1)
name, test_trash_acc = test_on_single_class(net, test_data, ctx, 3)
name, test_vertical_acc = test_on_single_class(net, test_data, ctx, 4)
test_score = '[Finished] Test-acc: {:.3f}'.format(test_acc)
test_classes_score = ('Test-pillars-acc: {:.3f} '
                      '| Test-person-acc: {:.3f} | Test-trash-acc: {:.3f} '
                      '| Test-vertical-acc: {:.3f} | Test-half-cut-person-acc: {:.3f}').format(
                    test_pillars_acc, test_person_acc, test_trash_acc, test_vertical_acc, test_half_p_acc)
print(test_score, '\n', test_classes_score)
if train:
    log.write(test_score + '\n')
    log.write(test_classes_score + '\n')
    log.close()
