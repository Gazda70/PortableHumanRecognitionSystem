from gluoncv import data, utils
from gluoncv.utils import viz
from matplotlib import pyplot as plt
from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
import mxnet as mx
from mxnet import autograd
from gluoncv.loss import SSDMultiBoxLoss
from mxnet import gluon



def train(train_loader, trainer, net, mbox_loss):
    for ib, batch in enumerate(train_loader):
        print('data:', batch[0].shape)
        print('class targets:', batch[1].shape)
        print('box targets:', batch[2].shape)
        with autograd.record():
            cls_pred, box_pred, anchors = net(batch[0])
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_pred, box_pred, batch[1], batch[2])
            # some standard gluon training steps:
            autograd.backward(sum_loss)
        trainer.step(1)