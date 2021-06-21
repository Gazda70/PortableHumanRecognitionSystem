from gluoncv import data, utils
from gluoncv.utils import viz
from matplotlib import pyplot as plt
from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd, gpu
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
import mxnet as mx
from mxnet import autograd
from gluoncv.loss import SSDMultiBoxLoss
from mxnet import gluon
from ssd_model_gluoncv_train import train

PATH_TO_VOC = "E:\PortableHumanRecognitionSystem\PortableHumanRecognitionSystem\datasets\VOC\VOCdevkit"

train_dataset = data.VOCDetection(PATH_TO_VOC, splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(PATH_TO_VOC, splits=[(2007, 'test')])
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))

train_image, train_label = train_dataset[0]
bboxes = train_label[:, :4]
cids = train_label[:, 4:5]
print('image:', train_image.shape)
print('bboxes:', bboxes.shape, 'class ids:', cids.shape)


width, height = 512, 512  # suppose we use 512 as base training size
train_transform = presets.ssd.SSDDefaultTrainTransform(width, height)
val_transform = presets.ssd.SSDDefaultValTransform(width, height)

utils.random.seed(233)  # fix seed in this tutorial

train_image2, train_label2 = train_transform(train_image, train_label)
print('tensor shape:', train_image2.shape)

batch_size = 32  # for tutorial, we use smaller batch-size
# you can make it larger(if your CPU has more cores) to accelerate data loading
num_workers = 0

#if __name__ == "__main__":
# behavior of batchify_fn: stack images, and pad labels
batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
train_loader = DataLoader(
    train_dataset.transform(train_transform),
    batch_size,
    shuffle=True,
    batchify_fn=batchify_fn,
    last_batch='rollover',
    num_workers=num_workers)
val_loader = DataLoader(
    val_dataset.transform(val_transform),
    batch_size,
    shuffle=False,
    batchify_fn=batchify_fn,
    last_batch='keep',
    num_workers=num_workers)

net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False)
#x = mx.nd.zeros(shape=(1, 3, 512, 512), ctx=gpu())
#net.initialize(ctx=gpu())
x = mx.nd.zeros(shape=(1, 3, 512, 512))
net.initialize()

with autograd.train_mode():
    cls_preds, box_preds, anchors = net(x)

train_transform = presets.ssd.SSDDefaultTrainTransform(width, height, anchors)
batchify_fn = Tuple(Stack(), Stack(), Stack())
train_loader = DataLoader(
    train_dataset.transform(train_transform),
    batch_size,
    shuffle=True,
    batchify_fn=batchify_fn,
    last_batch='rollover',
    num_workers=num_workers)

""""
cids, scores, bboxes = net(x)

SSD returns three values, where cids are the class labels, 
scores are confidence scores of each prediction, 
and bboxes are absolute coordinates of corresponding bounding boxes.

with autograd.train_mode():
    cls_preds, box_preds, anchors = net(x)
    
In training mode, SSD returns three intermediate values, w
here cls_preds are the class predictions prior to softmax, 
box_preds are bounding box offsets with one-to-one correspondence to anchors 
and anchors are absolute coordinates of corresponding anchors boxes, 
which are fixed since training images use inputs of same dimensions.
"""""
total_number_of_batches = 0
for ib, batch in enumerate(train_loader):
    total_number_of_batches += 1

print("Total number of batches: " + str(total_number_of_batches))

mbox_loss = SSDMultiBoxLoss()
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})

epochs = 2
for epoch in range(0, epochs):

    cumulative_sum_train_loss = []
    cumulative_cls_train_loss = []
    cumulative_box_train_loss = []
    training_samples = 0

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
        trainer.step(batch_size)
        #cumulative_sum_train_loss += sum_loss
        #cumulative_cls_train_loss += cls_loss
        #cumulative_box_train_loss += box_loss
        #training_samples += batch[0].shape
        print("Percent of actual epoch completed: " + str(ib*100/total_number_of_batches) + "\n")
        print("Epoch number: " + str(epoch))
    #print("Cumulative sum train loss: " + str(cumulative_sum_train_loss))
    #print("Cumulative class train loss: " + str(cumulative_cls_train_loss))
    #print("Cumulative box train loss: " + str(cumulative_box_train_loss))
    net.export('my_ssd_300_vgg16_atrous_voc' + str(epoch))

net.export('my_ssd_300_vgg16_atrous_voc_last')