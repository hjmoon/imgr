# -*- coding: utf-8 -*-
"""effe2e.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ymqCX-0bZgzOoDYCHpaFd3NgRucY6v9u
"""
if 0:
    from google.colab import drive
    drive.mount('/content/gdrive')

import os
# import sys
# sys.path.insert(0, '/content/gdrive/My Drive/project/craft')
import time
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from models import efficientnet, craft, smallnet, transformer
import crafte2e_dataset, craft_util, losses

ROOT_PATH = '.'#'/content/gdrive/My Drive/project/craft
vocab = {}
rev_vocab = {}
with open(os.path.join(ROOT_PATH, 'vocab.txt'), 'r', encoding='UTF8') as f:
    lines = f.readlines()
    for line_ind, line in enumerate(lines):
        line = line.rstrip()
        if len(line) == 0:
            continue
        ch = line.split('\t')[0]
        vocab[ch] = line_ind
        rev_vocab[line_ind] = ch

"""INIT PARAMS"""

BATCH_SIZE = 1
num_epochs = 40
init_lr = 6e-5
input_shape = (1440, 1440, 3)
input_shape1 = (64, 90, 82)
max_boxes = 150
max_char = 30
vocab_size = len(vocab)
if 0:#
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)

"""BUILD DATASET"""

dataset = crafte2e_dataset.Dataset(os.path.join(ROOT_PATH, 'dataset/train/gts'), vocab, max_boxes, max_char)
def get_dataset(batch_size, is_training=True):
    tfdataset = tf.data.Dataset.from_generator(dataset.generate,
                                            (tf.int32),
                                            (tf.TensorShape([])))
    def tf_numpy_fn(index):
        tf_image, tf_char, tf_aff, tf_labels, tf_boxes, tf_quad_boxes = tf.numpy_function(dataset.process,
                                                        [index],
                                                        [tf.float32, tf.float32, tf.float32, tf.int32, tf.float32, tf.float32])
        tf_image.set_shape([1440,1440,3])
        tf_char.set_shape([720,720])
        tf_aff.set_shape([720,720])
        tf_labels.set_shape([max_boxes, max_char])
        tf_boxes.set_shape([max_boxes, 4])
        tf_quad_boxes.set_shape([max_boxes, 4,2])
        return tf_image, tf_char, tf_aff, tf_labels, tf_boxes, tf_quad_boxes
    tfdataset = tfdataset.map(tf_numpy_fn)#, num_parallel_calls=16)
    if is_training:
        tfdataset = tfdataset.repeat()
    tfdataset = tfdataset.batch(batch_size, drop_remainder=True)
    return tfdataset
if 0:
    per_replica_batch_size = BATCH_SIZE // strategy.num_replicas_in_sync
    train_dataset = strategy.experimental_distribute_datasets_from_function(lambda _: get_dataset(per_replica_batch_size, is_training=True))
else:
    train_dataset = get_dataset(BATCH_SIZE, is_training=True)

"""BUILD OPTIMIZER & MODEL"""

# if 1:#with strategy.scope():
optimizer = tf.keras.optimizers.Adam(init_lr)
backbone_model = efficientnet.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
craft_model = craft.CRAFT()
for layer in backbone_model.layers:
    layer.trainable=False
for layer in craft_model.layers:
    layer.trainable=False
resnet = smallnet.Resnet(trainable=True)
trfmer = transformer.Transformer(target_vocab_size=vocab_size)
focal_loss = losses.focal()

"""BUILD CKPT"""

ckpt = tf.train.Checkpoint(backbone=backbone_model, craft=craft_model, smallresnet=resnet, transformer=trfmer)
manager = tf.train.CheckpointManager(ckpt, os.path.join(ROOT_PATH, 'summaries/crafte2e5'), max_to_keep=3)

"""BUILD TRAIN FUNC"""

# def train_step(iterator):
@tf.function
def det(inputs):
    image = inputs        
    backbone_features = backbone_model(image)
    y, features = craft_model(backbone_features)
    return y, features

@tf.function
def extract_features1(inputs):
    image, features, boxes, char_mask, aff_mask, quad_boxes = inputs
    features = tf.concat([features, 
                          tf.expand_dims(char_mask, axis=3), 
                          tf.expand_dims(aff_mask, axis=3)], axis=-1)
    def batch_crop_fn(_inputs):
        _features, _crop_boxes, _q_boxes = _inputs
        
        def warp_features_fn(__inputs):
            _crop_box, _q_box = __inputs
            _crop_box = tf.cast(_crop_box*_features.shape[0], tf.int32)
            y1 = _crop_box[0]
            x1 = _crop_box[1]
            y2 = _crop_box[2]
            x2 = _crop_box[3]
            crop_feature = _features[y1:y2, x1:x2]
            _q_box = _q_box *_features.shape[0]
            
            def my_numpy_func(np_x1, np_y1, np_q_box, np_crop_feature):
#                     np_q_box, np_crop_feature = ___inputs
#                     
                if np.sum(np_q_box) == 0 or np_crop_feature.shape[0] == 0 or np_crop_feature.shape[1] == 0:
                    return np.zeros((input_shape1[0], input_shape1[1], np_crop_feature.shape[2]), np.float32)
                else:
                    np_q_box[:,0] -= np_x1
                    np_q_box[:,1] -= np_y1
                    pts1 = np.array([np_q_box[0],
                                        np_q_box[1],
                                        np_q_box[3],
                                        np_q_box[2]], dtype="float32")
                    pts2 = np.array([[0, 0],
                                    [input_shape1[1] - 1, 0],
                                    [0, input_shape1[0] - 1],
                                    [input_shape1[1] - 1, input_shape1[0] - 1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    return cv2.warpPerspective(np_crop_feature, M, (input_shape1[1], input_shape1[0]))
            
            __warp_features = tf.numpy_function(my_numpy_func, [x1, y1, _q_box, crop_feature], tf.float32)
            __warp_features.set_shape([input_shape1[0], input_shape1[1], _features.shape[2]])
            return __warp_features
        
        _warp_features = tf.map_fn(warp_features_fn, [_crop_boxes, _q_boxes], fn_output_signature=tf.float32, parallel_iterations=max_boxes)
        return _warp_features
    img_feature = tf.map_fn(batch_crop_fn, [features, boxes, quad_boxes], fn_output_signature=tf.float32, back_prop=False)
    crop_images = tf.zeros([BATCH_SIZE*max_boxes, input_shape1[0], input_shape1[1], 3], tf.float32)#tf.map_fn(batch_crop_fn, [image, boxes, quad_boxes], fn_output_signature=tf.float32, back_prop=False)
    img_feature = tf.reshape(img_feature, [BATCH_SIZE*max_boxes, input_shape1[0], input_shape1[1], input_shape1[2]])
    # crop_images = tf.reshape(crop_images, [BATCH_SIZE*max_boxes, input_shape1[0], input_shape1[1], 3])
    return img_feature, crop_images

def extract_features(inputs):
    image, features, boxes, char_mask, aff_mask, quad_boxes = inputs
    if 0:
        boxes = np.zeros((BATCH_SIZE, max_boxes, 4), np.float32)
        quad_boxes = np.zeros((BATCH_SIZE, max_boxes, 4, 2), np.float32)
        box_indices = np.zeros((BATCH_SIZE, max_boxes), np.int32)
        for batch_ind, (text_map, aff_map) in enumerate(zip(char_mask.numpy(), aff_mask.numpy())):
            det_boxes = craft_util.getDetBoxes(text_map,
                                                aff_map,
                                                text_threshold=0.2,
                                                link_threshold=0.2,
                                                low_text=0.2)
            det_boxes = np.array(det_boxes, np.float32) / (input_shape[0]//2)
            num_det_boxes = len(det_boxes)
            boxes[batch_ind, :num_det_boxes, 0] = det_boxes[:, 0, 1]
            boxes[batch_ind, :num_det_boxes, 1] = det_boxes[:, 0, 0]
            boxes[batch_ind, :num_det_boxes, 2] = det_boxes[:, 2, 1]
            boxes[batch_ind, :num_det_boxes, 3] = det_boxes[:, 2, 0]
            quad_boxes[batch_ind, :num_det_boxes, :, 0] = det_boxes[:,:,1]
            quad_boxes[batch_ind, :num_det_boxes, :, 1] = det_boxes[:,:,0]
            box_indices[batch_ind, :] = batch_ind
        boxes = tf.reshape(tf.convert_to_tensor(boxes, tf.float32), [BATCH_SIZE*max_boxes, 4])
        box_indices = tf.reshape(tf.convert_to_tensor(box_indices, tf.int32), [BATCH_SIZE*max_boxes])
    elif 1:
        np_imgs = image.numpy()
        np_features = tf.concat([features, 
                                 tf.expand_dims(char_mask, axis=3), 
                                 tf.expand_dims(aff_mask, axis=3)], axis=-1).numpy()
        features_shape = np_features.shape
        np_quad_boxes = quad_boxes.numpy()
        np_boxes = boxes.numpy()
        tmp_features = np.zeros([BATCH_SIZE, max_boxes, input_shape1[0], input_shape1[1], input_shape1[2]], np.float32)
        tmp_imgs = np.zeros([BATCH_SIZE, max_boxes, input_shape1[0], input_shape1[1], 3], np.float32)
        for batch_i in range(BATCH_SIZE):
            batch_feature = np_features[batch_i]
            for iter_i in range(max_boxes):
                crop_box = np_boxes[batch_i, iter_i]*features_shape[1]
                crop_box = crop_box.astype(np.int32)
                y1, x1, y2, x2 = crop_box
                crop_feature = batch_feature[y1:y2, x1:x2]
                q_boxes = np_quad_boxes[batch_i, iter_i]*features_shape[1]
                if np.sum(q_boxes) == 0 or crop_feature.shape[0] == 0 or crop_feature.shape[1] == 0:
                    continue
                q_boxes[:,0] -= x1
                q_boxes[:,1] -= y1
                pts1 = np.array([q_boxes[0],
                                 q_boxes[1],
                                 q_boxes[3],
                                 q_boxes[2]], dtype="float32")
                pts2 = np.array([[0, 0],
                                [input_shape1[1] - 1, 0],
                                [0, input_shape1[0] - 1],
                                [input_shape1[1] - 1, input_shape1[0] - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(pts1, pts2)
                tmp_features[batch_i, iter_i] = cv2.warpPerspective(crop_feature, M, (input_shape1[1], input_shape1[0]))
            # for iter_i in range(max_boxes):
            #     crop_box = np_boxes[batch_i, iter_i]*np_imgs.shape[1]
            #     crop_box = crop_box.astype(np.int32)
            #     y1, x1, y2, x2 = crop_box
            #     crop_feature = np_imgs[y1:y2, x1:x2]
            #     q_boxes = np_quad_boxes[batch_i, iter_i]*np_imgs.shape[1]
            #     if np.sum(q_boxes) == 0 or crop_feature.shape[0] == 0 or crop_feature.shape[1] == 0:
            #         continue
            #     q_boxes[0::2] -= x1
            #     q_boxes[1::2] -= y1
            #     pts1 = np.array([q_boxes[0],
            #                      q_boxes[1],
            #                      q_boxes[3],
            #                      q_boxes[2]], dtype="float32")
            #     pts2 = np.array([[0, 0],
            #                     [input_shape1[1] - 1, 0],
            #                     [0, input_shape1[0] - 1],
            #                     [input_shape1[1] - 1, input_shape1[0] - 1]], dtype="float32")
            #     M = cv2.getPerspectiveTransform(pts1, pts2)
            #     tmp_imgs[batch_i, iter_i] = cv2.warpPerspective(crop_feature, M, (input_shape1[1], input_shape1[0]))
            
        img_feature = tf.convert_to_tensor(tmp_features.reshape(BATCH_SIZE*max_boxes, input_shape1[0], input_shape1[1], input_shape1[2]), tf.float32)
        crop_images = tf.convert_to_tensor(tmp_imgs.reshape(BATCH_SIZE*max_boxes, input_shape1[0], input_shape1[1], 3), tf.float32)
    else:
        box_indices = np.zeros((BATCH_SIZE, max_boxes), np.int32)
        for batch_ind in range(BATCH_SIZE):
            box_indices[batch_ind, :] = batch_ind
        boxes = tf.reshape(boxes, [BATCH_SIZE*max_boxes, 4])
        box_indices = tf.reshape(tf.convert_to_tensor(box_indices, tf.int32), [BATCH_SIZE*max_boxes])
        img_feature = tf.image.crop_and_resize(features,
                                                boxes=boxes,
                                                box_indices=box_indices,
                                                crop_size=[input_shape1[0], input_shape1[1]])
        mask = tf.expand_dims(char_mask + aff_mask, axis=3)
        mask = tf.cast(tf.greater(mask, 0.05), tf.float32)
        crop_mask = tf.image.crop_and_resize(mask,
                                                boxes=boxes,
                                                box_indices=box_indices,
                                                crop_size=[input_shape1[0], input_shape1[1]])
        img_feature = img_feature * crop_mask
        crop_images = tf.image.crop_and_resize(image,
                                                boxes=boxes,
                                                box_indices=box_indices,
                                                crop_size=[input_shape1[0], input_shape1[1]])
    return img_feature, crop_images

@tf.function
def step_fn(inputs):
    img_features, labels = inputs
    with tf.GradientTape() as tape:
        img_features = resnet(img_features)
        img_features_shape = img_features.shape
        img_features = tf.transpose(img_features, [0, 2, 1, 3])
        img_features = tf.reshape(img_features, [img_features_shape[0], img_features_shape[1] * img_features_shape[2], img_features_shape[3]])
        img_features_mask = tf.ones([img_features_shape[0], img_features_shape[1] * img_features_shape[2]])
        labels = tf.reshape(labels, [img_features_shape[0], max_char])
        tar_input = labels[:, :-1]
        tar_real = labels[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = transformer.create_masks(img_features_mask, tar_input)
        outputs, attention_weights = trfmer(img_features, tar_input, True, enc_padding_mask, combined_mask, dec_padding_mask)
        if 0:
            outputs = tf.math.softmax(outputs)
            mask = tf.cast(tf.greater(tar_real, 0), tf.int32)
            tar_real_onehot = tf.one_hot(tar_real, depth=vocab_size)
            loss_value = focal_loss(y_true=tar_real_onehot,
                                    y_pred=outputs,
                                    y_mask=mask)
        else:
            loss_value = transformer.loss_function(tar_real, outputs)
        trainable_weights = resnet.trainable_weights + trfmer.trainable_weights#
    grads = tape.gradient(loss_value, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))
    return [loss_value, outputs, tar_real]
    # if 0:
    #     strategy.run(step_fn, args=(next(iterator),))

"""RESTORE"""
num_dataset = len(dataset.gt_path_ind)
epoch_st = 0
steps = 0
num_iter_per_epoch = num_dataset//BATCH_SIZE
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
    ckpt.restore(manager.latest_checkpoint)
    steps = int(manager.latest_checkpoint.split('-')[1])
    epoch_st = steps // num_iter_per_epoch
else:
    ckpt = tf.train.Checkpoint(backbone=backbone_model, craft=craft_model, smallresnet=resnet, transformer=trfmer)
    ckpt.restore(os.path.join(ROOT_PATH, 'summaries/crafte2e5/ckpt-105600'))
    print("Initializing from scratch.")

"""START TRAIN"""

from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

iterator = iter(train_dataset)

for epoch_i in  range(epoch_st, num_epochs):
    sum_time = 0
    for iter_i in range(num_iter_per_epoch):
        # loss = train_step(iterator)
        start_time = time.time()
        image, char, aff, labels, bboxes, quad_boxes = next(iterator)
        y, features = det(image)
        img_features, crop_images = extract_features([image, features, bboxes, char, aff, quad_boxes])
        loss, _preds, _labels = step_fn([img_features, labels])
        end_time = time.time()
        sum_time += (end_time - start_time)
        if steps % 100 == 0:            
            print("epoch: {:d},\tsteps: {:d},\tloss: {:1.2f},\tfn_time: {:1.2f}".format(epoch_i, steps, loss.numpy(), sum_time/(iter_i+1)))
            
            save_path = manager.save(steps)
            print("Saved checkpoint for step {}: {}".format(steps, save_path))
            if 0:
                np_image = image[0].numpy()
                np_image = (((np_image * dataset.std) + dataset.mean)*255.0).astype(np.uint8)
                plt.imshow(np_image)
                plt.show()
                crop_images = crop_images[0].numpy()
                crop_images = (((crop_images * dataset.std) + dataset.mean)*255.0).astype(np.uint8)
                plt.imshow(crop_images)
                plt.show()
                print(_labels.shape)
               
            gt = _labels[0].numpy()
            txt = []
            eos_ind = np.where(gt == 2)[0] 
            for ch_id in gt[:eos_ind[0]]:
                ch = rev_vocab[ch_id]
                txt.append(ch)
            print("\tgt:\t",''.join(txt)) 
            pred = np.argmax(_preds[0].numpy(), axis=-1)
            txt = []
            eos_ind = np.where(pred == 2)[0]
            if len(eos_ind) > 0:
                for ch_id in pred[:eos_ind[0]]:
                    ch = rev_vocab[ch_id]
                    txt.append(ch)            
                print("\tpred:\t",''.join(txt))
        steps += 1
    

