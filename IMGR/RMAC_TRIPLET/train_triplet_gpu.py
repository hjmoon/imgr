"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import os.path
import time
import sys
import tensorflow as tf
import numpy as np
import importlib
import itertools
import argparse
from nets import nets_factory
import cv2
import random
import pickle

from tensorflow.python.ops import data_flow_ops

from six.moves import xrange

#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

def _add_loss_summaries(total_loss):
    """Add summaries for losses.
  
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
  
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
  
    return loss_averages_op

def triplet_loss_fn(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
      
    return loss


def resize_to_min_dimension_np(image, max_dimension=800):
    image_height,image_width = image.shape[:2]
    
    max_image_dimension = np.maximum(image_height, image_width)
    #max_target_dimension = np.minimum(max_image_dimension, max_dimension)
    #print('max:',max_target_dimension)
    target_ratio = max_dimension / float(max_image_dimension)
#    target_height = int(float(image_height) * target_ratio)
#    target_width = int(float(image_width) * target_ratio)
    result = cv2.resize( image, None, fx=target_ratio, fy=target_ratio, interpolation=cv2.INTER_LINEAR)#,
    return result


#def resize_to_min_dimension(image, max_dimension=800):
#    image.set_shape([None, None, None])
#    image_height = tf.shape(image)[0]
#    image_width = tf.shape(image)[1]
#    max_image_dimension = tf.maximum(image_height, image_width)
#    max_target_dimension = tf.maximum(max_image_dimension, max_dimension)
#    target_ratio = tf.to_float(max_target_dimension) / tf.to_float(max_image_dimension)
#    target_height = tf.to_int32(tf.to_float(image_height) * target_ratio)
#    target_width = tf.to_int32(tf.to_float(image_width) * target_ratio)
#    print(image, target_height)
#    image = tf.image.resize_images( image, [target_height, target_width])
#    result = image#tf.squeeze(image, axis=0)
#    return result

def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.
  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.
  Args:
    tensor: A tensor of any type.
  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_shape[index])
  return combined_shape

def get_box_inds(proposals):
    proposals_shape = proposals.get_shape().as_list()
    if any(dim is None for dim in proposals_shape):
       proposals_shape = tf.shape(proposals)
    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
    multiplier = tf.expand_dims( tf.range(start=0, limit=proposals_shape[0]), 1 )
    return tf.reshape(ones_mat * multiplier, [-1])
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is not None:
        # Add 0 dimension to the gradients to represent the tower.
        #expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(g)
    var = grad_and_vars[0][1]
    if grads:
      if len(grads) > 1:
        sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
      else:
        sum_grad = grads[0]
      average_grads.append((sum_grad, var))
  return average_grads
def main(args):
  
    #network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))

    np.random.seed(seed=args.seed)
    #train_set = #facenet.get_dataset(args.data_dir)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
    
   
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        image_a = []
        image_p = []
        image_n = []
        bbox_a = []
        bbox_p = []
        bbox_n = []
        ROOT_DATA = '/data/scom/workspace/hjmoon/deepfashion/cropped_images/Category-Attribute-Prediction-Benchmark'
        print('start load train list')
        f = open(os.path.join('/data/scom/workspace/hjmoon/deepfashion','divided_category_v2.pkl'),'r')
        dataset_dict = pickle.load(f)
        f.close()
        print('end load trainlist')
        def _read_py_function(image_path, box_path, np_label):
            ret_images = np.zeros((3, args.image_size, args.image_size, 3), dtype=np.float32)
            ret_box = np.zeros((3, args.num_bbox, 4), dtype=np.float32)

            im_a = cv2.imread(image_path[0], cv2.IMREAD_COLOR)
            im_p = cv2.imread(image_path[1], cv2.IMREAD_COLOR)
            im_n = cv2.imread(image_path[2], cv2.IMREAD_COLOR)
           
            im_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2RGB)
            im_p = cv2.cvtColor(im_p, cv2.COLOR_BGR2RGB)
            im_n = cv2.cvtColor(im_n, cv2.COLOR_BGR2RGB)
            im_a = ((im_a * 0.00392156) -0.5 ) * 2.0
            im_p = ((im_p * 0.00392156) -0.5 ) * 2.0
            im_n = ((im_n * 0.00392156) -0.5 ) * 2.0
            im_a_height, im_a_width = im_a.shape[:2]
            im_p_height, im_p_width = im_p.shape[:2]
            im_n_height, im_n_width = im_n.shape[:2]
            im_size_list = [[im_a_height, im_a_width],[im_p_height, im_p_width],[im_n_height, im_n_width]]
            im_scale_list = [0.5,0.7,0.9]
            num_box_per_case = 11
            for im_size_idx, im_size in enumerate(im_size_list): 
                im_height,im_width = im_size
                ret_box[im_size_idx,0,:]=[0, 0, im_height-1, im_width-1]
                for im_scale_idx, im_scale in enumerate(im_scale_list):
                    tmp_im_scale_idx = im_scale_idx*num_box_per_case
                    box_height = im_height * im_scale 
                    box_width = im_width * im_scale
                    height_im_box_half_diff = (im_height-box_height-1)*0.5
                    width_im_box_half_diff = (im_width-box_width-1)*0.5

                    ret_box[im_size_idx,tmp_im_scale_idx+1,:]=[height_im_box_half_diff,width_im_box_half_diff,height_im_box_half_diff+box_height,(im_width-box_width)*0.5+box_width]
                    ret_box[im_size_idx,tmp_im_scale_idx+2,:]=[0, 0, box_height-1, box_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+3,:]=[im_height-box_height-1,im_width-box_width-1 , im_height-1, im_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+4,:]=[0,im_width-box_width-1 , box_height-1, im_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+5,:]=[im_height-box_height-1,0 , im_height-1, box_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+6,:]=[0, 0, im_height-1, box_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+7,:]=[0, width_im_box_half_diff, im_height-1, width_im_box_half_diff+box_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+8,:]=[0, im_width-box_width-1, im_height-1, im_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+9,:]=[0, 0, box_height-1, im_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+10,:]=[height_im_box_half_diff, 0, height_im_box_half_diff+box_height-1, im_width-1]
                    ret_box[im_size_idx,tmp_im_scale_idx+11,:]=[im_height-box_height-1, 0, im_height-1, im_width-1]

                ret_box[im_size_idx,:,1::2] = ret_box[im_size_idx,:,1::2] / float(im_width)
                ret_box[im_size_idx,:,::2] = ret_box[im_size_idx,:,::2] / float(im_height)
 
            im_a = cv2.resize(im_a,(args.image_size,args.image_size))#resize_to_min_dimension_np(im_a, max_dimension=args.image_size)        
            im_p = cv2.resize(im_p,(args.image_size,args.image_size))#resize_to_min_dimension_np(im_p, max_dimension=args.image_size)
            im_n = cv2.resize(im_n,(args.image_size,args.image_size))#resize_to_min_dimension_np(im_n, max_dimension=args.image_size)

            if (random.uniform(0,1)>0.5):
                im_a = cv2.flip(im_a, 1)
                ret_box[0,:,1::2] = 1.0 - ret_box[0,:,1::2]
            if (random.uniform(0,1)>0.5):
                im_p = cv2.flip(im_p, 1)
                ret_box[1,:,1::2] = 1.0 - ret_box[1,:,1::2]
            if (random.uniform(0,1)>0.5):
                im_n = cv2.flip(im_n, 1)
                ret_box[2,:,1::2] = 1.0 - ret_box[2,:,1::2]
            ret_images[0] = im_a
            ret_images[1] = im_p
            ret_images[2] = im_n
            return ret_images, ret_box, np_label

        print('init dataset')
        num_depth2=0
        for depth1 in dataset_dict:
            for depth2 in dataset_dict[depth1]:
                num_depth2+=1
        num_cache_list = args.num_sample_per_category * num_depth2
        remain = num_cache_list % (args.batch_size * args.num_clone)
        num_cache_list = num_cache_list -remain
        image_tf = tf.placeholder(tf.string, [num_cache_list,3])
        bbox_tf = tf.placeholder(tf.string, [num_cache_list,3])
        label_tf = tf.placeholder(tf.int32, [num_cache_list,1])
        dataset = tf.data.Dataset.from_tensor_slices((image_tf,bbox_tf,label_tf))
        dataset = dataset.map(lambda image_path,box_path,label: tuple(tf.py_func(_read_py_function, [image_path, box_path,label], [tf.float32, tf.float32, tf.int32])),num_parallel_calls=16 )

        dataset = dataset.batch(args.batch_size)#batch(1)
        iterator = dataset.make_initializable_iterator()#dataset.make_one_shot_iterator()
        nrof_batches = num_cache_list / args.batch_size

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step, args.learning_rate_decay_epochs*int(nrof_batches/args.num_clone), args.learning_rate_decay_factor, staircase=True)

        # Compute gradients.
        #with tf.control_dependencies([loss_averages_op]):
        if args.optimizer=='ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif args.optimizer=='ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif args.optimizer=='ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif args.optimizer=='RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif args.optimizer=='MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        elif args.optimizer=='SGD':
            opt = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            raise ValueError('Invalid optimization algorithm')
        network_fn = nets_factory.get_network_fn(args.model_def, is_training=phase_train_placeholder,  num_classes=args.num_classes, weight_decay=args.weight_decay)
        opt_grads = [] 
        loss_label = []
        triplet_losses = []
        for i in range(args.num_clone):
#        i = 0 
#        with tf.device('/gpu:%d'%(i+1)):
           with tf.device('/gpu:%d'%i):
                with tf.name_scope('%s_%d' % ("gpu", i)) as scope:
                  with tf.variable_scope(tf.get_variable_scope(), reuse=True if i > 0 else None):
                    image_batch, bboxes_batch, label_batch = iterator.get_next()  
                    image_batch = tf.reshape(image_batch, [-1,args.image_size,args.image_size,3])
                    bboxes_batch = tf.reshape(bboxes_batch, [-1,args.num_bbox,4])
                    image_batch = tf.identity(image_batch, 'image_batch')
                    bboxes_batch = tf.identity(bboxes_batch, 'bboxes_batch')
                    with tf.device('/cpu:0'):
                        summary_image_a, summary_image_p, summary_image_n = tf.unstack(tf.reshape(image_batch, [-1,3,args.image_size,args.image_size,3]),3,1)
                        tf.summary.image('image/anchor',summary_image_a, max_outputs=1)
                        tf.summary.image('image/positive',summary_image_p, max_outputs=1)
                        tf.summary.image('image/negative',summary_image_n, max_outputs=1)

                    # Build the inference graph
                    print('init network')
                    _, end_points = network_fn(image_batch, create_aux_logits=False, spatial_squeeze=False)
            
                    with tf.variable_scope('Triplet'): 
                        bbox_inds = get_box_inds(bboxes_batch)
                        combined_shape=combined_static_and_dynamic_shape(bboxes_batch)
                        flattened_shape = tf.stack([combined_shape[0] * combined_shape[1]] + combined_shape[2:])
                        crop_feature_conv4 = tf.image.crop_and_resize(end_points['Mixed_6e'],  tf.reshape(bboxes_batch, flattened_shape), bbox_inds, (7, 7))
                        crop_feature_conv5 = tf.image.crop_and_resize(end_points['Mixed_7c'],  tf.reshape(bboxes_batch, flattened_shape), bbox_inds, (7, 7))
                        crop_feature = tf.concat([crop_feature_conv4, crop_feature_conv5],axis=3)
                        proposals_embeddings = tf.reshape(tf.nn.max_pool(crop_feature,ksize=[1,7,7,1], strides=[1, 2, 2, 1], padding='VALID'),[-1,2816])#tf.reduce_sum(crop_feature, [1,2], name='gap')
                        embeddings = tf.nn.l2_normalize(proposals_embeddings, 1, 1e-10, name='l2norm1')#proposals_embeddings
                        shift = embeddings - tf.reduce_mean(embeddings, [1], keep_dims=True)
                        fc = tf.contrib.layers.fully_connected(shift, args.embedding_size, activation_fn=None, scope='fc')
                        fc_drop = tf.nn.dropout(fc, 0.2)
                        reshape_fc = tf.reshape(fc_drop, [-1, args.num_bbox, args.embedding_size])
                        embeddings2 = tf.nn.l2_normalize(reshape_fc, 2, 1e-10, name='l2norm2')
                        sum_agg = tf.reduce_sum(embeddings2, [1], name='sum_aggregate')
                        embeddings3 = tf.nn.l2_normalize(sum_agg, 1, 1e-10, name='l2norm3')
                        
                        # Split embeddings into anchor, positive and negative and calculate triplet loss
                    anchor, positive, negative = tf.unstack(tf.reshape(embeddings3, [-1,3,args.embedding_size]), 3, 1)
                    triplet_loss = triplet_loss_fn(anchor, positive, negative, args.alpha)
                    loss_label.append([embeddings3, label_batch])
                    # Calculate the total losses
                    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')
                    triplet_losses.append(total_loss)
                    # Generate moving averages of all losses and associated summaries.
                    grads = opt.compute_gradients(total_loss)
                    opt_grads.append(grads)

        grads = average_gradients(opt_grads)
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        tf.summary.scalar('learning_rate', learning_rate)

        total_triplet_loss = tf.add_n(triplet_losses, name='total_triplet_loss')
        tf.summary.scalar('total_loss(reg)', total_triplet_loss)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(  args.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.model_variables())
    
        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Add histograms for trainable variables.
        if 1:#log_histograms:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
       
        # Add histograms for gradients.
        if 1:#log_histograms:
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
      
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto())#gpu_options=gpu_options))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
#        coord = tf.train.Coordinator()
#        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                exclusions = []
                if args.checkpoint_exclude_scopes:
                  exclusions = [scope.strip()
                                for scope in args.checkpoint_exclude_scopes.split(',')]
 
                # TODO(sguada) variables.filter_variables()
                variables_to_restore = []
                for var in tf.trainable_variables():#global_variables():#slim.get_model_variables():
                  excluded = False
                  for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                      excluded = True
                      break
                  if not excluded:
                    variables_to_restore.append(var)
                restore_saver = tf.train.Saver(variables_to_restore)
                restore_saver.restore(sess, args.pretrained_model)
            # Training and validation loop
            epoch = 0
            print('start training')
            while epoch < args.max_nrof_epochs:
                i = 0
                train_time = 0
                step = sess.run(global_step, feed_dict=None)
                #############
                # Preselect #
                #############
                selected_triplet=[]
                while len(selected_triplet)<num_cache_list:
                    pre_selected_triplet = []
                    pre_selected_triplet_bbox = []
                    start_time = time.time()
                    while len(pre_selected_triplet)<num_cache_list:
                        depth1_keys = dataset_dict.keys()
                        for depth1_idx, depth1_cate in enumerate(depth1_keys):
                            depth2_keys = dataset_dict[depth1_cate].keys()
                            for depth2_idx, depth2_key in enumerate(depth2_keys):
                                data_list = dataset_dict[depth1_cate][depth2_key]
                                len_data_list = len(data_list)
                                inds = [v for v in range(len_data_list)]
                                random.shuffle(inds)
                                anchor_path = os.path.join(ROOT_DATA,data_list[inds.pop()])
                                anchor_box_path = 'test'
                                len_inds = len(inds)
                                num_sample = min(len_inds, args.num_sample_per_category)
                                for selected_idx in range(num_sample):
                                    pos_ind = selected_idx
                                    positive_path = os.path.join(ROOT_DATA,data_list[inds[pos_ind]])
                                    positive_box_path = 'test'

                                    if len(depth2_keys) > 1:
                                        neg_inds = [v for v in range(len(depth2_keys)) if v != depth2_idx]
                                        random.shuffle(neg_inds)
                                        neg_depth2 = dataset_dict[depth1_cate][depth2_keys[neg_inds[0]]]
                                        neg_inds = [v for v in range(len(neg_depth2))]
                                        random.shuffle(neg_inds)
                                        negative_path = os.path.join(ROOT_DATA,neg_depth2[neg_inds[0]])
                                        negative_box_path = 'test'
                                    else:
                                        depth1_key_inds = [v for v in range(len(depth1_keys)) if v != depth1_idx]
                                        random.shuffle(depth1_key_inds)
                                        tmp_depth2_keys = dataset_dict[depth1_keys[depth1_key_inds[0]]].keys()

                                        neg_inds = [v for v in range(len(tmp_depth2_keys))]
                                        random.shuffle(neg_inds)

                                        neg_depth2 = dataset_dict[depth1_keys[depth1_key_inds[0]]][tmp_depth2_keys[neg_inds[0]]]
                                        neg_inds = [v for v in range(len(neg_depth2))]
                                        random.shuffle(neg_inds)
                                        negative_path = os.path.join(ROOT_DATA,neg_depth2[neg_inds[0]])
                                        negative_box_path = 'test'
        
                                    pre_selected_triplet.append([anchor_path, positive_path, negative_path])
                                    pre_selected_triplet_bbox.append([anchor_box_path, positive_box_path, negative_box_path])
                    pre_selected_triplet=pre_selected_triplet[:num_cache_list]
                                       
                    np_image=np.array(pre_selected_triplet[:num_cache_list])
                    np_bbox=np.array(pre_selected_triplet_bbox[:num_cache_list])
                    np_label = np.array([v for v in range(np_image.shape[0])],dtype=np.int).reshape(-1,1)
                    sess.run(iterator.initializer, feed_dict={image_tf: np_image, bbox_tf: np_bbox, label_tf:np_label})
    
                    print('init',len(np_image))
                    print('selected triplet size:',len(selected_triplet))
                    j=0
                    
                    while j<num_cache_list/(args.batch_size*args.num_clone):
                        try:    
                            feed_dict = {learning_rate_placeholder: 0.0, phase_train_placeholder: True}
    #                        for loss_label_i in range( len(loss_label) ):
                            op_tf = list(itertools.chain.from_iterable(loss_label))
                            result = sess.run(op_tf, feed_dict=feed_dict)
                            j+=1
                            for result_idx in range(0,len(result),2):
                                embed, np_label = result[result_idx:result_idx+2]
                                embed= embed.reshape(-1,3,args.embedding_size)
                                pos_dist_sqr = np.sum(np.square(embed[:,0,:]-embed[:,1,:]),axis=1)
                                neg_dist_sqr = np.sum(np.square(embed[:,0,:]-embed[:,2,:]),axis=1)
                                diff = neg_dist_sqr - pos_dist_sqr
                                all_neg = np.where(np.logical_and(neg_dist_sqr-pos_dist_sqr<args.alpha, pos_dist_sqr<neg_dist_sqr))[0]# all_neg = np.where(diff<args.alpha)[0]
                                for neg_idx in all_neg:
                                    selected_triplet.append([pre_selected_triplet[np_label[neg_idx,0]][0],
                                                            pre_selected_triplet[np_label[neg_idx,0]][1],
                                                            pre_selected_triplet[np_label[neg_idx,0]][2],
                                                            pre_selected_triplet_bbox[np_label[neg_idx,0]][0],
                                                            pre_selected_triplet_bbox[np_label[neg_idx,0]][1],
                                                            pre_selected_triplet_bbox[np_label[neg_idx,0]][2]])
                                    if len(selected_triplet)>=num_cache_list:
                                        break
                                if len(selected_triplet)>=num_cache_list:
                                        break
                            if len(selected_triplet)>=num_cache_list:
                                        break

                        except tf.errors.OutOfRangeError:
                            break

                print('time:',(time.time() - start_time))
                print('selected triplet:',len(selected_triplet))
                i=0
                random.shuffle(selected_triplet)
                np_selected_triplet = np.array(selected_triplet[:num_cache_list])
                np_image = np_selected_triplet[:,:3]
                np_bbox = np_selected_triplet[:,3:]
                np_label = np.array([v for v in range(np_selected_triplet.shape[0])],dtype=np.int).reshape(-1,1)
                sess.run(iterator.initializer, feed_dict={image_tf: np_image, bbox_tf: np_bbox, label_tf:np_label})
                j=1
                while j<num_cache_list/(args.batch_size*args.num_clone):
                    try:    
                        start_time = time.time()
                        feed_dict = {learning_rate_placeholder: args.learning_rate, phase_train_placeholder: True}
                        err, _, step, _summary_op = sess.run([total_loss, train_op, global_step, summary_op ], feed_dict=feed_dict)
                        duration = time.time() - start_time
                        if j % 10 == 0:
                            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' % (epoch, j+1, num_cache_list/(args.batch_size*args.num_clone), duration, err))
                        if args.log_interval < train_time:
                            train_time = 0
                            # Save variables and the metagraph if it doesn't exist already
                        if train_time == 0:
                            summary_writer.add_summary(_summary_op, step)
                            save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
                            
                        train_time += duration
                        j+=1
                    except tf.errors.OutOfRangeError:
                        break
                     
                epoch += 1
                # Evaluate on LFW
#                if args.lfw_dir:
#                    evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
#                            batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, args.batch_size, 
#                            args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)

    return model_dir


def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step, 
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets*3/args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)),(-1,3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths),1), (-1,3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples-i*args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab,:] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number+1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            
        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step
  
def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder, 
        batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, batch_size, 
        nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')
    
    nrof_images = len(actual_issame)*2
    assert(len(image_paths)==nrof_images)
    labels_array = np.reshape(np.arange(nrof_images),(-1,3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths),1), (-1,3))
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images-i*batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
            learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab,:] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time()-start_time))
    
    assert(np.all(label_check_array==1))
    
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='inception_v3')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=9)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=800)
    parser.add_argument('--alpha', type=float,
        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=1024)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.00004)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM', 'SGD'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--log_interval', type=int,
        help='log interval.', default=1800)
    parser.add_argument('--num_bbox', type=int,
        help='box size.', default=34)
    parser.add_argument('--num_classes', type=int,
        help='num calsses.', default=300)
    parser.add_argument('--checkpoint_exclude_scopes', type=str,
        help='exclude scopes.', default=None)
    parser.add_argument('--num_clone', type=int,
        help='num clone.', default=1)
    parser.add_argument('--num_sample_per_category', type=int,
        help='num_smaple_per_category.', default=5)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
