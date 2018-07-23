import tensorflow as tf
import numpy as np

print(tf.__version__)

RPN_POST_NMS_TOP_N = 2000

graph = tf.Graph()
with graph.as_default():
    tf_input_feature = tf.placeholder(dtype=tf.float32, shape=[None, 1024], name='input_feature')#tf.random_uniform([2000,1204], dtype=tf.float32)
    tf_input_position = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='input_position')
    # sum = tf.foldl(lambda a, e: a + e, tf_input_position, initializer=0.0)

    def grouped_convolution2D(inputs, filters, padding, num_groups,
                              strides=None,
                              dilation_rate=None):
        """
        Performs a grouped convolution by applying a normal convolution to each of the seperate groups
        :param inputs:
            Input of the shape [<batch_size>,H,W,inC]
        :param filters:
            [H,W,inC/num_groups,outC]
        :param padding:
            What padding to use
        :param num_groups:
            Number of seperate groups
        :param strides:
            Stride
        :param dilation_rate:
            Dilation rate
        :return:
            Output of shape [<batch_size>,H/stride,W/stride,outC]
        """
        # Split input and outputs along their last dimension
        input_list = tf.split(inputs, num_groups, axis=-1)
        filter_list = tf.split(filters, num_groups, axis=-1)

        output_list = []

        # Perform a normal convolution on each split of the input and filters
        for conv_idx, (input_tensor, filter_tensor) in enumerate(zip(input_list, filter_list)):
            output_list.append(tf.nn.convolution(
                input_tensor,
                filter_tensor,
                padding,
                strides=strides,
                dilation_rate=dilation_rate,
                name="grouped_convolution" + "_{}".format(conv_idx)
            ))
        # Concatenate ouptputs along their last dimentsion
        outputs = tf.concat(output_list, axis=-1)

        return outputs

    def extract_position_matrix(bbox, nongt_dim):
        pos_list = tf.unstack(tf_input_position, axis=1)
        y1_tensor = pos_list[0]
        x1_tensor = pos_list[1]
        y2_tensor = pos_list[2]
        x2_tensor = pos_list[3]

        bbox_width = x2_tensor - x1_tensor
        bbox_height = y2_tensor - y1_tensor

        center_x = 0.5 * (x1_tensor + x2_tensor)
        center_y = 0.5 * (y1_tensor + y2_tensor)

        delta_x = tf.reshape(center_x, [-1, 1])
        delta_x = tf.subtract(delta_x, tf.transpose(delta_x))
        delta_x = tf.divide(delta_x, bbox_width)
        delta_x = tf.log(tf.maximum(delta_x, 1e-3))

        delta_y = tf.reshape(center_y, [-1, 1])
        delta_y = tf.subtract(delta_y, tf.transpose(delta_y))
        delta_y = tf.divide(delta_y, bbox_height)
        delta_y = tf.log(tf.maximum(delta_y, 1e-3))

        delta_width = tf.reshape(bbox_width, [-1, 1])
        delta_width = tf.divide(delta_width, tf.transpose(delta_width))
        delta_width = tf.log(tf.maximum(delta_width, 1e-4))

        delta_height = tf.reshape(bbox_height, [-1, 1])
        delta_height = tf.divide(delta_height, tf.transpose(delta_height))
        delta_height = tf.log(tf.maximum(delta_height, 1e-4))

        delta_x = tf.slice(delta_x, [0,0], [-1, nongt_dim])
        delta_y = tf.slice(delta_y, [0, 0], [-1, nongt_dim])
        delta_width = tf.slice(delta_width, [0, 0], [-1, nongt_dim])
        delta_height = tf.slice(delta_height, [0, 0], [-1, nongt_dim])
        # print(delta_x.get_shape())
        position_matrix = tf.concat([tf.expand_dims(delta_x, axis=-1),
                                     tf.expand_dims(delta_y, axis=-1),
                                     tf.expand_dims(delta_width, axis=-1),
                                     tf.expand_dims(delta_height, axis=-1)], axis=-1)

        # print('test:',position_matrix.get_shape())

        return position_matrix


    def extract_position_embedding(position_mat, feat_dim=64, wave_length=1000):
        feat_range = tf.cast(tf.range(0, feat_dim/8),tf.float32)
        dim_mat = tf.pow(tf.cast(tf.fill([1,],wave_length),tf.float32), (8./feat_dim)*feat_range)
        dim_mat = tf.reshape(dim_mat, [1,1,1,-1])
        position_mat = tf.expand_dims(position_mat * 100.0, axis=3)
        div_mat = tf.divide(position_mat, dim_mat)
        sin_mat = tf.sin(div_mat)
        cos_mat = tf.cos(div_mat)
        embedding = tf.concat([sin_mat, cos_mat], axis=-1)

        embedding = tf.reshape(embedding,[tf.shape(embedding)[0], tf.shape(embedding)[1], feat_dim])

        return embedding

    def attention_module_multi_head(roi_feat, position_embedding, nongt_dim, fc_dim, feat_dim, dim=(1024,1024,1024), group=16, index=1):
        dim_group = (int(dim[0] / group), int(dim[1] / group), int(dim[2] / group))
        nongt_roi_feat = tf.slice(roi_feat, [0,0], [nongt_dim, feat_dim])
        position_embedding_reshape = tf.reshape(position_embedding, [-1, position_embedding.get_shape()[2]])
        position_feat_1 = tf.contrib.layers.fully_connected(position_embedding_reshape, fc_dim, activation_fn=tf.nn.relu)
        aff_weight = tf.reshape(position_feat_1, [-1, nongt_dim, fc_dim])
        aff_weight = tf.transpose(aff_weight, [0, 2, 1])

        assert dim[0] == dim[1], 'Matrix multiply requiers same dimensions!'

        q_data = tf.contrib.layers.fully_connected(roi_feat, dim[0])
        q_data_batch = tf.reshape(q_data, [-1, group, dim_group[0]])
        q_data_batch = tf.transpose(q_data_batch, [1, 0, 2])

        k_data = tf.contrib.layers.fully_connected(nongt_roi_feat, dim[1])
        k_data_batch = tf.reshape(k_data, [-1, group, dim_group[1]])
        k_data_batch = tf.transpose(k_data_batch, [1, 0, 2])

        v_data = nongt_roi_feat

        aff = tf.matmul(q_data_batch, k_data_batch, transpose_a=False, transpose_b=True)
        aff_scale = (1.0 / tf.sqrt(float(dim_group[1]))) * aff
        aff_scale = tf.transpose(aff_scale, [1, 0, 2])

        assert fc_dim == group, 'fc_dim != group'

        weighted_aff = tf.log(tf.maximum(aff_weight, 1e-6)) + aff_scale
        aff_softmax = tf.nn.softmax(weighted_aff, axis=2)
        aff_softmax_reshape = tf.reshape(aff_softmax, [-1, tf.shape(aff_softmax)[2]])

        output_t = tf.matmul(aff_softmax_reshape, v_data)
        output_t = tf.reshape(output_t, [-1, 1, 1, fc_dim * feat_dim])

        weights = tf.get_variable(name='group_conv2d_weight', shape=[1,1,feat_dim,1024], initializer=tf.contrib.layers.xavier_initializer())

        linear_output = grouped_convolution2D(output_t, weights, 'SAME', 16)

        return linear_output

    pos_matrix = extract_position_matrix(tf_input_position, RPN_POST_NMS_TOP_N)
    pos_embedding = extract_position_embedding(pos_matrix, 64)
    tf_op = attention_module_multi_head(tf_input_feature, pos_embedding, RPN_POST_NMS_TOP_N, fc_dim=16, feat_dim=1024)

    init = tf.global_variables_initializer()


np.random.seed(0)
np_input_feature = np.random.uniform(-1.0, 1.0, (4000, 1024))
np_input_position = np.random.uniform(0.0, 1.0, (4000, 4))
# print(np_input_position.shape)
with tf.Session(graph=graph) as sess:
    sess.run(init)

    np_test_input = sess.run([tf_op], feed_dict={tf_input_feature:np_input_feature, tf_input_position:np_input_position})[0]
    print(np_test_input)
    print(np_test_input.shape)
