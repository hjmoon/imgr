import tensorflow as tf
from tensorflow import keras

def mse_ohem(labels, preds):
    def batch_fn(_inputs):
        _labels, _preds = _inputs
        mask_shape = _labels.shape
        _labels = tf.reshape(_labels, [mask_shape[0]*mask_shape[1]])
        _preds = tf.reshape(_preds, [mask_shape[0]*mask_shape[1]])
        
        _diff = (_labels - _preds)
        _loss = tf.math.sqrt(_diff * _diff)
        
        _pos_mask = tf.greater(_labels, 0)
        _pos_ind = tf.squeeze(tf.where(_pos_mask), axis=1)
        _pos_loss = tf.gather(_loss, _pos_ind)
        _pos_k = tf.cast(tf.math.count_nonzero(_pos_mask)//5, tf.int32)
        _, k_ind = tf.nn.top_k(_pos_loss, _pos_k)
        _pos_loss = tf.gather(_pos_loss, k_ind)
        
        _neg_k = tf.cast(tf.math.count_nonzero(_pos_loss)*3, tf.int32)
        _neg_mask = tf.equal(_labels, 0)
        _neg_ind = tf.squeeze(tf.where(_neg_mask), axis=1)
        _neg_loss = tf.gather(_loss, _neg_ind)
        _, k_ind = tf.nn.top_k(_neg_loss, _neg_k)
        _neg_loss = tf.gather(_neg_loss, k_ind)
        
        _loss = tf.reduce_mean(_pos_loss) + tf.reduce_mean(_neg_loss)
        return _loss
    return tf.map_fn(batch_fn, [labels, preds], tf.float32)
    
def focal(alpha=0.25, gamma=1.5):
    """
    Create a functor for computing the focal loss.
    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred, y_mask):
        """
        Compute the focal loss given the target tensor and the predicted tensor.
        As defined in https://arxiv.org/abs/1708.02002
        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels = y_true#[:, :, :-1]
        # -1 for ignore, 0 for background, 1 for object
        anchor_state = y_mask#y_true[:, :, -1]
        classification = y_pred

        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, 0))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal