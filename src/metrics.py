import tensorflow as tf


class PerformanceMetric:

    def __init__(self):        
        pass

    @tf.function
    def macro_F1(self, y, y_hat, thresh=0.5):
        """ 
        Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch

        """

        y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
        tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
        fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
        fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
        f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        return tf.reduce_mean(f1)

pass
