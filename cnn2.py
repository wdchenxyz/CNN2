import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf 
tf.config.gpu.set_per_process_memory_growth(True)
import numpy as np

## Utility function
def cmpooling(fmaps, scale_list, pool_stride):
    # make sure the scale_list is in decending order
    if scale_list[0] - scale_list[1] < 0:
        scale_list = scale_list[::-1]
        
    # concentric multi-scale pooling
    offset = [0] + [-(scale_list[i+1] - scale_list[0])//2 for i in range(len(scale_list) - 1)]
    pool_maps = []
    for offset, scale in zip(offset, scale_list):
        slice_maps = tf.slice(fmaps, [0, offset, offset, 0], [-1, fmaps.shape[1]-offset*2, fmaps.shape[2]-offset*2, -1])
        pool_map = tf.nn.max_pool2d(slice_maps, scale, pool_stride, "VALID")
        pool_maps.append(pool_map)
    
    # assert same shape for all pool_map
    for i in range(len(pool_maps)-1):
        assert pool_maps[i].shape[1:] == pool_maps[-1].shape[1:]
    return pool_maps

# Concat the feature maps in different scale and convolution once. (paper version)
class Monocular(tf.keras.layers.Layer):
    def __init__(self, filters, ksize, **kwargs):
        super(Monocular, self).__init__(**kwargs)
        self.filters = filters
        self.ksize = ksize
    
    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(self.filters, self.ksize, input_shape=input_shape, activation='relu', padding='same')
    
    def call(self, fmaps, scale_list, pool_stride):
        pool_maps = cmpooling(fmaps, scale_list, pool_stride)
        pool_maps = tf.concat(pool_maps, axis=-1)
        return self.conv(pool_maps)

SCALE_LIST = [1, 3, 5]
def CNN2(input_shape, num_classes, scale_list):
    left_eye = tf.keras.Input(input_shape, name='left_eye')
    right_eye = tf.keras.Input(input_shape, name='right_eye')
    
    # parallax augmentation
    parallax = left_eye - right_eye 
    left = tf.concat([left_eye, -parallax], axis=-1)
    right = tf.concat([right_eye, parallax], axis=-1)
    # 
    left1 = Monocular(12, 5, input_shape=input_shape, name='mono1_left')(left, scale_list=scale_list, pool_stride=2)
    right1 = Monocular(12, 5, input_shape=input_shape, name='mono1_right')(right, scale_list=scale_list, pool_stride=2)
    
    left2 = Monocular(24, 5, name='mono2_left')(tf.concat([left1, right1], axis=-1), scale_list=scale_list, pool_stride=2)
    right2 = Monocular(24, 5, name='mono2_right')(tf.concat([right1, left1], axis=-1), scale_list=scale_list, pool_stride=2)
    
    left3 = Monocular(40, 5, name='mono3_left')(tf.concat([left2, right2], axis=-1), scale_list=scale_list, pool_stride=1)
    right3 = Monocular(40, 5, name='mono3_right')(tf.concat([right2, left2], axis=-1), scale_list=scale_list, pool_stride=1)
    
    merge_binocular = tf.concat([left3, right3], axis=-1)
    
    merge_binocular = tf.keras.layers.Conv2D(128, 3, strides=1, activation='relu', name='conv1')(merge_binocular)
    merge_binocular = tf.keras.layers.Conv2D(64, 1, strides=1, activation='relu', name='conv2')(merge_binocular)
    merge_binocular = tf.keras.layers.Conv2D(16, 1, strides=1, activation='relu', name='conv3')(merge_binocular)
    x = tf.keras.layers.Flatten()(merge_binocular)
    x = tf.keras.layers.Dropout(0.5)(x)
    predicted_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='dense')(x)
    
    return tf.keras.Model([left_eye, right_eye], predicted_output)