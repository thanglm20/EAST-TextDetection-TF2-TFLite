#import tensorflow as tf
#from tensorflow.summary import FileWriter

#sess = tf.Session()
#tf.train.import_meta_graph("/media/tripc/data2/Text_detection/EAST_1/east_icdar2015_resnet_v1_50_rboxmodel.ckpt-331.meta")
#FileWriter("__tb", sess.graph)

import cv2
import time
import math
import os
from skimage import io
import numpy as np
import tensorflow as tf
from PIL import Image
import locality_aware_nms as nms_locality
import lanms
import copy
import model

tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './model_train/','')
tf.app.flags.DEFINE_string('meta', './model_train/model.ckpt-1.meta','')
tf.app.flags.DEFINE_string('output_dir', './export_model/', '')
FLAGS = tf.app.flags.FLAGS

def freeze():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        for op in sess.graph.get_operations():
            # print op
            print (op.name)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        # print variable_averages.variables_to_restore()

        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        output_node_names ="feature_fusion/Conv_7/Sigmoid,feature_fusion/concat_3"

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names.split(","))
        tf.graph_util.remove_training_nodes(frozen_graph_def)

        # Save the frozen graph
        with open(FLAGS.output_dir+'east_model.pb', 'wb') as f:
          f.write(frozen_graph_def.SerializeToString())

if __name__ == '__main__':
    freeze()
