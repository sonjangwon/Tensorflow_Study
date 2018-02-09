import sys
import gensim
import tensorflow as tf

checkpoint_dir = sys.argv[1]
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'test')
print(checkpoint_dir + 'test')
