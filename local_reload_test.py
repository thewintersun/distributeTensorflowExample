#coding=utf-8
import numpy as np
import tensorflow as tf

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.00003, 'Initial learning rate.')

# Hyperparameters
learning_rate = FLAGS.learning_rate

def main(_):

  global_step = tf.Variable(0, name='global_step', trainable=False)

  input = tf.placeholder("float")
  label = tf.placeholder("float")

  weight = tf.get_variable("weight", [1], tf.float32, initializer=tf.random_normal_initializer())
  biase  = tf.get_variable("biase", [1], tf.float32, initializer=tf.random_normal_initializer())
  pred = tf.mul(input, weight) + biase

  loss_value = loss(label, pred)

  train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value, global_step=global_step)
  init_op = tf.initialize_all_variables()
  
  saver = tf.train.Saver()
  session     = tf.Session()
  ckpt = tf.train.get_checkpoint_state("./checkpoint/")
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    saver.restore(session, ckpt.model_checkpoint_path)
    global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    print("%s, global_step = %d" % (ckpt.model_checkpoint_path, global_step))
  else:
	  return

  w,b = session.run([weight,biase])
  print("weight: %f, biase: %f" %(w, b))


def loss(label, pred):
  return tf.square(label - pred)

if __name__ == "__main__":
  tf.app.run()

















