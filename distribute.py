#coding=utf-8
import numpy as np
import tensorflow as tf

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 0.000000001, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('steps_to_validate', 100000,
                     'Steps to validate and print loss')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

# Hyperparameters
learning_rate = FLAGS.learning_rate
steps_to_validate = FLAGS.steps_to_validate

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % FLAGS.task_index,
                    cluster=cluster)):
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
      tf.scalar_summary('cost', loss_value)
      summary_op = tf.merge_all_summaries()
 
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            logdir="./checkpoint/",
                            init_op=init_op,
                            summary_op=summary_op,
                            saver=saver,
                            global_step=global_step,
                            save_model_secs=60)      
    with sv.managed_session(server.target) as sess:
      step = 0
      while step < 1000000:
        train_x = np.random.randn(1)
        train_y = 2 * train_x + np.random.randn(1) * 0.33  + 10
        _, loss_v, step = sess.run([train_op, loss_value,global_step], feed_dict={input:train_x, label:train_y})
        if step % steps_to_validate == 0:
          w,b = sess.run([weight,biase])
          print("step: %d, weight: %f, biase: %f, loss: %f" %(step, w, b, loss_v))


def loss(label, pred):
  return tf.square(label - pred)



if __name__ == "__main__":
  tf.app.run()

















