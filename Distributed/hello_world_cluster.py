# Start a TensorFlow server as a single-process "cluster".

from __future__ import print_function
import tensorflow as tf

c = tf.constant("Hello, distributed TensorFlow!")
server = tf.distribute.Server.create_local_server() # Create a single-process cluster, with an in-process server.
sess = tf.compat.v1.Session(server.target)  # Create a session on the server.
result = sess.run(c)
print(result)
exit()