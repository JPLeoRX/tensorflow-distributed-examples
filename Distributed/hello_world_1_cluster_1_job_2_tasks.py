from __future__ import print_function
import tensorflow as tf

message1 = tf.constant("Hello, distributed TensorFlow! This is sample message 1")
message2 = tf.constant("Hello, distributed TensorFlow! This is sample message 2")
jobName = "SampleJobName"
tasks = ["localhost:2222", "localhost:2223"]
jobs = {jobName: tasks}

# Create cluster
cluster = tf.train.ClusterSpec(jobs)

# Create servers
server1 = tf.train.Server(cluster, job_name=jobName, task_index=0)
server2 = tf.train.Server(cluster, job_name=jobName, task_index=1)

# Create sessions
session1 = tf.compat.v1.Session(server1.target)
session2 = tf.compat.v1.Session(server2.target)

# Execute
print(session1.run(message1))
print(session2.run(message2))

exit()
