from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from distributed_mnist import main

# Create cluster
# This variable must be set on each worker with changing index
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'worker': ["178.63.21.5", "195.201.84.116:2222"]
    },
    'task': {'type': 'worker', 'index': 1}
})

main()

exit()
