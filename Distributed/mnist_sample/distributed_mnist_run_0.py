from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from distributed_mnist import main

# Create cluster
# This variable must be set on each worker with changing index
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'worker': ["195.201.84.116:2222", "195.201.84.118:2222"]
    },
    'task': {'type': 'worker', 'index': 0}
})

main()

exit()
