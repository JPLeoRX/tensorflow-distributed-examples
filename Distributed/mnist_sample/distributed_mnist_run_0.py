from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from distributed_mnist import main

# Create cluster
# This variable must be set on each worker with changing index
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'worker': ["10.111.54.31:2222", "10.111.54.32:2222"]
    },
    'task': {'type': 'worker', 'index': 0}
})

main()

exit()
