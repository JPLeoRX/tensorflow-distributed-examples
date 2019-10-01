from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from cifar import main

# Create cluster
# This variable must be set on each worker with changing index
# Create cluster
# This variable must be set on each worker with changing index
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'chief': ["178.63.21.5:2222"],
        'worker': ["195.201.84.116:2222"]
    },
    'task': {'type': 'chief', 'index': 0}
})

main()

exit()