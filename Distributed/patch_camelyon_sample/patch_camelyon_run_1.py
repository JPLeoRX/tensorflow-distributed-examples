from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from patch_camelyon import main

# Create cluster
# This variable must be set on each worker with changing index
# Create cluster
# This variable must be set on each worker with changing index
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'chief': ["172.23.100.10:2222"],
        'worker': ["172.23.100.11:2222"]
    },
    'task': {'type': 'worker', 'index': 0}
})

main()

exit()