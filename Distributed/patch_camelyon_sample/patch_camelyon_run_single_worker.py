from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os
from patch_camelyon_main import main

# Create cluster
# This variable must be set on each worker with changing index
os.environ["TF_CONFIG"] = json.dumps({
    'cluster': {
        'worker': ["localhost:2222"]
    },
    'task': {'type': 'worker', 'index': 0}
})

main()

exit()
