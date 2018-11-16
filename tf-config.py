from __future__ import print_function

import sys
import json

rank = sys.argv[1]

print(json.dumps({ "worker": ["%s:12180"%host.strip() for host in sys.stdin],
        "task": {"type": "worker", "index": sys.argv[1] } }))

