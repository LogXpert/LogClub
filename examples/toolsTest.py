

import json
import logging
import os
import subprocess
import sys
import time
from os.path import dirname


scope_dir = os.path.join(os.path.dirname(__file__), "../code")
sys.path.insert(0, scope_dir)

from template_miner_tools import TemplateMiner
from template_miner_config import TemplateMinerConfig

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

in_log_file = os.path.join(os.path.dirname(__file__), "scopeTestFile.txt")

config = TemplateMinerConfig()
config.load(f"{dirname(__file__)}/tools.ini")
config.profiling_enabled = True
template_miner = TemplateMiner(config=config)

line_count = 0

def preprocess_log(data_file):
    lineStrings = []

    with open(data_file, 'r', errors='ignore') as f:
        for log_line in f:

            parts = log_line.split()

            if len(parts) > 1 and parts[1].lower().startswith(("po-", "fsp-")):

                lineStrings.append(' '.join(parts[5:]))
            else:
                lineStrings.append(log_line)

    return lineStrings


lines = preprocess_log(in_log_file)

start_time = time.time()
batch_start_time = start_time
batch_size = 10000

for line in lines:
    line = line.rstrip()

    result = template_miner.add_log_message(line)
    line_count += 1
    if line_count % batch_size == 0:
        time_took = time.time() - batch_start_time
        rate = batch_size / time_took

        batch_start_time = time.time()
    if result["change_type"] != None:
        result_json = json.dumps(result)

        logger.info(f"Result: {result_json}")

time_took = time.time() - start_time
rate = line_count / time_took
logger.info(f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
            f"{len(template_miner.scope.clusters)} clusters")

sorted_clusters = sorted(template_miner.scope.clusters, key=lambda it: it.matchedLogSize, reverse=True)
for cluster in sorted_clusters:

    if(cluster.matchedLogSize == 1):
        print("size = 1", cluster.getTemplateStr())
    else:
        print("siez != 1", cluster.getTemplateStr())




template_miner.profiler.report(0)
