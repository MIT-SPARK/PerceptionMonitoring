#!/usr/bin/env bash

poetry run python3 tools/postprocessor.py postprocess \
  --runid "384749f86d0d4127a051d1221d10ff2f" \
  --dir "/home/antonap/sparklab/apollo-master/data/bag/" \
  --config "/home/antonap/sparklab/diagnosability/configs/dfg_obstacle_detection_without_mismatch_type.yaml" \
  --input "log_regular_adaptive.dat" \
  --output "inference.zip"
