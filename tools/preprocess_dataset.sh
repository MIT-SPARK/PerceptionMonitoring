#!/usr/bin/env bash

poetry run python3 tools/preprocessor.py process \
  -d "/home/antonap/sparklab/apollo-master/data/bag/" \
  -o "/home/antonap/sparklab/dataset/" \
  -c "/home/antonap/sparklab/diagnosability/configs/dfg_obstacle_detection_without_mismatch_type.yaml" \
  -i "log_temporal_adaptive.dat" \
  --train 0.8 --validation 0.1 --test 0.1 -f
  # --test 1.0 --validation 0.0 --train 0.0  -f 

poetry run python3 tools/preprocessor.py merge \
  -d "/home/antonap/sparklab/apollo-master/data/bag/" \
  -o "/home/antonap/sparklab/dataset/" \
  --train 0.8 --validation 0.1 --test 0.1 -f

