poetry run python3 tools/train_factor_graph.py -c configs/train_factor_graph.gin -d /home/antonap/sparklab/dataset/aij/temporal && \
poetry run python3 tools/train_gnn.py -c configs/train_gcn.gin -d /home/antonap/sparklab/dataset/aij/temporal && \
poetry run python3 tools/train_gnn.py -c configs/train_gcn2.gin -d /home/antonap/sparklab/dataset/aij/temporal && \
poetry run python3 tools/train_gnn.py -c configs/train_sage.gin -d /home/antonap/sparklab/dataset/aij/temporal && \
poetry run python3 tools/train_gnn.py -c configs/train_gin.gin -d /home/antonap/sparklab/dataset/aij/temporal && \
poetry run python3 tools/baseline.py -d /home/antonap/sparklab/dataset/aij/temporal && \
poetry run python3 tools/deterministic.py -d /home/antonap/sparklab/dataset/aij/temporal