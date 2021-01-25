# Pytorch Sparse AdamW
This repository contains the sparse version of AdamW optimizer.

The SparseAdamW optimizer behaves like AdamW optimizer, but updates only the statistics for gradients which are computed, in the same way as SparseAdam optimizer. The optimizer can only be used on modules, which produce sparse gradients, e.g., nn.Embedding.

## Install
Install by running:
```bash
pip install torch-sparse-adamw
```
