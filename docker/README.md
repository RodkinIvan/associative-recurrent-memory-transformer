# Docker usage

Build the image (from repo root):

```bash
docker build -t armt-pretrain -f Dockerfile .
```

Run with all GPUs and mount output/token cache dirs:

```bash
docker run -dit --gpus all --shm-size=16g \
  -e WANDB_API_KEY=... \
  -e HF_TOKEN=... \
  armt-pretrain bash docker/run_pipeline.sh
```

Notes:
- The container captures all GPUs via `--gpus all`, and the script sets `CUDA_VISIBLE_DEVICES` accordingly.
- `create_env.sh` builds the `pretrain` conda env inside the image.
- The training script is `scripts/pretrain/finetune_armt_inner_llama3.2_cc_sliding_deepspeed.sh`; the runner auto-updates its `CUDA_VISIBLE_DEVICES` to use all GPUs in the container.


