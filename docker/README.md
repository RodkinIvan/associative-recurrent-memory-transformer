# Docker usage
## Option 1 (pull and run):
```bash
docker pull rodkin/armt-pretrain:v1
```
Run with all GPUs:

```bash
docker run -dit \
  --gpus all \
  --memory=100g \
  --memory-swap=100g \
  --shm-size=32g \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -e WANDB_API_KEY=... \
  -e HF_TOKEN=... \
  rodkin/armt-pretrain:v1 bash docker/run_pipeline.sh
```

## Option 2 (build and run):
Build the image (from repo root):

```bash
docker build -t armt-pretrain -f Dockerfile .
```

Run with all GPUs:

```bash
docker run -dit \
  --gpus all \
  --memory=100g \
  --memory-swap=100g \
  --shm-size=32g \
  --ipc=host \
  --ulimit memlock=-1:-1 \
  -e WANDB_API_KEY=... \
  -e HF_TOKEN=... \
  armt-pretrain bash docker/run_pipeline.sh
```
Notes:
- The container captures all GPUs via `--gpus all`, and the script sets `CUDA_VISIBLE_DEVICES` accordingly.
- `create_env.sh` builds the `pretrain` conda env inside the image.
- The training script is `scripts/pretrain/finetune_armt_inner_gemma3-1b_fineweb_deepspeed.sh`; the runner auto-updates its `CUDA_VISIBLE_DEVICES` to use all GPUs in the container.


