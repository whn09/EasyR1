docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all -v /home/ubuntu/EasyR1:/workspace/EasyR1 hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

cd EasyR1

# pip3 install torch==2.9.0+cu129 torchvision==0.24.0+cu129 torchaudio==2.9.0+cu129 --index-url https://download.pytorch.org/whl/cu129

pip install -e .

# pip install flash-attn --no-build-isolation

wandb offline

bash examples/qwen3_vl_4b_oil_gauge_grpo.sh

python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/qwen3_vl_4b_oil_gauge_grpo/global_step_1/actor
