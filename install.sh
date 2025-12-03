docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all -v /home/ubuntu/EasyR1:/workspace/EasyR1 -v /home/ubuntu/LLaMA-Factory/qwen3_4b_20epochs:/workspace/qwen3_4b_20epochs hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

cd EasyR1

# pip3 install torch==2.9.0+cu129 torchvision==0.24.0+cu129 torchaudio==2.9.0+cu129 --index-url https://download.pytorch.org/whl/cu129

pip install -e .

# pip install flash-attn --no-build-isolation

python examples/convert_data.py examples/train_data.json examples/train_data_converted.json
python examples/convert_data.py examples/test_data.json examples/test_data_converted.json

# wandb offline
# wandb login

bash examples/qwen3_vl_4b_oil_gauge_grpo.sh

python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/qwen3_vl_4b_oil_gauge_grpo/global_step_120/actor

sudo chown -R ubuntu:ubuntu /home/ubuntu/EasyR1/checkpoints/

mkdir -p /home/ubuntu/EasyR1/checkpoints/easy_r1/qwen3_vl_4b_oil_gauge_grpo/global_step_120/actor/eval/

llamafactory-cli train \
    --stage sft \
    --model_name_or_path /home/ubuntu/EasyR1/checkpoints/easy_r1/qwen3_vl_4b_oil_gauge_grpo/global_step_120/actor/huggingface \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --quantization_method bnb \
    --template qwen3_vl_nothink \
    --flash_attn auto \
    --dataset_dir data \
    --eval_dataset oil_gauge_test \
    --cutoff_len 1024 \
    --max_samples 100000 \
    --per_device_eval_batch_size 16 \
    --predict_with_generate True \
    --report_to none \
    --max_new_tokens 512 \
    --top_p 0.7 \
    --temperature 0.95 \
    --output_dir /home/ubuntu/EasyR1/checkpoints/easy_r1/qwen3_vl_4b_oil_gauge_grpo/global_step_120/actor/eval/ \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --do_predict True