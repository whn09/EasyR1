#!/bin/bash

set -x

# MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct  # replace it with your local file path

# python3 -m verl.trainer.main \
#     config=examples/config-oil-gauge.yaml \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     trainer.experiment_name=qwen3_vl_4b_oil_gauge_grpo_nosft \
#     trainer.n_gpus_per_node=4

MODEL_PATH=/workspace/qwen3_4b_20epochs

python3 -m verl.trainer.main \
    config=examples/config-oil-gauge.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_4b_oil_gauge_grpo \
    trainer.n_gpus_per_node=4
