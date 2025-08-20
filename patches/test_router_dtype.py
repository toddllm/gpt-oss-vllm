#!/usr/bin/env python3
"""
Minimal loader to instantiate GPT-OSS block and verify router dtype.
Does not start full server; imports the model class and checks dtype.
"""
import sys
import os
sys.path.insert(0, '/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages')

import torch
from vllm.config import ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.transformers_utils.configs import GptOssConfig

# Configure a tiny VllmConfig to build the module graph without loading weights
hf_cfg = GptOssConfig(
    hidden_size=2880,
    num_hidden_layers=1,
    num_attention_heads=30,
    num_key_value_heads=5,
    intermediate_size=2880,
    num_local_experts=32,
    num_experts_per_tok=4,
    max_position_embeddings=2048,
    rope_theta=10000.0,
    rope_scaling={"factor": 1.0, "original_max_position_embeddings": 2048},
    rope_ntk_beta=32,
    rope_ntk_alpha=1.0,
    vocab_size=32000,
)

model_config = ModelConfig(
    model="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    tokenizer="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    trust_remote_code=True,
    dtype=torch.float16,
)

vllm_config = VllmConfig(model_config=model_config,
                         cache_config=CacheConfig(),
                         parallel_config=ParallelConfig(),
                         scheduler_config=SchedulerConfig())

from vllm.model_executor.models.gpt_oss import MLPBlock

mlp = MLPBlock(hf_cfg, layer_idx=0, quant_config=None, prefix="model.layers.0.mlp")

print("Router dtype:", mlp.router.weight.dtype)
print("Experts present:", hasattr(mlp, 'experts'))
