import json
import re

import torch
from composer.models import write_huggingface_pretrained_from_composer_checkpoint

def update_config(source_config):
    # Create target config with values from first config format
    target_config = {
        # "_name_or_path": "ModernBERT-base",
        "architectures": ["ModernBertForMaskedLM"],
        "attention_bias": source_config["attn_out_bias"],
        "attention_dropout": source_config["attention_probs_dropout_prob"],
        "bos_token_id": 50281,
        "classifier_activation": "gelu", #source_config["head_class_act"],
        "classifier_bias": source_config["head_class_bias"],
        "classifier_dropout": source_config["head_class_dropout"],
        "classifier_pooling": "mean",
        "cls_token_id": 50281,
        "decoder_bias": source_config["decoder_bias"],
        "deterministic_flash_attn": source_config["deterministic_fa2"],
        "embedding_dropout": source_config["embed_dropout_prob"],
        "eos_token_id": 50282,
        "global_attn_every_n_layers": source_config["global_attn_every_n_layers"],
        "global_rope_theta": source_config["rotary_emb_base"],
        "gradient_checkpointing": source_config["gradient_checkpointing"],
        "hidden_activation": source_config["hidden_act"],
        "hidden_size": source_config["hidden_size"],
        "initializer_cutoff_factor": source_config["init_cutoff_factor"],
        "initializer_range": source_config["initializer_range"],
        "intermediate_size": source_config["intermediate_size"],
        "layer_norm_eps": source_config["norm_kwargs"]["eps"],
        "local_attention": source_config["sliding_window"],
        "local_rope_theta": source_config["local_attn_rotary_emb_base"] if source_config["local_attn_rotary_emb_base"] else source_config["rotary_emb_base"],
        "max_position_embeddings": 8192,  # Override with first config value
        "mlp_bias": source_config["mlp_in_bias"],
        "mlp_dropout": source_config["mlp_dropout_prob"],
        "model_type": "modernbert",
        "norm_bias": source_config["norm_kwargs"]["bias"],
        "norm_eps": source_config["norm_kwargs"]["eps"],
        "num_attention_heads": source_config["num_attention_heads"],
        "num_hidden_layers": source_config["num_hidden_layers"],
        "pad_token_id": 50283,
        "position_embedding_type": source_config["position_embedding_type"],
        "sep_token_id": 50282,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.48.0",
        "vocab_size": source_config["vocab_size"]
    }
        
    return target_config

    


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="model name e.g. google-bert/bert-base-uncased",
)
args = parser.parse_args()

write_huggingface_pretrained_from_composer_checkpoint(
    "checkpoints/{args.model_name}/latest-rank0.pt", f"hf_checkpoint/{args.model_name}/"
)

state_dict = torch.load(
    f"hf_checkpoint/{args.model_name}/pytorch_model.bin",
    map_location=torch.device("cpu"),
)
var_map = ((re.compile(r"encoder\.layers\.(.*)"), r"layers.\1"),)
for pattern, replacement in var_map:
    state_dict = {
        re.sub(pattern, replacement, name): tensor
        for name, tensor in state_dict.items()
    }
torch.save(state_dict, f"hf_checkpoint/{args.model_name}/pytorch_model.bin")

with open(f"hf_checkpoint/{args.model_name}/config.json", "r") as f:
    config_dict = json.load(f)

config_dict = update_config(config_dict)

# Save the modified config
with open(f"hf_checkpoint/{args.model_name}/config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

with open(f"hf_checkpoint/{args.model_name}/tokenizer_config.json", "r") as f:
    config_dict = json.load(f)

config_dict["model_max_length"] = 8192
config_dict["added_tokens_decoder"]["50284"]["lstrip"] = True
config_dict["model_input_names"] = ["input_ids", "attention_mask"]
config_dict["tokenizer_class"] = "PreTrainedTokenizerFast"
if "extra_special_tokens" in config_dict:
    del config_dict["extra_special_tokens"]
with open(f"hf_checkpoint/{args.model_name}/tokenizer_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)


with open(f"hf_checkpoint/{args.model_name}/special_tokens_map.json", "r") as f:
    config_dict = json.load(f)

config_dict["mask_token"]["lstrip"] = True
with open(f"hf_checkpoint/{args.model_name}/special_tokens_map.json", "w") as f:
    json.dump(config_dict, f, indent=2)
