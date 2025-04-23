import warnings
warnings.filterwarnings("ignore")
import torch
import random
from transformers import LlamaConfig, MistralConfig, AutoTokenizer
from datasets import load_dataset
import argparse

""" modified from kivi example.py
"""

@torch.no_grad()
def main():
    # For reproducibility
    random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description='LLaMA model with SQuat')
    parser.add_argument('--dataset', type=str, default='math500', help='Dataset to use')
    parser.add_argument('--k_bits', type=int, default=2, help='K bits (2 or 4)')
    parser.add_argument('--v_bits', type=int, default=2, help='V bits (2 or 4)') 
    parser.add_argument('--group_size', type=int, default=32, help='Group size for quantization')
    parser.add_argument('--residual_length', type=int, default=32, help='Number of recent fp16 tokens')
    parser.add_argument('--model_path', type=str, default='meta-llama/Llama-3.1-8B-Instruct', help='Path to pretrained model')
    parser.add_argument('--prompt', type=str, default="Repeat the following sentence: 'The capital of California is Beijing.'", help='Input prompt')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate')
    parser.add_argument('--method', type=str, default='squat', help='Method to use')
    parser.add_argument('--subspace_dim', type=int, default=20, help='Subspace dimension')
    parser.add_argument('--squat_lambda', type=float, default=0.001, help='Lambda for Lagrangian')
    parser.add_argument('--quant_group_size', type=int, default=64, help='Quantization group size')
    parser.add_argument("--shared_svd", type=str, default="false", help='Shared SVD')
    parser.add_argument("--fill_key_quant_with_first_col", type=str, default="false")
    parser.add_argument("--force_use_flash", action="store_true")
    parser.add_argument("--save_config_only", action="store_true")
    parser.add_argument("--save_config_path", type=str, default="config.json")
    parser.add_argument("--squat_query_subspace_prerope", type=str, default="after")
    parser.add_argument("--power_method", action="store_true")
    parser.add_argument("--cuda_bmm_implementation", type=str, default="cos_sin")
    parser.add_argument("--key_quant", type=str, default='per_channel', choices=['per_channel', 'per_token'])
    parser.add_argument("--no_hadamard", action="store_true")
    parser.add_argument("--strict_no_residual", action="store_true")
    args = parser.parse_args()

    if "mistral" in args.model_path.lower():
        config = MistralConfig.from_pretrained(args.model_path)
    else:
        config = LlamaConfig.from_pretrained(args.model_path)
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length
    config.use_flash = True
    config.attn_implementation = "flash_attention_2"
    config.method = args.method
    config.subspace_dim = args.subspace_dim
    config.squat_lambda = args.squat_lambda
    config.quant_group_size = args.quant_group_size
    config.fill_key_quant_with_first_col = args.fill_key_quant_with_first_col
    config.shared_svd = args.shared_svd
    config.power_method = args.power_method
    config.squat_query_subspace_prerope = args.squat_query_subspace_prerope
    config.force_use_flash = args.force_use_flash
    config.cuda_bmm_implementation = args.cuda_bmm_implementation
    config.key_quant = args.key_quant
    config.no_hadamard = args.no_hadamard
    config.strict_no_residual = args.strict_no_residual
    if "mistral" in args.model_path.lower():
        if "kivi" in args.method:
            from models.mistral_kivi import MistralForCausalLM_KIVI
            model = MistralForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            ).cuda()
        elif "squat" in args.method:
            from models.mistral_squat import MistralForCausalLM_SQuat
            model = MistralForCausalLM_SQuat.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            ).cuda()
        elif args.method in ["fp16", "baseline"]:
            from transformers import MistralForCausalLM
            model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).cuda()
        else:
            raise NotImplementedError(f"Method {args.method} not implemented")
    else:
        if "kivi" in args.method:
            from models.llama_kivi import LlamaForCausalLM_KIVI
            model = LlamaForCausalLM_KIVI.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            ).cuda()
        elif "squat" in args.method:
            from models.llama_squat import LlamaForCausalLM_SQuat
            model = LlamaForCausalLM_SQuat.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            ).cuda()
        elif args.method in ["fp16", "baseline"]:
            from transformers import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=args.model_path,
                config=config,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).cuda()
        else:
            raise NotImplementedError(f"Method {args.method} not implemented")
    
    if not args.no_hadamard:
        from rotation_utils import rotate_model_v
        rotate_model_v(model, config)
        print("Rotated model")

    enc = AutoTokenizer.from_pretrained(
        args.model_path, 
        use_fast=False, 
        trust_remote_code=True)

    dataset = load_dataset('gsm8k', 'main')

    prompt = ''
    for i in range(5):
        prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
    prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
    inputs = enc(prompt, return_tensors="pt").input_ids.to('cuda')

    output = model.generate(
        inputs, 
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
