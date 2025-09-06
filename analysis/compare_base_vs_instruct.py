print("Version 4.5.0") 

import sys
import io
from datetime import datetime
import os

# Global variable to store all output
captured_output = []

def log_print(*args, **kwargs):
    """Function that both prints to console and saves to our global variable"""
    # Convert all arguments to strings and join them
    output_str = " ".join(str(arg) for arg in args)
    
    # Handle the 'end' parameter (default to newline if not specified)
    end = kwargs.get('end', '\n')
    output_str += end
    
    # Print to console
    print(*args, **kwargs)
    
    # Save to our global variable
    captured_output.append(output_str)

# Function to save captured output to file
def save_captured_output(filename):
    """Save all captured output to a file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(captured_output)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    __version__ as transformers_version
)
import gc
import psutil
import os
import shutil
from huggingface_hub import login
import logging
import contextlib
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

def get_memory_usage():
    """Get current memory usage and disk space statistics"""
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Get disk space info
    total, used, free = shutil.disk_usage("/")
    disk_free_gb = free / (1024**3)
    
    if torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB
        return f"RAM: {ram_usage:.0f}MB, GPU Used: {gpu_usage:.0f}MB, GPU Cached: {gpu_cached:.0f}MB, Disk Free: {disk_free_gb:.1f}GB"
    return f"RAM: {ram_usage:.0f}MB, Disk Free: {disk_free_gb:.1f}GB"

def clear_memory(model_name=None):
    """Aggressively clear memory and disk cache"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Triple GC collection for more thorough cleanup
    for _ in range(3):
        gc.collect()
        
    # Clear model cache if model_name is provided
    if model_name:
        try:
            cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                log_print(f"Cleared cache for {model_name}")
        except Exception as e:
            log_print(f"Warning: Could not clear cache for {model_name}: {e}")
    
    log_print(f"Memory after cleanup: {get_memory_usage()}")

def prepare_data_for_csv(outputs, model_pairs):
    """Prepare model outputs for CSV export"""
    data = []
    for base_name, instruct_name in model_pairs:
        # Extract model family name (e.g., 't5', 'llama', etc.)
        for model_name in [base_name, instruct_name]:
            model_family = model_name.split('/')[1].split('-')[0].lower()
            base_or_instruct = 'base' if model_name == base_name else 'instruct'
            
            for prompt in outputs.get(model_name, {}):
                result = outputs[model_name][prompt]
                data.append({
                    'prompt': prompt,
                    'model': model_name,
                    'model_family': model_family,
                    'base_or_instruct': base_or_instruct,
                    'model_output': result.get('completion', 'N/A'),
                    'yes_prob': result.get('yes_prob', float('nan')),
                    'no_prob': result.get('no_prob', float('nan')),
                    'odds_ratio': result.get('odds_ratio', float('nan'))
                })
    return pd.DataFrame(data)

# Set up Hugging Face authentication
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HUGGINGFACE_TOKEN not found. Some models may not be accessible.")

device = "cuda" if torch.cuda.is_available() else "cpu"
log_print("Using device:", device)
log_print(f"Transformers version: {transformers_version}")

# Memory optimization settings
USE_8BIT = True  # Load models in 8-bit quantization
USE_4BIT = False  # Even more aggressive quantization (only works with some models)
LOW_CPU_MEM_USAGE = True  # Load model weights in chunks to reduce memory spikes
UNLOAD_UNUSED_MODELS = True  # Explicitly unload unused model components

# Here is an extended list of (base_model, instruct_model) pairs.
# CAUTION: This is a lot of models to load in one script.
model_pairs = [
    # 1) T5 base vs. Flan-T5
    ("google/t5-v1_1-base", "google/flan-t5-base"),

    # 2) EleutherAI Pythia vs. Dolly v2
    ("EleutherAI/pythia-6.9b", "databricks/dolly-v2-7b"),

    # 3) StableLM Base vs. StableLM Tuned
    ("stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-7b"),

    # 4) Mosaic MPT-7B vs. MPT-7B Instruct
    # ("mosaicml/mpt-7b", "mosaicml/mpt-7b-instruct"), # CURRENTLY DROPPING THIS ONE BECAUSE IT REQUIRES TRANSFORMERS VERSION 4.33.0

    # 5) Meta LLaMA2-7B Base vs. LLaMA2-7B Chat
    #   Requires you to have accepted the license on Hugging Face
    #   (https://huggingface.co/meta-llama/Llama-2-7b-hf)
    #   and be logged in via `huggingface-cli login`.
    ("meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf"),

    # 6) TII Falcon 7B Base vs. Falcon 7B Instruct
    ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"),

    # 7) Mistral 7B Base vs. Mistral 7B Instruct
    ("mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"),

    # 8) Qwen 7B Base vs. Qwen 7B Chat
    #   Requires you to agree to the Qwen license on HF:
    #   https://huggingface.co/Qwen/Qwen-7B
    ("Qwen/Qwen-7B", "Qwen/Qwen-7B-Chat"),

    # 9) Baichuan2 7B Base vs. Baichuan2 7B Chat
    #   Requires you to agree to the Baichuan2 license:
    #   https://huggingface.co/baichuan-inc/Baichuan2-7B-Base
    # ("baichuan-inc/Baichuan2-7B-Base", "baichuan-inc/Baichuan2-7B-Chat"), # CURRENTLY DROPPING THIS ONE BECAUSE IT ALSO SEEMS TO REQUIRE AN OLDER VERSION OF TRANSFORMERS

    # 10) RedPajama-INCITE Base vs. Instruct
    ("togethercomputer/RedPajama-INCITE-7B-Base", "togethercomputer/RedPajama-INCITE-7B-Instruct"),

    # 11) Salesforce XGen 7B Base vs. XGen 7B Inst
    # ("Salesforce/xgen-7b-8k-base", "Salesforce/xgen-7b-8k-inst"), # Currently dropping this one because there's a problem with the "add_special_tokens" parameter

    # 12) BLOOM vs. BLOOMZ
    ("bigscience/bloom-7b1",  "bigscience/bloomz-7b1"),

]

# Dictionary to store outputs
outputs = {}

def get_yes_no_logprobs(model, tokenizer, prompt, device):
    """Get log probabilities for 'Yes' and 'No' responses and generate completion"""
    MAX_LOOK_AHEAD = 10  # Maximum number of tokens to look ahead for Yes/No
    
    # Format input
    is_encoder_decoder = hasattr(model, 'encoder') and hasattr(model, 'decoder')
    
    if is_encoder_decoder:
        # For T5 models
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        # Generate with scores
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Get token IDs for Yes and No
        yes_token_id = tokenizer("Yes").input_ids[0]
        no_token_id = tokenizer("No").input_ids[0]
        
        # Look through each position's scores
        yes_no_found = False
        position_found = -1
        
        for pos, scores in enumerate(outputs.scores[:MAX_LOOK_AHEAD]):
            scores = scores[0]  # Get scores for first (only) sequence
            probs = torch.softmax(scores, dim=-1)
            
            # Get top token probabilities
            top_probs, top_tokens = torch.topk(probs, k=2)
            
            # Check if Yes or No is among top tokens
            if yes_token_id in top_tokens or no_token_id in top_tokens:
                yes_prob = probs[yes_token_id].item()
                no_prob = probs[no_token_id].item()
                yes_no_found = True
                position_found = pos
                break
        
        if not yes_no_found:
            # If no clear Yes/No found, use first position
            scores = outputs.scores[0][0]
            probs = torch.softmax(scores, dim=-1)
            yes_prob = probs[yes_token_id].item()
            no_prob = probs[no_token_id].item()
            position_found = 0
        
        # Decode the completion
        completion = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
    else:
        # For decoder-only models
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        yes_tokens = tokenizer(" Yes").input_ids
        no_tokens = tokenizer(" No").input_ids
        yes_token_id = yes_tokens[0]
        no_token_id = no_tokens[0]
        
        with torch.no_grad():
            # Generate completion
            completion_ids = model.generate(
                inputs.input_ids,
                max_length=len(inputs.input_ids[0]) + 50,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Look through each position's scores
            yes_no_found = False
            position_found = -1
            
            # Get all logits for the sequence
            for pos, scores in enumerate(completion_ids.scores[:MAX_LOOK_AHEAD]):
                probs = torch.softmax(scores[0], dim=-1)
                
                # Get top token probabilities
                top_probs, top_tokens = torch.topk(probs, k=2)
                
                # Check if Yes or No is among top tokens
                if yes_token_id in top_tokens or no_token_id in top_tokens:
                    yes_prob = probs[yes_token_id].item()
                    no_prob = probs[no_token_id].item()
                    yes_no_found = True
                    position_found = pos
                    break
            
            if not yes_no_found:
                # If no clear Yes/No found, use first position
                scores = completion_ids.scores[0]
                probs = torch.softmax(scores[0], dim=-1)
                yes_prob = probs[yes_token_id].item()
                no_prob = probs[no_token_id].item()
                position_found = 0
            
            completion = tokenizer.decode(completion_ids.sequences[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # Calculate log probabilities and odds ratio
    yes_logprob = np.log(yes_prob) if yes_prob > 0 else float('-inf')
    no_logprob = np.log(no_prob) if no_prob > 0 else float('-inf')
    odds_ratio = yes_prob / no_prob if no_prob > 0 else float('inf')
    
    return {
        'yes_logprob': yes_logprob,
        'no_logprob': no_logprob,
        'yes_prob': yes_prob,
        'no_prob': no_prob,
        'odds_ratio': odds_ratio,
        'prompt': prompt,
        'completion': completion.strip(),
        'position_found': position_found,
        'yes_no_found': yes_no_found
    }

# List of questions to ask each model
prompts = [
    'Is a "screenshot" a "photograph"?',
    'Is "advising" someone "instructing" them?',
    'Is an "algorithm" a "procedure"?',
    'Is a "drone" an "aircraft"?',
    'Is "reading aloud" a form of "performance"?',
    'Is "training" an AI model "authoring" content?',
    'Is a "wedding" a "party"?',
    'Is "streaming" a video "broadcasting" that video?',
    'Is "braiding" hair a form of "weaving"?',
    'Is "digging" a form of "construction"?',
    'Is a "smartphone" a "computer"?',
    'Is a "cactus" a "tree"?',
    'Is a "bonus" a form of "wages"?',
    'Is "forwarding" an email "sending" that email?',
    'Is a "chatbot" a "service"?',
    'Is "plagiarism" a form of "theft"?',
    'Is "remote viewing" of an event "attending" it?',
    'Is "whistling" a form of "music"?',
    'Is "caching" data in computer memory "storing" that data?',
    'Is a "waterway" a form of "roadway"?',
    'Is a "deepfake" a "portrait"?',
    'Is "humming" a form of "singing"?',
    'Is "liking" a social media post "endorsing" it?',
    'Is "herding" animals a form of "transporting" them?',
    'Is an "NFT" a "security"?',
    'Is "sleeping" an "activity"?',
    'Is a "driverless car" a "motor vehicle operator"?',
    'Is a "subscription fee" a form of "purchase"?',
    'Is "mentoring" someone a form of "supervising" them?',
    'Is a "biometric scan" a form of "signature"?',
    'Is a "digital wallet" a "bank account"?',
    'Is "dictation" a form of "writing"?',
    'Is a "virtual tour" a form of "inspection"?',
    'Is "bartering" a form of "payment"?',
    'Is "listening" to an audiobook "reading" it?',
    'Is a "nest" a form of "dwelling"?',
    'Is a "QR code" a "document"?',
    'Is a "tent" a "building"?',
    'Is a "whisper" a form of "speech"?',
    'Is "hiking" a form of "travel"?',
    'Is a "recipe" a form of "instruction"?',
    'Is "daydreaming" a form of "thinking"?',
    'Is "gossip" a form of "news"?',
    'Is a "mountain" a form of "hill"?',
    'Is "walking" a form of "exercise"?',
    'Is a "candle" a "lamp"?',
    'Is a "trail" a "road"?',
    'Is "repainting" a house "repairing" it?',
    'Is "kneeling" a form of "sitting"?',
    'Is a "mask" a form of "clothing"?'
]

# Context manager to disable HF progress bars
@contextlib.contextmanager
def suppress_progress_bars():
    """Context manager to suppress HuggingFace progress bar outputs"""
    # Disable logging for transformers
    log = logging.getLogger("transformers")
    log.setLevel(logging.ERROR)
    
    # Disable progress bars from tqdm
    original_tqdm = tqdm.__init__
    
    def silent_tqdm(*args, **kwargs):
        kwargs['disable'] = True
        return original_tqdm(*args, **kwargs)
    
    tqdm.__init__ = silent_tqdm
    
    try:
        yield
    finally:
        # Restore original tqdm
        tqdm.__init__ = original_tqdm
        # Restore logging level
        log.setLevel(logging.INFO)

for base_name, instruct_name in model_pairs:
    # Initialize nested dictionaries for this model pair
    outputs[base_name] = {}
    outputs[instruct_name] = {}
    
    for model_name in [base_name, instruct_name]:
        log_print(f"\n=== Loading model: {model_name} ===")
        log_print(f"System state before loading: {get_memory_usage()}")
        model = None  # Initialize model as None
        try:
            # Skip MPT models if transformers version is incompatible
            if "mpt" in model_name.lower() and transformers_version < "4.33.0":
                raise ValueError(f"MPT models require transformers >= 4.33.0, current version: {transformers_version}")

            # Load tokenizer and model with progress bars suppressed
            with suppress_progress_bars():
                # Load tokenizer with minimal memory usage
                tokenizer_kwargs = {
                    "use_fast": True,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": LOW_CPU_MEM_USAGE
                }
                
                # Special handling for XGen models
                if "xgen" in model_name.lower():
                    # XGen models need special handling - pass parameters directly
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        low_cpu_mem_usage=LOW_CPU_MEM_USAGE
                    )
                else:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        **tokenizer_kwargs
                    )

                # Common model loading arguments
                model_kwargs = {
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": LOW_CPU_MEM_USAGE
                }

                # Add quantization config if enabled
                if USE_8BIT:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                elif USE_4BIT:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                else:
                    model_kwargs["torch_dtype"] = torch.float16

                # For T5-like models, use AutoModelForSeq2SeqLM
                if "t5" in model_name.lower():
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )
                else:
                    # For most decoder-only LLMs
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )

            # Process each prompt
            for prompt in prompts:
                log_print(f"\nProcessing prompt: {prompt}")
                
                # Format prompt differently for base vs instruction models
                few_shot_examples = 'Question: Is "soup" a "beverage"? Answer either \'Yes\' or \'No\', without any other text.\nAnswer: No.\n\nQuestion: Is a "tweet" a "publication"? Answer either \'Yes\' or \'No\', without any other text.\nAnswer: Yes.\n\n'
                formatted_prompt = f"{few_shot_examples}Question: {prompt} Answer either 'Yes' or 'No', without any other text.\nAnswer:" if "base" in model_name.lower() or model_name.lower() in ["bigscience/bloom-7b1"] else f"{few_shot_examples}{prompt} Answer either 'Yes' or 'No', without any other text."
                log_print(f"Formatted prompt: {formatted_prompt}")
                
                # Get log probabilities
                try:
                    logprobs = get_yes_no_logprobs(model, tokenizer, formatted_prompt, device)
                    outputs[model_name][prompt] = logprobs
                    log_print(f"Model completion: {logprobs['completion']}")
                    log_print(f"Yes prob: {logprobs['yes_prob']:.3f}, No prob: {logprobs['no_prob']:.3f}, Odds ratio: {logprobs['odds_ratio']:.3f}")
                except Exception as e:
                    log_print(f"Error getting logprobs: {e}")
                    outputs[model_name][prompt] = {
                        'yes_prob': float('nan'),
                        'no_prob': float('nan'),
                        'odds_ratio': float('nan'),
                        'prompt': formatted_prompt,  # Store the formatted prompt even if we get an error
                        'completion': 'N/A'
                    }

        except Exception as e:
            log_print(f"Error running {model_name}: {e}")
            # Store error for all prompts
            for prompt in prompts:
                outputs[model_name][prompt] = {
                    'yes_prob': float('nan'),
                    'no_prob': float('nan'),
                    'odds_ratio': float('nan'),
                    'prompt': formatted_prompt,  # Store the formatted prompt even if we get an error
                    'completion': 'N/A'
                }

        finally:
            # Aggressive cleanup
            if model is not None:
                model.cpu()  # Move model to CPU first
                del model
                if UNLOAD_UNUSED_MODELS:
                    del tokenizer
                    if 'output_ids' in locals():
                        del output_ids
                    if 'input_ids' in locals():
                        del input_ids
                    clear_memory(model_name)  # Pass model_name for cache cleanup
                    log_print(f"System state after model unloading: {get_memory_usage()}")

# Save results to CSV
log_print("\nSaving results to CSV file...")
df = prepare_data_for_csv(outputs, model_pairs)
csv_filename = "gdrive/My Drive/Computational/llm_interpretation/model_comparison_results.csv"
df.to_csv(csv_filename, index=False)
log_print(f"Results saved to {csv_filename}")

# Restore original stdout
sys.stdout = sys.__stdout__

# Print final summary
log_print("\n\n============= Final Model Responses =============\n")
for base_name, instruct_name in model_pairs:
    log_print(f"\nComparing {base_name} vs {instruct_name}")
    log_print("=" * 80 + "\n")
    
    for prompt in prompts:
        base_resp = outputs.get(base_name, {}).get(prompt, {})
        instr_resp = outputs.get(instruct_name, {}).get(prompt, {})
        log_print(f"Question: {prompt}")
        log_print("\nBASE MODEL:")
        log_print(f"Input prompt: {base_resp.get('prompt', 'N/A')}")
        log_print(f"Model completion: {base_resp.get('completion', 'N/A')}")
        log_print(f"Yes prob: {base_resp.get('yes_prob', 'N/A'):.3f}, "
              f"No prob: {base_resp.get('no_prob', 'N/A'):.3f}, "
              f"Odds ratio: {base_resp.get('odds_ratio', 'N/A'):.3f}")
        log_print(f"Found Yes/No at position: {base_resp.get('position_found', 'N/A')}, "
              f"Clear Yes/No found: {base_resp.get('yes_no_found', 'N/A')}")
        
        log_print("\nINSTRUCT/CHAT MODEL:")
        log_print(f"Input prompt: {instr_resp.get('prompt', 'N/A')}")
        log_print(f"Model completion: {instr_resp.get('completion', 'N/A')}")
        log_print(f"Yes prob: {instr_resp.get('yes_prob', 'N/A'):.3f}, "
              f"No prob: {instr_resp.get('no_prob', 'N/A'):.3f}, "
              f"Odds ratio: {instr_resp.get('odds_ratio', 'N/A'):.3f}")
        log_print(f"Found Yes/No at position: {instr_resp.get('position_found', 'N/A')}, "
              f"Clear Yes/No found: {instr_resp.get('yes_no_found', 'N/A')}")
        log_print("-" * 80 + "\n")

# Save the captured output to a file
output_file = "gdrive/My Drive/Computational/llm_interpretation/model_comparison_output.txt"
save_captured_output(output_file)
log_print(f"\nComplete output log has been saved to: {output_file}")
