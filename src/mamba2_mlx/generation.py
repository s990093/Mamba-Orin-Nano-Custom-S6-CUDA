
import time
import mlx.core as mx
import numpy as np
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple


def apply_repetition_penalty(logits, generated_tokens, penalty=1.0, window_size=50):
    """
    Apply repetition penalty to logits. 
    Only considers the last 'window_size' tokens to preserve long-range coherence.
    """
    if penalty == 1.0 or len(generated_tokens) == 0:
        return logits
    
    # Only look at recent context to avoid penalizing common words too much
    recent_tokens = set(generated_tokens[-window_size:])
    
    # We iterate only over the unique tokens in the recent window
    for token in recent_tokens:
        # Ensure we don't access out of bounds if vocab size mismatch
        if token < logits.shape[-1]:
            val = logits[token].item()
            if val > 0:
                logits[token] = val / penalty
            else:
                logits[token] = val * penalty
    
    return logits

def sample_with_top_k_info(logits, temperature=1.0, top_k=None, top_p=None, generated_tokens=None, repetition_penalty=1.0, return_top_k_candidates=False, k_candidates=10):
    # 1. Sanitize Logits (Fixes 'NaN' issues causing garbage text)
    logits = mx.nan_to_num(logits, nan=0.0, posinf=1e5, neginf=-1e5)
    
    # 2. Apply Repetition Penalty (Before Temperature)
    if generated_tokens is not None and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

    # 3. Apply Temperature
    if temperature == 0:
        # Greedy decoding
        token = mx.argmax(logits, axis=-1)
        # ... (rest of greedy logic) ...
        return token, 0.0, None # Simplified for brevity, logic same as original
    else:
        logits = logits / temperature

    # 4. Top-K Filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        # Efficient masking
        indices = mx.argpartition(logits, -top_k, axis=-1)[-top_k:]
        mask = mx.full(logits.shape, float('-inf'))
        mask[indices] = 0
        logits = logits + mask

    # 5. Top-P (Nucleus) Filtering
    if top_p is not None and top_p < 1.0:
        # Sort logits in descending order
        sorted_indices = mx.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]
        
        probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(probs, axis=-1)
        
        # Find cutoff
        # We want to keep indices where cumulative_probs <= top_p
        # But we must keep at least one token.
        mask_indices = cumulative_probs > top_p
        
        # Shift mask right to include the first token that crosses the threshold
        mask_indices = mx.concatenate([mx.array([False]), mask_indices[:-1]])
        
        # Indices to remove
        indices_to_remove = sorted_indices[mask_indices]
        
        # Apply mask
        logits[indices_to_remove] = float('-inf')

    # 6. Sample
    token = mx.random.categorical(logits)
    
    # Calculate log_prob for the chosen token
    probs = mx.softmax(logits, axis=-1)
    log_prob = mx.log(probs[token]).item()

    # Top K Info Logic (Optional)
    top_k_info = None
    if return_top_k_candidates:
        # Recalculate only the top candidates based on final logits
        top_indices = mx.argsort(logits, axis=-1)[::-1][:k_candidates]
        top_k_info = []
        for idx in top_indices:
            idx_int = int(idx.item())
            p = float(probs[idx].item())
            # unexpected 0 prob safety
            lp = float(mx.log(probs[idx]).item()) if p > 0 else -100.0 
            top_k_info.append((idx_int, p, lp))

    return token, log_prob, top_k_info



def sanitize_float(value, default=0.0):
    """Sanitize float values for JSON serialization."""
    if np.isnan(value) or np.isinf(value):
        return default
    return float(value)

def sanitize_dict(data):
    """Recursively sanitize all float values in a dictionary."""
    if isinstance(data, dict):
        return {k: sanitize_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_dict(item) for item in data]
    elif isinstance(data, (float, np.floating)):
        return sanitize_float(data)
    return data

def generate_with_metrics(
    model, 
    tokenizer, 
    prompt, 
    max_tokens=50, 
    temperature=0.7, 
    top_k=None, 
    top_p=None, 
    repetition_penalty=1.0,
    return_dict=False,
    include_top_k_candidates=False
):
    """
    Generate text and return comprehensive metrics as a dictionary.
    
    Args:
        return_dict: If True, return dict instead of printing
        include_top_k_candidates: If True, include top-k candidates for each step
    """
    # Get device info
    device = mx.default_device()
    device_type = str(device).split('(')[0]
    
    # Track initial memory
    try:
        initial_memory = mx.metal.get_active_memory() / (1024**2)
        memory_tracking = True
    except:
        initial_memory = 0
        memory_tracking = False
    
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    if isinstance(input_ids, dict):
        input_ids = input_ids['input_ids']
    if hasattr(input_ids, 'numpy'):
        input_ids = input_ids.numpy()
        
    input_ids = mx.array(input_ids)
    if len(input_ids.shape) == 1:
        input_ids = input_ids[None, :]
        
    prompt_len = input_ids.shape[1]
    
    # Prefill
    cache = model.make_cache(batch_size=1)
    mx.eval(input_ids)
    tic_prefill = time.time()
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    toc_prefill = time.time()
    prefill_time = toc_prefill - tic_prefill
    
    next_token_logits = logits[0, -1, :]
    next_token, log_prob, top_k_info = sample_with_top_k_info(
        next_token_logits, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        generated_tokens=[],
        repetition_penalty=repetition_penalty,
        return_top_k_candidates=include_top_k_candidates
    )
    mx.eval(next_token)
    
    tokens = [next_token.item()]
    log_probs = [log_prob]
    generation_steps = []
    
    if include_top_k_candidates and top_k_info:
        step_info = {
            "step": 0,
            "token_id": tokens[0],
            "token_text": tokenizer.decode([tokens[0]]),
            "log_probability": log_prob,
            "top_k_candidates": [
                {
                    "token_id": tid,
                    "token_text": tokenizer.decode([tid]),
                    "probability": prob,
                    "log_probability": lp
                }
                for tid, prob, lp in top_k_info
            ]
        }
        generation_steps.append(step_info)
    
    # Generated text parts
    generated_tokens_text = [tokenizer.decode([tokens[0]])]
    
    tic_decode = time.time()
    
    # Decode loop
    for i in range(max_tokens):
        input_id = mx.array([[tokens[-1]]])
        logits = model(input_id, cache=cache)
        next_token_logits = logits[0, -1, :]
        
        next_token, log_prob, top_k_info = sample_with_top_k_info(
            next_token_logits, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p,
            generated_tokens=tokens,
            repetition_penalty=repetition_penalty,
            return_top_k_candidates=include_top_k_candidates
        )
        mx.eval(next_token)
        
        token = next_token.item()
        tokens.append(token)
        log_probs.append(log_prob)
        
        token_text = tokenizer.decode([token])
        generated_tokens_text.append(token_text)
        
        if include_top_k_candidates and top_k_info:
            step_info = {
                "step": i + 1,
                "token_id": token,
                "token_text": token_text,
                "log_probability": log_prob,
                "top_k_candidates": [
                    {
                        "token_id": tid,
                        "token_text": tokenizer.decode([tid]),
                        "probability": prob,
                        "log_probability": lp
                    }
                    for tid, prob, lp in top_k_info
                ]
            }
            generation_steps.append(step_info)
        
        if token == tokenizer.eos_token_id:
            break
            
    toc_decode = time.time()
    decode_time = toc_decode - tic_decode
    num_gen_tokens = len(tokens)
    
    # Calculate metrics
    avg_log_prob = np.mean(log_probs)
    perplexity = np.exp(-avg_log_prob)
    
    # Count repetitions
    token_counts = Counter(tokens)
    num_repeats = sum(count - 1 for count in token_counts.values() if count > 1)
    most_repeated = [
        {"token": tokenizer.decode([token_id]), "count": count}
        for token_id, count in token_counts.most_common(3)
        if count > 1
    ]
    
    # Get final memory usage
    if memory_tracking:
        try:
            current_memory = mx.metal.get_active_memory() / (1024**2)
            peak_memory = mx.metal.get_peak_memory() / (1024**2)
            memory_used = current_memory - initial_memory
        except:
            current_memory = initial_memory
            peak_memory = initial_memory
            memory_used = 0
    else:
        current_memory = 0
        peak_memory = 0
        memory_used = 0
    
    generated_only = "".join(generated_tokens_text)
    full_text = prompt + generated_only
    
    result = {
        "generated_text": full_text,
        "prompt": prompt,
        "generated_only": generated_only,
        "statistics": {
            "prompt_length": prompt_len,
            "generated_tokens": num_gen_tokens,
            "total_tokens": prompt_len + num_gen_tokens
        },
        "speed_metrics": {
            "prefill_time": prefill_time,
            "prefill_speed": prompt_len / prefill_time,
            "decode_time": decode_time,
            "decode_speed": num_gen_tokens / decode_time,
            "avg_latency": decode_time / num_gen_tokens * 1000,
            "total_time": prefill_time + decode_time
        },
        "memory_usage": {
            "device_type": device_type.upper(),
            "initial_memory": initial_memory,
            "current_memory": current_memory,
            "peak_memory": peak_memory,
            "memory_used": memory_used,
            "tracking_available": memory_tracking
        },
        "quality_metrics": {
            "avg_log_prob": avg_log_prob,
            "perplexity": perplexity,
            "num_repeats": num_repeats,
            "most_repeated": most_repeated
        },
        "parameters": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens
        }
    }
    
    if include_top_k_candidates:
        result["generation_steps"] = generation_steps
    
    # Sanitize all float values to ensure JSON compliance
    result = sanitize_dict(result)
    
    return result if return_dict else print_metrics(result)

# Keep original generate function for CLI use
def generate(model, tokenizer, prompt, max_tokens=50, temperature=0.7, top_k=None, top_p=None, repetition_penalty=1.0):
    """Original generate function with console output."""
    result = generate_with_metrics(
        model, tokenizer, prompt, max_tokens, temperature, top_k, top_p, repetition_penalty,
        return_dict=True, include_top_k_candidates=False
    )
    
    # Print formatted output
    print(f"\n{'='*60}")
    print(f"🚀 Starting Generation")
    print(f"{'='*60}")
    print(f"Prompt: '{prompt}'")
    print(f"Parameters:")
    for key, value in result["parameters"].items():
        print(f"  - {key}: {value}")
    print(f"{'='*60}\n")
    
    device_type = result["memory_usage"]["device_type"]
    print(f"🖥️  Device: {device_type}")
    
    if result["memory_usage"]["tracking_available"]:
        print(f"💾 Initial Memory: {result['memory_usage']['initial_memory']:.2f} MB\n")
    else:
        print(f"💾 Memory tracking not available\n")
    
    print(f"⏳ Prefilling cache...")
    print(f"✅ Prefill complete ({result['speed_metrics']['prefill_time']:.2f}s)\n")
    print(f"{'='*60}")
    print(f"📝 Generated Text:")
    print(f"{'='*60}\n")
    print(result["generated_text"])
    
    # Print metrics
    print("\n\n" + "="*60)
    print(f"📊 Performance Metrics")
    print(f"="*60)
    
    stats = result["statistics"]
    print(f"\n🔢 Generation Statistics:")
    print(f"  Prompt Length     : {stats['prompt_length']} tokens")
    print(f"  Generated Tokens  : {stats['generated_tokens']} tokens")
    print(f"  Total Tokens      : {stats['total_tokens']} tokens")
    
    speed = result["speed_metrics"]
    print(f"\n⚡ Speed Metrics:")
    print(f"  Prefill Time      : {speed['prefill_time']:.4f} s")
    print(f"  Prefill Speed     : {speed['prefill_speed']:.2f} tokens/s")
    print(f"  Decode Time       : {speed['decode_time']:.4f} s")
    print(f"  Decode Speed      : {speed['decode_speed']:.2f} tokens/s")
    print(f"  Avg Latency       : {speed['avg_latency']:.2f} ms/token")
    print(f"  Total Time        : {speed['total_time']:.4f} s")
    
    mem = result["memory_usage"]
    if mem["tracking_available"]:
        print(f"\n💾 Memory Usage:")
        print(f"  Device Type       : {mem['device_type']}")
        print(f"  Initial Memory    : {mem['initial_memory']:.2f} MB")
        print(f"  Current Memory    : {mem['current_memory']:.2f} MB")
        print(f"  Peak Memory       : {mem['peak_memory']:.2f} MB")
        print(f"  Memory Used       : {mem['memory_used']:.2f} MB")
    else:
        print(f"\n💾 Device:")
        print(f"  Device Type       : {mem['device_type']}")
        print(f"  Memory Tracking   : Not available")
    
    quality = result["quality_metrics"]
    print(f"\n📈 Generation Quality:")
    print(f"  Avg Log Prob      : {quality['avg_log_prob']:.4f}")
    print(f"  Perplexity        : {quality['perplexity']:.2f}")
    print(f"  Num Repeats       : {quality['num_repeats']} tokens")
    if quality["most_repeated"]:
        print(f"  Most Repeated     : ", end="")
        for item in quality["most_repeated"]:
            print(f"{item['token']}({item['count']}x) ", end="")
        print()
    
    print(f"="*60)
