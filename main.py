import requests

def evaluate_tps(model_name, prompt):
    """
    Evaluate tokens per second for a given model and prompt.
    
    Args:
        model_name: The Ollama model identifier
        prompt: The input prompt to generate from
        
    Returns:
        float: Tokens per second during generation phase
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    response = requests.post(url, json=payload).json()
    
    eval_count = response.get("eval_count", 0)
    eval_duration_ns = response.get("eval_duration", 1)
    eval_duration_s = eval_duration_ns / 1e9
    
    tps = eval_count / eval_duration_s if eval_duration_s > 0 else 0
    
    return tps

def main():
    prompt = "Explain the architectural differences between monolithic and microservice application designs in deep technical detail."
    
    print("Starting TPS benchmark...\n")
    
    # Evaluate FP16 model
    print("Evaluating FP16 model...")
    tps_fp16 = evaluate_tps("llama3.1:8b-instruct-fp16", prompt)
    print(f"Model: FP16 | TPS: {tps_fp16:.2f}\n")
    
    # Evaluate Q4_K_M model
    print("Evaluating Q4_K_M model...")
    tps_q4 = evaluate_tps("llama3.1:8b-instruct-q4_K_M", prompt)
    print(f"Model: Q4_K_M | TPS: {tps_q4:.2f}\n")
    
    # Calculate speedup
    if tps_fp16 > 0:
        speedup = (tps_q4 / tps_fp16) * 100
        print(f"Q4_K_M is {speedup:.1f}% the speed of FP16")

if __name__ == "__main__":
    main()
