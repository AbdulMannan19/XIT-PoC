import requests

def evaluate_tps(model_name, prompt):
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
    
    models = [
        ("llama3.1:8b-instruct-fp16", "FP16 (full precision)"),
        ("llama3.1:8b-instruct-q8_0", "Q8_0 (8-bit)"),
        ("llama3.1:8b-instruct-q4_K_M", "Q4_K_M (4-bit improved)"),
        ("llama3.1:8b", "Q4_0 (4-bit default)")
    ]
    
    print("Starting TPS benchmark...\n")
    
    results = []
    for model_name, label in models:
        print(f"Evaluating {label}...")
        tps = evaluate_tps(model_name, prompt)
        results.append((label, tps))
        print(f"Model: {label} | TPS: {tps:.2f}\n")
    
    print("\nComparison (relative to FP16):")
    fp16_tps = results[0][1]
    for label, tps in results[1:]:
        if fp16_tps > 0:
            speedup = (tps / fp16_tps) * 100
            print(f"{label} is {speedup:.1f}% the speed of FP16")

if __name__ == "__main__":
    main()
