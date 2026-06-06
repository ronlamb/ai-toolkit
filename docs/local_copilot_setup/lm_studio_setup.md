# Accuracy‑First Optimization Guide  
### LM Studio + MLX + Qwen3 Coder Next (MacBook Pro M5 Max 128GB)

This guide focuses on **maximum accuracy**, **minimal hallucinations**, and **stable long‑context behavior** when running **Qwen3 Coder Next** locally using **LM Studio with the MLX backend**.

It is based on real measurements from an **M5 Max 128GB** system and extensive testing of MLX quantization behavior.

---

# 1. Why Accuracy Drops With Lower Quantization

Qwen3 Coder Next is a **Mixture‑of‑Experts (MoE)** model.  
MoE models are **more sensitive** to quantization than dense models because:

- Routing logits must remain precise  
- Attention layers are deep and high‑variance  
- Coding heads rely on high‑frequency patterns  
- Long‑context rotary embeddings amplify quantization error  

As a result:

### Safe quantizations for coding accuracy:
- **Q8_K**  
- **Q6_K**

### Risky quantizations for coding:
- **Q5_K_M** (acceptable but degraded)  
- **Q4_K_M / Q4_K_S** (noticeable drift)  
- **Q3_K_XL** (significant hallucinations)

---

# 2. KV Cache Quantization and Hallucinations

Quantizing the **KV cache** affects **attention**, not weights.

Attention is where hallucinations originate.

### KV quantization effects:
- **8‑bit KV** → stable, accurate, recommended  
- **4‑bit KV** → known to cause hallucinations, logic drift, and code errors  
- **2‑bit KV** → not recommended for any coding model  

For MoE models like Qwen3 Coder Next, **4‑bit KV is especially risky**.

---

# 3. Accuracy‑First Recommended Settings

These settings prioritize correctness over maximum context length.

### Model Settings
- **Quantization:** `Q6_K`  
- **KV Cache Quantization:** `8-bit`  
- **GPU Layers:** `Auto`  
- **GPU KV Cache:** **ON**  
- **Backend:** MLX  

### Context Settings
- **Max Context Length:** `65536`  
- **Max Output Tokens:** `8000–12000`  

### Advanced Settings
- **Batch Size:** `1`  
- **FlashAttention:** ON (if available)  
- **Speculative Decoding:** OFF  
- **MoE Optimizations:** ON  

### Why this is the best accuracy profile
- Q6_K preserves routing logits and coding heads  
- 8‑bit KV keeps attention stable  
- 65k context avoids the 75k–90k slowdown zone  
- GPU KV offload keeps attention fast and precise  

This is the **most accurate configuration** you can run locally on MLX.

---

# 4. If You Need Longer Context (100k–150k)

To extend context without destroying accuracy:

### Use:
- **Quantization:** `Q5_K_M`  
- **KV Cache:** `8-bit`  
- **Context:** `100k–120k`  

### Notes:
- Slight accuracy drop vs Q6  
- Still safe for coding  
- Much better than Q4/Q3  

This is the best compromise between **accuracy** and **context length**.

---

# 5. Quantizations to Avoid for Coding

These combinations are known to cause hallucinations, logic drift, or unstable long‑context behavior:

### ❌ Q4_K_M + 4‑bit KV  
### ❌ Q4_K_S + 4‑bit KV  
### ❌ Q3_K_XL (any KV)  
### ❌ Q4_K_M with context > 100k  
### ❌ 4‑bit KV on MoE models  

These are fine for chat or summarization, but **not for code**.

---

# 6. Memory Usage Reference (Real Measurements)

At **89,000 tokens** with Q6_K + 8‑bit KV:

- **Total memory used:** ~91 GB  
- **node (LM Studio server):** ~74 GB  
- **Code Helper:** ~2 GB  

This corresponds to:

- **KV cache:** ~70–75 GB  
- **Model + buffers:** ~15–20 GB  

This confirms:

### Stable max context with Q6_K + 8‑bit KV is ~65k–75k tokens.

Above that, MLX hits memory‑bandwidth saturation and Copilot times out.

---

# 7. Accuracy‑Focused Profiles Summary

| Profile | Quant | KV | Context | Accuracy | Notes |
|--------|--------|------|-----------|-----------|--------|
| **Accuracy‑First (Recommended)** | Q6_K | 8‑bit | **65k** | ⭐⭐⭐⭐⭐ | Best coding reliability |
| **Long‑Context (Accurate)** | Q5_K_M | 8‑bit | **100k–120k** | ⭐⭐⭐⭐ | Good balance |
| **High‑Context (Risky)** | Q4_K_M | 8‑bit | **120k–150k** | ⭐⭐⭐ | Drift increases |
| **Not Recommended** | Q4_K_M / Q3_K_XL | 4‑bit | Any | ⭐ | Hallucinations likely |

---

# 8. Final Recommendation

For the **best accuracy and coding reliability** on your **M5 Max 128GB**, use:

