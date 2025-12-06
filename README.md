# LLM-InternalAlignment-SparseAutoEncoders  
**CS 494 Final Project — Pratyay Banerjee, Agnidipto Sinha**

## Abstract
This project explores how to improve **mathematical reasoning reliability** in Large Language Models (LLMs) by moving from **external correction** (better outputs) to **internal alignment** (better internal computations). We build an end-to-end pipeline that combines:

1. **Contrastive Preference Optimization (CPO)** for targeted reasoning alignment,  
2. **Sparse Autoencoders (SAEs)** to probe the internal representation changes induced by alignment, and  
3. **Hybrid Feature Steering** to test whether the discovered features can be causally manipulated at inference time.

We evaluate on **GSM8K** using **Qwen 2.5-7B-Instruct**, demonstrating that reasoning improvements are **layer-localized**, and that aligned reasoning correlates with a strong **reduction in latent feature activity**, suggesting a more selective and efficient internal reasoning state.

---

## Research Questions
We structured the project around three core questions:

### 1) Where does reasoning alignment most effectively occur in the transformer?
We hypothesize that reasoning is **not uniformly distributed** across layers. The goal is to identify which depth ranges show the greatest improvement when aligned using preference-based optimization.

### 2) What internal changes correspond to improved reasoning?
We hypothesize that improved reasoning corresponds to a **measurable internal signature**. Using SAEs, we aim to identify which latent features systematically change under alignment.

### 3) Can we reproduce some alignment benefit without retraining?
We test whether inference-time activation edits—guided by SAE features—can partially recover aligned behavior.

---

## Method Overview

### Phase 1 — Layer-Specific CPO (Behavioral Alignment)

#### Why CPO?
CPO is a preference-optimization approach designed to push models away from **near-miss incorrect reasoning** and toward **correct reasoning chains**, which is ideal for math where minor errors can derail entire solutions.

#### Triplet Construction
CPO requires `{prompt, chosen, rejected}` triplets. We construct these from GSM8K by:

- Sampling multiple chain-of-thought (CoT) solutions from the base model.
- Extracting final numeric answers via a regex-based parser.
- Marking incorrect CoTs as **rejected**.
- Using the GSM8K ground-truth solutions as **chosen**.

This yields a dataset of ~4.9k reasoning-focused triplets that reflect real model failure modes.

#### Layer-Targeted Training
Instead of fine-tuning the full model, we apply **LoRA + CPO** to specific layer blocks to test reasoning plasticity across depth:

- Early layers
- Early-middle layers
- Middle layers
- Late-middle layers
- Late layers

This design directly tests our **localization hypothesis**.

#### Training Stack
- Quantized training to reduce compute
- PEFT with LoRA
- TRL CPOTrainer
- Controlled evaluation on GSM8K test

---

### Phase 2 — Sparse Autoencoders (Mechanistic Interpretability)

#### Why SAEs?
Transformer representations contain significant **superposition**. SAEs aim to learn **sparse, more interpretable latent features**, making it possible to compare:

- the **base model’s** internal reasoning state
- the **CPO-aligned model’s** internal reasoning state

#### What We Encode
We collect activations from a strategically chosen layer (based on Phase 1 results), focusing on the residual/MLP pathway associated with reasoning behaviors.

#### SAE Architecture
- Overcomplete design with an expanded latent space (8×)
- ReLU encoder, linear decoder
- Reconstruction loss + L1 sparsity
- Hyperparameter sweep for sparsity strength

#### Differential Feature Analysis
We identify:
- **Correction features**: activations that increase post-alignment
- **Error features**: activations that decrease post-alignment

We then quantify how alignment shifts sparse feature usage.

---

### Phase 3 — Hybrid Feature Steering (Inference-Time Control)

#### Goal
We test whether the features identified by the SAE analysis are **causally linked** to reasoning outcomes.

#### Approach
We implement a forward-pass intervention:

1. Intercept activations at the target layer during inference.
2. Encode activations into SAE latent space.
3. **Amplify** top correction features.
4. **Suppress** top error features.
5. Decode back and continue the forward pass.

This is a **hybrid push-pull steering strategy** designed to preserve representational structure while testing causal control.

---

## Results Summary

### 1) Layer localization of reasoning alignment
Our layer-specific CPO experiments show that **middle to late-middle layers** yield the strongest GSM8K gains, supporting the view that these layers are the most effective locus for reasoning alignment.

### 2) Alignment-linked sparsity signature
SAE analysis shows that aligned models exhibit a strong reduction in active latent features and overall activation magnitude. This suggests:

- better reasoning correlates with **more selective internal computation**
- alignment may act like a **semantic pruner**, suppressing noisy or competing reasoning pathways

### 3) Partial inference-time recoverability
Hybrid feature steering does not fully replicate CPO fine-tuning performance, but it provides:

- evidence that at least some identified features are **causally meaningful**
- a proof-of-concept that **activation control** can recover reasoning improvements without updating weights

---

## Why This Project Matters
This project contributes to the broader effort of making LLM reasoning:

- **more reliable**
- **more interpretable**
- and potentially **more controllable at runtime**

Rather than treating alignment as purely behavioral, we connect performance gains to **specific internal representational changes**, offering a step toward mechanistic alignment research.

---

## Tech Stack
- **Model:** Qwen 2.5-7B-Instruct  
- **Benchmark:** GSM8K  
- **Training:** TRL, CPOTrainer  
- **Finetuning:** LoRA, PEFT  
- **Efficiency:** 4-bit quantization (NF4)  
- **Interpretability:** custom SAE training pipeline  
- **Steering:** PyTorch hooks / inference-time activation edits  

---


---

## How to Reproduce (High-Level)

1. **Generate CPO triplets**
   - Sample multiple CoTs from the base model
   - Parse final answers
   - Build `{prompt, chosen, rejected}`

2. **Run layer-specific CPO**
   - Train LoRA adapters per layer block
   - Evaluate and compare GSM8K test accuracy

3. **Collect activations**
   - Use a fixed prompt subset
   - Extract target-layer activations for base and aligned models

4. **Train SAEs**
   - Tune sparsity coefficient
   - Validate reconstruction vs interpretability tradeoff

5. **Compute differential features**
   - Identify correction and error feature sets

6. **Apply hybrid feature steering**
   - Implement inference hooks
   - Evaluate qualitative/quantitative effects

---

## Limitations
- SAE analysis focused on a limited set of layers.
- Steering is static and likely underestimates the role of distributed feature interactions.
- Results are demonstrated on one model family and one reasoning benchmark.

---

## Future Work
- Multi-layer SAE analysis to build a full **alignment atlas**
- Dynamic/conditional steering strategies
- Extending beyond math into:
  - logic
  - coding
  - safety-critical reasoning
- Investigating sparsity as a **reliability signal** in real-time deployments

---

## Authors
- **Pratyay Banerjee**
- **Agnidipto Sinha**

---

## Course
**CS 494 — Generative AI**

---

## License
Add a license of your choice (e.g., MIT) if you plan to make the repository public.

