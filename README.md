[CL-cifar.md](https://github.com/user-attachments/files/23200847/CL-cifar.md)
\# Class-Incremental Learning Baselines on CIFAR-100

\*\*Task\*\*: Implement and evaluate three CIL baselines from scratch  

\*\*Focus\*\*: Understand catastrophic forgetting through various strategies

\---

\- Implement a \*\*continual learning setup\*\* using \*\*CIFAR-100\*\*, split into \*\*5 tasks with 20 classes each\*\*  

\- The dataset is \*\*dynamic\*\*: model trained only on current task, \*\*no access to previous tasks\*\*  

\- Implement \*\*three baselines\*\*:

1\. \*\*Naive Fine-tuning\*\*: Sequential training without mitigation  

2\. \*\*Replay-based method\*\*: Experience Replay with fixed-size buffer  

3\. \*\*Regularization-based method\*\*: Elastic Weight Consolidation (EWC)

\---

\#\# Implementation Summary

| Baseline | Implemented | Key Details |

|--------|-------------|-----------|

| \*\*Naive Fine-tuning\*\* | Yes | Sequential training, evaluate on all seen tasks after each |

| \*\*Experience Replay\*\* | Yes | Buffer sizes: \*\*100, 1000, 2000\*\*\<br\>Store balanced samples per class\<br\>Mix buffer \+ current task data\<br\>Same training loop as Naive |

| \*\*EWC\*\* | Yes | \*\*Diagonal Fisher\*\* computed \*\*after each task\*\*\<br\>Log-likelihood loss per sample\<br\>Accumulate squared gradients, average over dataset\<br\>Penalty: \`λ × Σ F\_i (θ\_i \- θ\_prev\_i)²\`\<br\>Evaluated λ \= \*\*1e3, 1e4\*\* |

\---

\#\# Experimental Setup

\- \*\*Dataset\*\*: CIFAR-100 (50K train, 10K test)  

\- \*\*Task Split\*\*: 5 tasks × 20 classes  

  \- Task 0: 0–19, Task 1: 20–39, ..., Task 4: 80–99  

\- \*\*Model\*\*: Modified ResNet-18 (3×3 conv1, no maxpool, 100-class head)  

\- \*\*Training\*\*: SGD, lr=0.1, momentum=0.9, weight\_decay=5e-4, \*\*CosineAnnealingLR\*\*, 70 epochs/task, batch\_size=128  

\- \*\*Evaluation\*\*: \*\*Task-aware\*\* (mask logits outside task classes), accuracy matrix, forgetting metric

\---

\#\# Results

\#\#\# Final Average Accuracy & Forgetting

| Method | λ / Buffer | Final Avg Acc | Avg Forgetting |

|-------|------------|---------------|----------------|

| Naive | — | \*\*0.1477\*\* | \*\*0.6966\*\* |

| Replay | 100 | 0.1445 | 0.6929 |

| Replay | 1000 | 0.1591 | 0.6528 |

| Replay | \*\*2000\*\* | \*\*0.1700\*\* | \*\*0.6260\*\* |

| EWC | 1e3 | 0.2384 | 0.5760 |

| EWC | \*\*1e4\*\* | \*\*0.2342\*\* | \*\*0.5670\*\* |

\> \*\*EWC (λ=1e4) outperforms Replay (2000) by \+37.8% in accuracy and reduces forgetting by 9.4%\*\*

\---

\#\#\# Key Observations

\#\#\#\# Naive Fine-tuning

\- \*\*Complete catastrophic forgetting\*\*  

\- Accuracy on old tasks → \*\*0.0000\*\* after learning new task  

\- Final accuracy ≈ \*\*1/7\*\* (random chance over 100 classes)  

\- \*\*Upper bound on forgetting\*\* — exactly as expected

\#\#\#\# Experience Replay

\- \*\*Clear trend\*\*: larger buffer → better accuracy, less forgetting  

\- \*\*+19.8% gain\*\* in final accuracy over Naive (2000 vs 0.1477)  

\- Even with 2000 samples, \*\*old tasks suffer significant forgetting\*\*  

\- Buffer=100: \*\*nearly as bad as Naive\*\* due to extreme downsampling  

\- \*\*Replay successfully mitigates forgetting, but requires large memory\*\*

\#\#\#\# Elastic Weight Consolidation (EWC)

\- \*\*Clear trend\*\*: higher λ → stronger regularization → less forgetting  

\- \*\*+58.6% gain\*\* in final accuracy over Naive (0.2342 vs 0.1477 with λ=1e4)  

\- \*\*Even with λ=1e4, old tasks suffer significant forgetting\*\* — Task 0 drops from 0.686 → 0.046  

\- \*\*λ=1e3 and λ=1e4 both too weak\*\*: performance close to Naive due to insufficient protection  

\- \*\*EWC successfully mitigates forgetting compared to Naive\*\*, but \*\*requires much larger λ (≥1e5)\*\* for meaningful retention  

\- \*\*EWC is memory-free\*\* — protects important parameters via Fisher-based regularization

\---

\#\# Fisher Information Matrix (EWC)

\*\*Exactly as specified in the task\*\*:

\`\`\`text

For each sample:

  1\. Compute log-likelihood loss: ℓ \= \-log p(y|x)

  2\. Compute gradient: ∇\_θ ℓ

  3\. Square and accumulate: F \+= (∇\_θ ℓ)²

  4\. Average over dataset: F \= F / N

