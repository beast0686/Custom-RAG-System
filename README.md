# ⚛️ QRAG & Advanced Classical Retrieval Architectures

## 📌 Project Overview

This repository houses a bifurcated experimental framework for evaluating and resolving structural limitations in Retrieval-Augmented Generation (RAG).

The repository consists of two entirely decoupled pipelines:

1. **The Classical RAG Benchmarking Suite**  
   An applied engineering framework evaluating multi-modal retrieval strategies across standardized linguistic metrics.

2. **The QRAG Research Module**  
   A theoretical and experimental quantum pipeline designed to resolve structural linguistic ambiguity using quantum tensor networks and compositional semantics.

---

# 🏛️ Part 1 — Classical RAG Benchmarking Suite

The classical component evaluates the performance boundaries of modern heuristic and agentic retrieval systems.

The framework utilizes a State-of-the-Art (SOTA) cross-encoder reranker:

- **BGE-Reranker-v2.5**

and benchmarks generation quality across three distinct architectural baselines.

---

## 🔍 Evaluated Architectures

### Baseline 0 — Plain LLM
Zero-shot generative baseline without retrieval intervention.

### Baseline 1 — Mongo Vector RAG
Dense vector retrieval operating in linear latent semantic space.

### Baseline 2 — Dynamic KG RAG
Knowledge Graph routing architecture with multi-agentic graph extraction and traversal.

---

## 📊 Evaluation Metrics

Performance is strictly measured using the **RAGAS (Retrieval Augmented Generation Assessment)** framework.

The evaluation specifically targets the **"Linearity Trap"** inherent in overlapping syntactic structures.

### Context Relevance
Measures retrieval precision and identifies lexical overfitting behavior in dense retrievers.

### Answer Faithfulness
Evaluates hallucination resistance and adherence to retrieved context.

### Answer Relevance
Measures direct applicability and alignment of the generated answer to the user query.

---

## ⚠️ Structural Failure Modes Investigated

The benchmark suite focuses on pathological linguistic phenomena that degrade classical retrieval systems:

- Garden-path sentence ambiguity
- Lexical echo traps
- Structural attachment collapse
- Semantic eclipses
- Dense-vector locality bias
- Multi-interpretation syntactic overlap

---

# 🔬 Part 2 — QRAG Research Module

## Quantum Retrieval-Augmented Generation (QRAG)

The QRAG module is a theoretical and experimental quantum retrieval architecture designed to overcome structural disambiguation failures in classical RAG systems.

The framework demonstrates that classical latent-space retrieval fundamentally struggles with overlapping syntactic structures due to local attachment bias and semantic linearization constraints.

QRAG addresses this limitation by mapping language structures into a high-dimensional complex Hilbert space using the **DisCoCat (Distributional Compositional Categorical)** framework.

---

# ⚙️ Core Quantum Mechanisms

## 🧠 Semantic Encoding

Natural language semantics are encoded as parameterized single-qubit rotations:

```math
Ry(\theta)
```

Each token contributes to the quantum semantic manifold through learned angular representations.

---

## 🔗 Topological Entanglement

Syntactic dependencies are preserved using two-qubit controlled entangling operations:

```math
CZ
```

This prevents structural conflation between overlapping linguistic interpretations.

---

## 🌲 DAG Filtering

Directed acyclic graph (DAG) pruning dynamically removes non-structural tokens, allowing scalable parameterized quantum circuit (PQC) construction.

### Constraint

```math
N \le 7
```

logical qubits per query representation.

---

# 💻 Hardware & Systems Engineering

All experimental telemetry is extracted directly from utility-scale quantum processors.

---

## 🖥️ Target Hardware

- **IBM Quantum ibm_fez**
- 156-qubit heavy-hex lattice architecture

---

## ⚡ Disjoint Sub-Topology Mapping

Circuit compilation is dynamically routed onto isolated 7-qubit sub-topologies separated by 1-qubit physical buffers.

This enables:

- Parallel query execution
- Crosstalk minimization
- Reduced coherent interference

### Parallelization Characteristics

- Up to **5 concurrent query executions**
- **4.88× throughput acceleration**
- Reduced amortized NISQ tax

### Runtime Performance

```math
26.62 \text{ seconds/query}
```

---

## 🧪 Quantum Error Mitigation (QEM)

The pipeline integrates:

### TREX — Twirled Readout Error Extinction

TREX is applied at the API level to purify:

- SPAM noise
- State preparation instability
- Readout corruption

---

# 📈 Empirical Research Results

Non-parametric statistical validation was performed over a highly adversarial:

```math
N = 150
```

challenge dataset.

---

## 🏆 Top-1 Parsing Accuracy

| Architecture | Accuracy |
|---|---|
| QRAG | **77.33%** |
| Agentic RAG | 54.00% |

---

## 📊 Statistical Significance

### Wilcoxon Signed-Rank Test

```math
p < 0.0001
```

---

## 📐 Effect Size

### Cohen's h

```math
h = 0.498
```

### Paired Odds Ratio

```math
2.67
```

---

# 🧪 Experimental Objectives

This repository investigates whether:

- Structural ambiguity can be resolved through topological entanglement
- Classical latent retrieval inherently collapses under overlapping syntax
- Quantum compositional semantics provide measurable retrieval advantages
- Tensor-network linguistic representations outperform heuristic retrieval routing

---

# 📚 Research Domains

This work intersects multiple advanced domains:

- Retrieval-Augmented Generation (RAG)
- Knowledge Graph Systems
- Quantum Natural Language Processing (QNLP)
- Tensor Networks
- Quantum Information Theory
- DisCoCat Semantics
- Quantum Error Mitigation
- Statistical NLP Evaluation

---

# 🚀 Key Contributions

- Multi-baseline adversarial RAG benchmarking framework
- Dynamic Knowledge Graph retrieval architecture
- Quantum tensor-network retrieval formalism
- Hilbert-space semantic encoding pipeline
- TREX-integrated quantum execution layer
- Empirical validation using non-parametric statistical testing
- Parallelized heavy-hex sub-topology execution strategy

---

# ⚠️ Research Disclaimer

The QRAG module is an experimental research framework intended for theoretical exploration and empirical analysis within quantum-enhanced NLP systems.

The quantum pipeline is not positioned as a production retrieval system, but as a research investigation into the structural limitations of classical retrieval architectures.

---

# 👨‍🔬 Authors & Contributors

- Anirudh R
- Aman Fayazahmed Soudagar
- We would like to thank the folks over at MIT, MAHE and BNMIT who made this possible.

---

# 📜 License

This project is released under the MIT License.

```text
MIT License © 2026
```
