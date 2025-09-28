# 🌍 Neural Machine Translation (NMT) Labs – PyTorch Replications

Welcome to the **Programming Ocean Academy – NMT Labs** repository.  
This repo contains faithful **PyTorch replications of landmark NMT papers**, each structured as an educational lab with:

- 📄 **Paper summary** (abstract, problems, solutions, contributions)  
- 🧩 **PyTorch replication** (training, evaluation, prediction, visualization)  
- 📊 **Analysis** (loss curves, prediction quality, embedding visualizations)  
- 🎓 **Academic insights** (takeaways, mathematical highlights, related works)  

---

## 📚 Included Labs

### 1. [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- Introduced the **Seq2Seq** framework with encoder–decoder LSTMs.  
- Solved phrase-based limitations in SMT.  
- 🚀 PyTorch lab: train & test Seq2Seq on toy datasets.  
- 🔎 Insights: importance of LSTMs for long-range dependencies.  

---

### 2. [Learning Phrase Representations using RNN Encoder–Decoder for SMT (Cho et al., 2014)](https://arxiv.org/abs/1406.1078)
- First **RNN Encoder–Decoder** model for translation.  
- Introduced **gating mechanisms** (foundation for GRU).  
- 🚀 PyTorch lab: replicate RNN encoder–decoder pipeline.  
- 🔎 Insights: paved way for gated recurrent architectures.  

---

### 3. [Long Short-Term Memory Networks for Machine Reading (Hochreiter & Schmidhuber, 1997; applied in NMT)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- Solved **vanishing gradients** with gating (input, forget, output).  
- 🚀 PyTorch lab: implement and train LSTMs in translation tasks.  
- 🔎 Insights: key for stable training of Seq2Seq models.  

---

### 4. [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2015)](https://arxiv.org/abs/1409.0473)
- Introduced **attention mechanism** in NMT.  
- Solved fixed-length bottleneck of encoder.  
- 🚀 PyTorch lab: implement additive (Bahdanau) attention.  
- 🔎 Insights: foundation for modern attention-based architectures.  

---

### 5. [Effective Approaches to Attention-based NMT (Luong et al., 2015)](https://arxiv.org/abs/1508.04025)
- Introduced **global vs local attention** variants.  
- Extended Bahdanau’s attention with multiplicative scoring.  
- 🚀 PyTorch lab: implement Luong attention mechanisms.  
- 🔎 Insights: improved alignment, stronger baselines for NMT.  

---

### 6. [On Using Very Large Target Vocabulary for Neural Machine Translation (Jean et al., 2015)](https://arxiv.org/abs/1412.2007)
- Tackled **large vocabulary problem** with sampled softmax.  
- 🚀 PyTorch lab: efficient training with large vocabularies.  
- 🔎 Insights: improved scalability & robustness in NMT.  

---

### 7. [Google Neural Machine Translation (GNMT) (Wu et al., 2016)](https://arxiv.org/abs/1609.08144)
- Introduced **GNMT system** (8-layer LSTMs + attention).  
- Achieved **human-level quality** in many benchmarks.  
- 🚀 PyTorch lab: simplified GNMT replication.  
- 🔎 Insights: industrial-scale deployment lessons.  

---

### 8. [Google’s Multilingual NMT: Enabling Zero-Shot Translation (Johnson et al., 2016)](https://arxiv.org/abs/1611.04558)
- One shared model for **many-to-many translations**.  
- Introduced **target forcing tokens** (`<2es>`, `<2fr>`, etc).  
- Enabled **zero-shot translation** (unseen pairs).  
- 🚀 PyTorch lab: toy multilingual model with embedding visualization.  
- 🔎 Insights: evidence of **emergent interlingua representations**.  

---

### 9. [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- Introduced the **Transformer architecture**.  
- Removed recurrence, relying entirely on **self-attention**.  
- 🚀 PyTorch lab: implement mini-Transformer for NMT.  
- 🔎 Insights: foundation for modern LLMs and GPT-family.  

---

## 🔬 Structure of Each Lab

Each `.ipynb` notebook includes:
1. **Abstract & Problem Statement**  
2. **Proposed Solution & Methodology**  
3. **Mathematical & Statistical Highlights**  
4. **PyTorch Implementation (Training, Evaluation, Prediction)**  
5. **Analysis (loss curves, predictions, embedding visualizations)**  
6. **Academic Takeaways & Related Work**

---

## 🎯 Purpose

- Recreate the **evolution of NMT** (2014 → 2017 → beyond).  
- Provide **hands-on PyTorch implementations** for learners.  
- Serve as **educational companion labs** for AI/ML students, researchers, and practitioners.  

---

## 🚀 How to Use

1. Clone the repo:
   ```bash
   git clone https://github.com/Programming-Ocean-Academy/deep-learning.git
   cd deep-learning/neural-machine-translation

