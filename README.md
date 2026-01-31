# 📦 SinOffen — Chinese Offensive Speech Detection Dataset

[![license](https://img.shields.io/badge/license-CC--BY--NC--4.0-green)](https://github.com/raylin281/SinOffen-Chinese-OSD-Dataset)
![language](https://img.shields.io/badge/language-Chinese-red)
![task](https://img.shields.io/badge/task-Offensive_Speech_Detection-blue)

**SinOffen** is a large-scale Chinese offensive speech dataset designed to support research on both **explicit and implicit offensive language**, with a particular emphasis on nuanced and fine-grained implicit strategies common in Chinese internet discourse.

This repository hosts the dataset and utilities for benchmarking and analysis. 
Paper: *SinOffen: A Fine-grained Chinese Dataset for Explicit and Implicit Offensive Speech Detection*.

---

## ⚠️ Data Access

The SinOffen dataset contains offensive and sensitive language.  
To ensure responsible use, the full dataset is released under a controlled access policy.

### Request Access

Please complete the following form to request access to the dataset:

👉 **Google Form:** https://forms.gle/8rPKJF36exeU3nNE9

Approved requests will receive a download link.

### Usage Terms

- Research and educational use only
- No redistribution

---

## 📌 Dataset Overview

- **Language:** Chinese (Standard Mandarin)  
- **Domain:** Online social media text  
- **Tasks Supported:**
  - Offensive Speech Detection (OSD)
  - Binary: Non-OS vs OS
  - Coarse-grained: Non-OS / Explicit OS / Implicit OS
  - Fine-grained Implicit strategy classification (multi-label)
  - Category-level performance analysis

---

## 🧠 Label Taxonomy

### Coarse-grained Labels
- `Non-OS` — Non-offensive content  
- `Explicit OS` — Directly offensive language  
- `Implicit OS` — Indirect or linguistically nuanced offense

### Fine-grained Implicit Labels
Implicit OS may involve multiple co-occurring strategies. Supported categories:

| Code | Strategy | Brief Description |
|------|----------|------------------|
| `circumlocution` | 委婉表达 | Indirect expression to imply offense |
| `homophones` | 谐音 | Offensive via phonetic substitution |
| `metonymy` | 借代 | Uses related terms to signal offense |
| `context` | 语境依赖 | Requires conversation context |
| `metaphor` | 隐喻 | Uses metaphorical structure |
| `irony` | 反讽 | Opposite of literal intent |
| `visual_signs` | 视觉符号 | Symbols/emojis as cues |
| `extra_knowledge` | 外部知识 | World knowledge needed |
| `humiliation` | 羞辱 | Degrading or demeaning hints |
| `black_humor` | 黑色幽默 | Dark humor used offensively |

> ⚠️ A sample may contain **multiple implicit strategies** simultaneously.

---

## 🧾 Data Format

Each sample is stored in CSV format with the following fields:

id: unique sample id  
content: text content  
coarse_label: Non-OS / Explicit OS / Implicit OS  
implicit_labels: list of fine-grained labels (multi-label)  
root_comment_id: root context id  
need_context: whether context is required (0/1)  

---

## 🤖 Pretrained Langugae Models

Several Chinese-adapted pretrained language models (PLMs) suitable for offensive speech detection are available on HuggingFace 👉 https://huggingface.co/Yulinn/SinOffen-CN-PLMs. 
