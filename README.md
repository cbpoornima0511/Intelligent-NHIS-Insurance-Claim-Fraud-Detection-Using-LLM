<div align="center">

# 🏥 Intelligent NHIS Insurance Claim Fraud Detection Using LLM

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LLM](https://img.shields.io/badge/LLM-Llama_3.1_8B-FF6F00?style=for-the-badge&logo=meta&logoColor=white)](https://huggingface.co/meta-llama)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**An end-to-end intelligent fraud detection system that combines an XGBoost classifier with Meta's Llama 3.1 8B LLM to detect fraudulent NHIS healthcare insurance claims — delivering predictions, natural language explanations, investigation reports, and decision recommendations through an interactive Streamlit dashboard.**

[📌 Overview](#-project-overview) • [🏗️ Architecture](#️-system-architecture) • [📊 Dataset](#-dataset) • [🚀 Quick Start](#-quick-start) • [📈 Results](#-model-performance) • [🖥️ Demo](#️-app-walkthrough)

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [System Architecture](#️-system-architecture)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [App Walkthrough](#️-app-walkthrough)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 📌 Project Overview

Healthcare insurance fraud costs health systems billions annually and degrades care for genuine patients. This project builds a **production-grade fraud detection pipeline** for the **National Health Insurance Scheme (NHIS)** using a two-layer approach:

1. **XGBoost Classifier** — trained on 20,388 real NHIS claims to predict fraud type from structured claim data
2. **Llama 3.1 8B Instruct (via HuggingFace Router)** — generates human-readable explanations, investigation reports, and actionable decisions for every prediction

The result is a fully interactive **Streamlit web app** where a fraud analyst enters claim details and receives not just a classification, but a complete AI-powered narrative explaining *why* the claim was flagged — making the system genuinely useful for non-technical investigators.

---

## ❗ Problem Statement

> *Traditional NHIS fraud detection relies on manual audits and rigid rule-based checks — slow, expensive, and unable to keep pace with evolving fraud schemes like Phantom Billing, Ghost Enrollees, and deliberate Wrong Diagnosis coding.*

**This project addresses that gap** by training a robust ML classifier on real NHIS claim patterns and augmenting it with an LLM that reasons contextually, produces investigation-grade narratives, and recommends concrete next steps — bridging the gap between a model prediction and an investigator's decision.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Claim Input (Streamlit UI)                  │
│   Patient ID · Age · Gender · Dates · Amount · Diagnosis     │
└─────────────────────────┬────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │    sklearn Pipeline     │
              │                         │
              │  ColumnTransformer      │
              │  ┌──────────────────┐   │
              │  │ Numeric Features │   │
              │  │ SimpleImputer    │   │
              │  │ + StandardScaler │   │
              │  └──────────────────┘   │
              │  ┌──────────────────┐   │
              │  │ Object Features  │   │
              │  │ SimpleImputer    │   │
              │  │ + OrdinalEncoder │   │
              │  └──────────────────┘   │
              │         ↓               │
              │   XGBClassifier         │
              │   (class_weight=        │
              │    "balanced")          │
              └────────────┬────────────┘
                           │
          ┌────────────────▼────────────────┐
          │      Prediction Output          │
          │  Fraud Type + Confidence Score  │
          │  + Per-class Probabilities      │
          └────────────────┬────────────────┘
                           │
              ┌────────────▼────────────┐
              │   LLM Layer             │
              │   Llama 3.1 8B Instruct │
              │   HuggingFace Router    │
              │   /v1/chat/completions  │
              │                         │
              │  Tab 1: AI Explanation  │
              │  Tab 2: Interactive Q&A │
              │  Tab 3: Investigation   │
              │         Report          │
              │  Tab 4: Decision Rec.   │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Streamlit Dashboard   │
              │   4-Tab Interactive UI  │
              └─────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **LLM-Powered Explanations** | Llama 3.1 8B explains every prediction in plain English for fraud investigators |
| 💬 **Interactive Q&A** | Ask any question about a flagged claim and receive a contextual AI answer |
| 📝 **Auto Investigation Report** | One-click generation of a structured 5-section report with risk factors and recommendations |
| 🎯 **Decision Recommendations** | AI-generated actionable plan: approve / escalate / verify — with priority level and timeline |
| 📊 **Probability Dashboard** | Animated confidence bars for all 4 fraud types per claim |
| ⚙️ **End-to-End ML Pipeline** | sklearn Pipeline with imputation, encoding, scaling, and XGBoost serialised to `model.pkl` |
| 🎨 **Colour-Coded Animated UI** | CSS-animated prediction cards uniquely styled per fraud type |

---

## 📊 Dataset

**File:** `NHIS Healthcare_claim_fraud.csv`

| Attribute | Value |
|---|---|
| **Total Records** | 20,388 insurance claims |
| **Input Features** | 7 |
| **Target Column** | `FRAUD_TYPE` (4 classes) |

### Features

| Column | Type | Description |
|---|---|---|
| `Patient ID` | int | Unique patient identifier |
| `AGE` | float | Patient age |
| `GENDER` | object | M / F |
| `DATE OF ENCOUNTER` | object | Date of hospital visit (YYYY-MM-DD) |
| `DATE OF DISCHARGE` | object | Date of discharge (YYYY-MM-DD) |
| `Amount Billed` | float | Claim amount |
| `DIAGNOSIS` | object | Medical diagnosis description |

### Target Class Distribution

```
No Fraud          ██████████████████████████  11,704  (57.4%)
Phantom Billing   ████████░░░░░░░░░░░░░░░░░░   4,233  (20.8%)
Ghost Enrollee    ███████░░░░░░░░░░░░░░░░░░░░   4,099  (20.1%)
Wrong Diagnosis   █░░░░░░░░░░░░░░░░░░░░░░░░░     352   (1.7%)
Total                                          20,388
```

### Fraud Type Definitions

| Fraud Type | Description |
|---|---|
| **Phantom Billing** | Charges submitted for services or procedures never actually rendered |
| **Ghost Enrollee** | Claims filed for deceased, non-existent, or ineligible patients |
| **Wrong Diagnosis** | Deliberate misclassification of medical conditions to justify higher billing |
| **No Fraud** | Legitimate claim with no fraudulent activity detected |

---

## 🛠️ Tech Stack

### Machine Learning
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-189AB4?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

### LLM
![HuggingFace](https://img.shields.io/badge/HuggingFace_Router-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Llama](https://img.shields.io/badge/Meta_Llama_3.1_8B_Instruct-0467DF?style=flat-square&logo=meta&logoColor=white)

### App & Deployment
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pickle](https://img.shields.io/badge/Pickle-Model_Serialization-4A4A4A?style=flat-square)

---

## 📂 Project Structure

```
Intelligent-NHIS-Insurance-Claim-Fraud-Detection-Using-LLM/
│
├── 📄 NHIS Healthcare_claim_fraud.csv   # 20,388 NHIS claims (raw dataset)
├── 🧠 model.py                          # Training script: preprocessing + XGBoost pipeline
├── 🥒 model.pkl                         # Serialised sklearn Pipeline (ready to serve)
├── 🖥️  app.py                            # Streamlit 4-tab web application + LLM integration
└── 📖 README.md
```

---

## 🚀 Quick Start

### Prerequisites

```
Python >= 3.9
HuggingFace account with API token  →  huggingface.co/settings/tokens
```

### 1. Clone the Repository

```bash
git clone https://github.com/cbpoornima0511/Intelligent-NHIS-Insurance-Claim-Fraud-Detection-Using-LLM.git
cd Intelligent-NHIS-Insurance-Claim-Fraud-Detection-Using-LLM
```

### 2. Install Dependencies

```bash
pip install streamlit pandas scikit-learn xgboost requests
```

### 3. Set Your HuggingFace Token

```bash
# Linux / macOS
export HF_TOKEN="hf_your_token_here"

# Windows (PowerShell)
$env:HF_TOKEN="hf_your_token_here"
```

### 4. (Optional) Retrain the Model

```bash
# Only needed if you modify the dataset or want a fresh model
python model.py
# ✅ Model saved successfully as model.pkl
```

### 5. Launch the App

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔬 Methodology

### 1. Data Loading & Splitting
The 20,388-record NHIS CSV is loaded with Pandas. `FRAUD_TYPE` is the target; the remaining 7 columns are features. Data is split 80/20 (train/test) with `random_state=42` for reproducibility.

### 2. Preprocessing Pipeline
A `ColumnTransformer` embedded inside a sklearn `Pipeline` handles both feature types:

```
Numeric  (AGE, Amount Billed)    →  SimpleImputer(mean)     →  StandardScaler
Categorical (GENDER, DIAGNOSIS,  →  SimpleImputer(constant) →  OrdinalEncoder
             dates)                  fill_value="unknown"       handle_unknown=-1
```

This ensures the model is robust to missing values and unseen diagnosis codes at inference time — with zero data leakage.

### 3. Label Encoding
`LabelEncoder` maps the 4 fraud type strings to integer indices for XGBoost. The encoder is preserved to reverse-map predictions to readable labels in the app.

### 4. XGBoost Classification
`XGBClassifier(class_weight="balanced")` directly addresses the class imbalance (Wrong Diagnosis = 1.7%). The complete Pipeline is serialised to `model.pkl` via Pickle — enabling single-file deployment.

### 5. LLM Integration (Llama 3.1 8B)
For every prediction, a carefully crafted prompt is built with claim context + ML output, then posted to the **HuggingFace Router** (`meta-llama/Llama-3.1-8B-Instruct:novita` via `/v1/chat/completions`). Four distinct prompt templates power the four app tabs:

- **Explanation prompt** — plain-English reasoning for the fraud classification
- **Q&A prompt** — grounded answers to investigator questions
- **Report prompt** — 5-section structured investigation report
- **Recommendation prompt** — prioritised action plan with department routing

### 6. Streamlit App
Four tabs surface all outputs with animated CSS cards colour-coded per fraud type, animated probability bars, and a live system stats footer.

---

## 📈 Model Performance

Trained on 20,388 NHIS claims with 80/20 split. Key design decisions that drive performance:

| Design Decision | Benefit |
|---|---|
| `class_weight="balanced"` in XGBoost | Prevents the 1.7% Wrong Diagnosis class from being ignored |
| `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` | Handles unseen diagnosis codes at inference without crashing |
| `SimpleImputer` in Pipeline | Gracefully handles missing values in real-world claim data |
| Full sklearn Pipeline → `model.pkl` | No preprocessing drift between training and serving |

**Fraud Type Breakdown (Test Set)**

```
No Fraud          → Strong baseline (57.4% of data)
Phantom Billing   → Well-represented (20.8%)
Ghost Enrollee    → Well-represented (20.1%)
Wrong Diagnosis   → Handled via class_weight (1.7%)
```

---

## 🖥️ App Walkthrough

### Tab 1 — 📊 Make Prediction
Enter claim details (Patient ID, Age, Gender, Encounter & Discharge dates, Amount Billed, Diagnosis) and click **Analyze Claim**:
- Colour-coded prediction card: 🟢 No Fraud / 🔴 Phantom Billing / ⚠️ Wrong Diagnosis / 👻 Ghost Enrollee
- Confidence score for the predicted class
- Animated probability bars for all 4 fraud types
- AI explanation from Llama 3.1 8B in plain English

### Tab 2 — 💬 Interactive Q&A
Ask the LLM questions about the most recent claim:
- *"Why was this flagged as Phantom Billing?"*
- *"What documents should I request to verify this?"*
- *"How serious is a 73% confidence score?"*

### Tab 3 — 📝 Investigation Report
One click auto-generates a structured report:
1. Executive Summary
2. Fraud Classification Details
3. Risk Factors Identified
4. Suspicious Patterns
5. Recommendations for Investigation Team

### Tab 4 — 🎯 Decision Recommendation
AI-generated action plan including:
- Immediate action (approve / escalate / verify)
- Priority level (High / Medium / Low)
- Specific verification documents required
- Suggested department assignment and timeline

---

## 🔮 Future Work

- [ ] **SHAP Explainability** — Add SHAP waterfall plots to show which features drove each ML prediction
- [ ] **Batch Upload** — Allow CSV upload for bulk claim screening with a downloadable flagged-claims report
- [ ] **SMOTE Oversampling** — Further address the Wrong Diagnosis minority class (1.7%)
- [ ] **Fine-tuned LLM** — Fine-tune Llama on NHIS clinical terminology for sharper, domain-specific reasoning
- [ ] **Docker Deployment** — Containerise the app for one-command cloud deployment (AWS / GCP / Azure)
- [ ] **Audit Trail & Logging** — Persist every prediction and LLM response for compliance and model monitoring
- [ ] **Authentication** — Role-based access control for investigators vs. admins

---

## 🤝 Contributing

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feature/your-feature

# 3. Commit your changes
git commit -m "Add: your feature description"

# 4. Push and open a Pull Request
git push origin feature/your-feature
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Poornima C B**
*Data Science & AI/ML Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-cbpoornima0511-181717?style=flat-square&logo=github)](https://github.com/cbpoornima0511)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

*Built to make NHIS fraud detection smarter, faster, and explainable — one claim at a time.*

</div>
