Project Overview

This project implements a domain-specific NLP pipeline designed to summarize unstructured clinical consultation notes into concise, actionable summaries for physicians.

Unlike generic LLMs, this system utilizes BioBERT (Biomedical Bidirectional Encoder Representations from Transformers) to accurately interpret medical terminology, ensuring that critical information (diagnoses, medications, dosages) is preserved during the summarization process.

Role: Lead Data Scientist / Architect

Status: Production (Internal Deployment)

🏗 System Architecture

The pipeline is designed with strict adherence to data privacy standards. PII (Personally Identifiable Information) is scrubbed before tokenization.

graph TD
    subgraph "Secure Internal Network"
        A[("Internal EHR Database\n(SQL/Patient Records)")] --> B(Data Ingestion Pipeline\nPython/Pandas)
    end

    subgraph "Preprocessing & Privacy"
        B --> C{PII Scrubber\n(Regex + NER)}
        C -->|Anonymized Text| D[Tokenizer\n(BioBERT Vocabulary)]
    end

    subgraph "AI Core (Hugging Face)"
        D --> E[Transformer Model\n(BioBERT / ClinicalBERT)]
        E --> F[Summarization Head\n(Extractive/Abstractive)]
    end

    subgraph "Application Layer"
        F --> G[FastAPI Endpoint\n(REST API)]
        G --> H(("Doctor's Dashboard\n(UI)"))
    end

    style C fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#dfd,stroke:#333,stroke-width:2px


⚙️ Technical Stack

Core NLP: Hugging Face Transformers, PyTorch, Spacy (for NER).

Models: dmis-lab/biobert-v1.1 (Fine-tuned on MIMIC-III dataset).

API layer: FastAPI, Uvicorn.

Data Processing: Pandas, NumPy, regex.

Infrastructure: AWS SageMaker (Training), Docker (Inference).

🚀 Key Features

1. Domain-Specific Tokenization

Generic tokenizers often split medical terms incorrectly (e.g., "Myocardial Infarction" -> "My", "o", "cardial"). We utilized a custom vocabulary adapted from PubMed abstracts to ensure medical terms remain intact as single tokens.

2. Privacy-First Design (HIPAA Compliance)

The PII_Scrubber module utilizes a hybrid approach:

Pattern Matching: Regex for dates, MRNs (Medical Record Numbers), and phone numbers.

NER (Named Entity Recognition): Spacy model to detect names and locations.

Replacement: All identified PII is replaced with tags (e.g., [PATIENT_NAME], [DATE]) before model inference.

3. Latency Optimization

To serve real-time predictions during patient consults, the model was optimized using ONNX Runtime quantization, reducing inference latency from 450ms to <120ms.

📂 Project Structure (Showcase)

├── src/
│   ├── processing/
│   │   ├── ingestion.py       # SQL connectors for EHR
│   │   └── pii_scrubber.py    # Anonymization logic
│   ├── model/
│   │   ├── tokenizer.py       # Custom BioBERT tokenizer
│   │   └── inference.py       # Model prediction logic
│   └── api/
│       └── main.py            # FastAPI routes
├── notebooks/
│   └── fine_tuning_poc.ipynb  # Proof of concept training loop
├── Dockerfile                 # Containerization for SageMaker
├── requirements.txt           # Dependency list
└── README.md                  # Documentation


⚠️ Data Privacy Notice

This repository serves as a Technical Architecture Showcase. Due to the sensitivity of patient data and the proprietary nature of the fine-tuned weights, the source code and datasets are not publicly available. This repository demonstrates the architectural decisions, MLOps practices, and NLP strategies employed in the project.
