Overview

Contract documents are long, complex, and time-consuming to analyze manually. This project automates the creation of a Contract Appreciation Document (CAD) from any contract PDF—whether digital or scanned.

The system extracts text, interprets contract clauses, generates structured outputs, and performs compliance and conflict detection, using three progressively improved approaches:

Approach-1: Retrieval-Augmented Generation (RAG)

Approach-2: Sliding-Window JSON Generation

Approach-3: One-Shot ChatGPT (Final & Best Approach)

This repository contains code to extract contract data, generate CAD outputs in JSON/DOCX/PDF, run compliance checks, detect contractual conflicts, and support contract-based Q&A.


📁 Repository Structure
├── app/                        # Streamlit app for UI
├── cad_generation/             # CAD JSON, DOCX, PDF generators
├── extraction/                 # PDF extraction + OCR modules
├── compliance/                 # Rule-based + LLM-based compliance engine
├── conflict_detection/         # Regex and LLM hybrid conflict detector
├── approaches/                 # Approach-1, Approach-2, Approach-3 implementations
├── utils/                      # Helper functions
├── generated_outputs/          # Sample outputs (JSON/DOCX/PDF)
└── README.md                   # This file



Project Goals

Automate generation of fully structured CAD from PDFs

Handle scanned PDFs using OCR and fallback extraction

Maintain page-level traceability

Ensure accuracy, completeness, and consistency

Provide contract-based Q&A, compliance validation, and conflict detection

Output in multiple usable formats: JSON, DOCX, PDF


🧩 System Architecture
🔍 1. PDF Text Extraction

pdfplumber for text-based PDFs

pytesseract OCR for scanned/blank pages

Adds --- PAGE X --- markers for source tracking

Handles mixed-format or partially scanned contracts

🤖 2. LLM Processing (Three Approaches)

Each approach improves the system accuracy and consistency:


Approach-1: RAG (Retrieval-Augmented Generation)

Chunk the contract → generate sentence embeddings

Store embeddings in Chroma vector DB

Retrieve relevant chunks for each CAD section

Generate answers using FLAN-T5

Combine section-wise results into DOCX CAD

Pros:
✔ Good for Q&A
✔ Flexible and modular

Cons:
✘ Lacks global context
✘ CAD output inconsistent across sections


### Approach-2: Sliding Window CAD JSON (Implemented in Code)

Main function:

✔ CAD_JSON_PROMPT

Strict JSON schema covering:

Salient features

Submittals

Notices

Payment

Risks

Claims & arbitration

✔ Process

For each window:

Pass window text into the JSON prompt

Parse output (with fallback raw→cleaner)

Merge using:

merge_json_objects()

✔ Token numbers used

Window = 2048 tokens

Overlap = 256 tokens

Reserved for answers = 512 tokens

✔ Source Mapping

Function:

find_quote_sources()


Searches exact or fuzzy quotes across pages.

### Approach-3: One-Shot ChatGPT (Final System)
Key ideas:

Send entire contract in one single API call

Force strict JSON output using response_format="json"

No merging needed

Most accurate + cleanest CAD structure

Outputs in your code:

JSON → via LLM

DOCX → via python-docx

PDF → via reportlab

Benefits:

Highest accuracy

No context loss

Very consistent structure

Best readability for CAD


📘 5. CAD Generation Module (According to Code)

Main function:

✔ node_generate_cad_json_docx_pdf()
Performs:

Runs JSON generation window-wise

Merges all JSON pieces

DOCX creation:

headings

tables

salient features

PDF creation:

one-page summary

project name

employer

contractor

scope overview (wrapped text)

## 🛡️ 6. Compliance Check (According to Code)

Main LLM prompt:

✔ COMPLIANCE_JSON_PROMPT

Produces for each rule:

{
  "rule": "...",
  "present": true/false,
  "summary": "...",
  "quote": "...",
  "sources": ["page X"],
  "confidence": 0.90
}

Engine:

Splits contract into token windows

Searches per window

Aggregates best results

Main function:

✔ compliance_check_json()
## ⚠️ 7. Conflict Detection Module (5 Implemented Rules)

Implemented functions:

✔ check_commencement_vs_site_possession()

Date mismatch using regex + dateparser

Conflict if possession > commencement

✔ check_payment_term_mismatch()

Regex:

pay within (\d+) days


Flag when different values appear

✔ check_retention_mismatch()

Regex captures retention %

Detects inconsistent values

✔ check_defect_liability_mismatch()

Matches warranty or DLP periods

Converts “years” → months

✔ check_arbitration_vs_court_conflict()

If both "arbitration" and "court" appear → conflict

All conflicts passed through:

✔ map_conflict_to_practical_category()

Adds:

category

resolution_hint

Combined via:
✔ run_conflict_detection()
## 8. Contract Q&A (LangChain)

Function:

✔ ask_direct_langchain()

Works in two modes:

concatenate → full contract

sliding → window-wise Q&A

Prompts used:

CONTRACT_Q_PROMPT

WINDOW_Q_PROMPT

## 🖥️ 9. Streamlit UI Workflow (As per your app.py)

UI features:

Upload PDF

View extracted pages

Chat with LLM

Generate CAD

Download JSON/DOCX/PDF

Run compliance check

Run conflict detection

Debug window display options

Session state keys used:

pages

raw_text

chunks

trans_pipe

tokenizer

llm_wrapper

conversation

cad_json / cad_docx / cad_pdf

## 10. Tech Stack
Extraction

pdfplumber

pytesseract

pdf2image

LLM / Processing

HuggingFace Transformers

LangChain

FLAN-T5-large

Sentencepiece tokenizer

Output

python-docx

reportlab

UI

Streamlit

## 🧾 11. Limitations

OCR noise from scanned pages reduces accuracy

Some conflict rules still require expansion

Very long contracts may exceed token limits

Table extraction is not automated

## 🔮 12. Future Enhancements

Add remaining 11+ conflict rules

Table and figure extraction using layout models

Cross-contract comparison

Risk matrix generation

Fine-tuned LLM for contract domain
