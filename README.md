

#### Contract CAD Generator – Automated Contract Analysis using LLMs

This project automates the generation of **Contract Appreciation Documents (CAD)** from construction contract PDFs (200–300+ pages).
Traditionally, preparing a CAD manually takes hours.
This system produces it **automatically**, with:

* ✔ Structured CAD (JSON + DOCX + PDF)
* ✔ Clause-level compliance check
* ✔ Contract conflict detection
* ✔ Contract-based Q&A
* ✔ Support for both **digital** and **scanned** PDFs (OCR included)

The project evolves through **three approaches**, with the final one (ChatGPT One-Shot) delivering the most accurate, stable, and production-ready results.

---

###  How to Run

```bash
pip install -r requirements.txt
pip install openai
streamlit run Final_code.py
```

---

# 📌 Project Overview

Contract documents are long, repetitive, and difficult to understand.
A Contract Appreciation Document (CAD) is a highly-important 20–25 page summary used in construction project management.

This system automates:

* Extracting text from PDF (OCR + selectable text)
* Generating a structured CAD using LLMs
* Detecting conflicts (dates, payments, retention, LDs, etc.)
* Running compliance checks (does the contract contain a specific clause?)
* Asking direct Q&A questions over the contract

Outputs are generated in:

* **JSON**
* **DOCX**
* **PDF**

All with **page-level traceability**.

---

### System Architecture

## 1️⃣ PDF Text Extraction

* **pdfplumber** for normal PDFs
* **pytesseract OCR** for scanned pages
* **pdf2image** for converting pages to images
* Each page is annotated with

  ```
  --- PAGE X ---
  ```

  to preserve traceability.

Handles:

* Mixed scanned + digital PDFs
* Missing text layers
* Irregular formatting

---

## 2️⃣ LLM Processing — Three Approaches

The system was developed in three progressive, improving approaches:

---

### **Approach-1: RAG (Retrieval-Augmented Generation)**

**Pipeline**

* Chunk contract → build embeddings
* Store in **ChromaDB**
* Retrieve relevant chunks
* Generate CAD section-wise using **FLAN-T5**
* Combine into final CAD

**Pros**

✔ Good for Q&A
✔ Modular architecture

**Cons**

✘ CAD becomes inconsistent across sections
✘ No global context

---

### **Approach-2: Sliding-Window JSON Generation**

Sliding window across full contract (e.g., 2048 tokens per window).

Each window is passed through a **strict JSON CAD schema** containing:

* Salient features
* Submittals
* Notice clauses
* Payment
* Risks
* Claims & arbitration
* And more…

The system:

* Merges window outputs
* Repairs malformed JSON
* Locates sources using fuzzy text search

**Pros**

✔ More structured
✔ Page-level traceability

**Cons**

✘ Still window-based
✘ Requires complex merging logic

---

### **Approach-3 (Final): One-Shot ChatGPT**

This is the **best and final approach**, used in your code.

Key features:

* Entire contract passed in **one single LLM call**
* Uses `response_format={"type": "json_object"}` → **strict JSON output**
* Automatic repair if JSON breaks
* Directly generates:

  * JSON CAD
  * DOCX (via python-docx)
  * PDF (via reportlab)
* No windowing or merging required

**Benefits**

✔ Highest accuracy
✔ Most complete
✔ Most consistent CAD
✔ Perfect for long legal documents

---

## 3️⃣ CAD Generation

Main function:

```
node_generate_cad_json_docx_pdf()
```

Creates:

* Full CAD (JSON)
* Fully formatted DOCX:

  * Headings
  * Tables (salient features, submittals, notices, payment terms)
* One-page PDF summary:

  * Key contract details
  * Most important clauses

---

## 4️⃣ Compliance Check

Each rule returns:

```json
{
  "rule": "...",
  "present": true,
  "summary": "...",
  "quote": "...",
  "sources": ["page X"],
  "confidence": 0.91
}
```

Used for:

* Checking if a contract contains mandatory clauses
* Ensuring completeness
* Legal compliance review

---

## 5️⃣ Conflict Detection (5 Implemented Rules)

Functions implemented:

* **Commencement vs Site Possession mismatch**
* **Payment term mismatch**
* **Retention % mismatch**
* **Defect liability / warranty mismatch**
* **Arbitration vs court conflict**

Each conflict is mapped to:

* category
* resolution_hint
* evidence with page numbers

---

## 6️⃣ Contract Q&A

Two modes:

* Full-contract Q&A
* Sliding-window fallback

Allows users to ask questions like:

> *“What is the payment schedule?”*
> *“Is there an arbitration clause?”*

---

## 7️⃣ Streamlit UI

Features:

* Upload PDF
* View extracted pages
* Ask contract Q&A
* Generate CAD
* Download JSON / DOCX / PDF
* Run compliance checks
* Run conflict detection
* Debug views

---

# 🛠 Tech Stack

### Extraction

* pdfplumber
* pdf2image
* pytesseract

### LLM / Processing

* OpenAI GPT-4.1 / GPT-5.1
* HuggingFace (FLAN-T5 for early approaches)
* LangChain

### Output

* python-docx
* reportlab

### UI

* Streamlit

---

### Limitations (Realistic & Updated)

* OCR introduces noise in low-quality scanned PDFs
* Very long contracts may still hit token limits
* Some edge-case conflicts not yet implemented
* Table/figure extraction is not automated
* Handling completely irregular formatting is difficult
* **Privacy concern:** legal contracts are confidential

  * **Solution:** use *OpenAI Enterprise* / self-hosted embeddings for private deployment

---

### Future Enhancements

* Add 10+ more conflict detection rules
* Improve table/figure extraction using layout models
* Automatic clause linking and cross-referencing
* Multilingual support for global contracts
* Fully interactive UI with summary visualizations
* Fine-tuned LLM optimized for contract domain


