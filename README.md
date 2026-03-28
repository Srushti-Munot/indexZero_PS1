# 🌿GREEN LENS
Our system uses a hybrid pipeline combining rule-based heuristics and zero-shot NLP for claim classification. A rule-based pre-filter handles clear cases, while ambiguous inputs are analyzed using a transformer-based NLI model. The text is further processed for buzzwords, certifications, and claim verification, along with fuzzy matching against a trusted brand database (RAG). A composite GreenTrust Index is then computed, followed by an AI-generated explanation and category-based trusted recommendations.

## Overview
This project analyzes product descriptions or URLs and classifies claims into:
- Vague environmental claims  
- Specific measurable claims  
- Certified sustainable products  

It also verifies claims using certifications, material checks, and a trusted brand database.

## Approach
- *Zero-shot NLP (NLI model)* for classification  
- *Rule-based heuristics* for detecting:
  - Certifications
  - Carbon claims
  - Material claims  
- *Fuzzy matching (RAG)* with trusted brands dataset  
- *Web scraping* for extracting product text from URLs  
- *Risk scoring* based on detected signals  

## Pipeline
1. Input (text or URL)  
2. Web scraping (if URL)  
3. Preprocessing  
4. Claim detection (buzzwords, certifications, numbers)  
5. Classification using NLI model  
6. Verification using:
   - Trusted brands CSV  
   - Rule-based checks  
7. Risk score calculation  
8. Output explanation (LLM-assisted or rule-based)

## Outputs
- Claim classification  
- Verification status (verified / unverified)  
- Greenwashing risk score (0–100)  
- Explanation of detected issues  

## Tech Stack
- Python  
- Streamlit (UI)  
- HuggingFace Transformers  
- Pandas, Scikit-learn  
- BeautifulSoup (web scraping)  
- RapidFuzz (matching)  

## ▶Run
```bash
streamlit run app.py
```

<span style="font-size: 0.8em;">Accuracy = 66.67%.</span>
