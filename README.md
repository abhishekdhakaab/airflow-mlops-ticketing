# 🧠 Intelligent Support Ticket Classifier (Airflow + LLM)

An **end-to-end ML pipeline** orchestrated with **Apache Airflow**, powered by:
- **Sentence Transformers** for embeddings  
- **Logistic Regression** for classification  
- **OpenRouter LLM** for optional auto-labeling of new tickets  
- **Batch inference** for daily predictions  
- **Dockerized Airflow** environment for full reproducibility  

---

## 📋 Features

✅ Extract and clean support ticket data  
✅ Generate semantic embeddings  
✅ Train a lightweight classifier  
✅ Automatically promote good models to production  
✅ Run batch predictions on the latest tickets  
✅ Optionally enrich unlabeled tickets using an LLM (OpenRouter)  

---

## 🗂️ Project Structure

```
ml-airflow-project/
├── dags/
│   └── support_ticket_pipeline.py
├── data/
│   ├── raw/
│   │   └── tickets.csv
│   ├── interim/
│   ├── processed/
│   ├── features/
│   ├── metrics/
│   ├── predictions/
│   └── models/
│       ├── candidates/
│       └── prod/
├── logs/
├── plugins/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone & Build

```bash
git clone <your_repo_url>
cd ml-airflow-project
docker compose build
```

---

### 2️⃣ Configure Environment

Create a `.env` file in the project root:

```env
AIRFLOW_UID=50000
AIRFLOW_GID=0
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxx
RUN_LLM=false
```

**Notes**
- Set `RUN_LLM=true` if you want the LLM enrichment stage to run.
- For OpenRouter’s `:free` models, enable *Free-tier model publication* in your OpenRouter privacy settings, or switch to a non-free model.

---

### 3️⃣ Initialize Airflow Database

```bash
docker compose run --rm airflow-init
```

---

### 4️⃣ Start Airflow Services

```bash
docker compose up -d
```

Then open the Airflow UI at  
👉 **http://localhost:8080**  
Login: `airflow / airflow`

---

## 🚀 Running the Pipeline

### Trigger the DAG
1. Go to the Airflow UI  
2. Find the DAG named **`support_ticket_pipeline`**  
3. Trigger it manually, or set a schedule in the code.

---

## 🧩 Pipeline Stages

| Stage | Description |
|--------|--------------|
| **Extract** | Reads raw CSV (`data/raw/tickets.csv`) into Parquet |
| **Clean** | Lowercases, strips punctuation, and normalizes whitespace |
| **Embed** | Uses SentenceTransformer to create embeddings |
| **Train** | Trains Logistic Regression; caches model + metrics |
| **Promote** | Promotes model to `models/prod/` if metrics ≥ threshold |
| **Predict** | Runs batch inference using prod model |
| **Enrich (LLM)** | Optionally classifies unlabeled tickets via OpenRouter LLM |

---

## 🧠 Example: `data/raw/tickets.csv`

```csv
id,text,label
1,"I was double charged on my invoice for July",Billing
2,"App keeps crashing when I click the upload button",Technical Issue
3,"Can you add SSO login with Okta?",Feature Request
4,"Not receiving password reset email",Technical Issue
5,"How do I change my subscription plan?",Billing
```

If you want to test the LLM enrichment, remove some labels and set `RUN_LLM=true`.

---

## 📊 Outputs

| Folder | Purpose |
|---------|----------|
| `data/interim/` | Raw → Parquet |
| `data/processed/` | Cleaned data |
| `data/features/` | Embeddings `.npz` |
| `data/models/` | Trained + prod models |
| `data/metrics/` | Training metrics JSON |
| `data/predictions/` | Batch predictions Parquet |

---

## 🕒 Scheduling Options

| Mode | Schedule | Description |
|------|-----------|-------------|
| **Manual (default)** | `None` | Run only when triggered |
| **Daily** | `"@daily"` | For regular scoring |
| **Yearly retrain** | `"@yearly"` | Long-term training cycle |

---

## 🤖 LLM Label Enrichment (Optional)

To enable:
```bash
export RUN_LLM=true
docker compose up -d
```

This triggers the **`enrich_labels_with_llm`** task which:
- Scans for unlabeled tickets in `data/raw/tickets.csv`
- Classifies them into one of:
  - `Billing`, `Technical Issue`, `Feature Request`, or `Other`
- Uses your **OpenRouter API key** via the `OpenAI` SDK

---

## 🧩 Extending

- Add **Slack/email notifications** for model promotions  
- Add **data validation** using `pandera`  
- Create a **daily-only inference DAG** if you want to score without retraining  

---

## 🧹 Clean Up

To stop and remove containers:

```bash
docker compose down
```

To clear Airflow data:

```bash
docker compose down -v
```

---

## 🧾 License

MIT License © 2025 — Your Name

---

## ❤️ Acknowledgements

Built with:
- [Apache Airflow](https://airflow.apache.org/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenRouter](https://openrouter.ai)
- [Scikit-learn](https://scikit-learn.org/)

---

**Enjoy your fully-orchestrated ML + LLM pipeline!**
