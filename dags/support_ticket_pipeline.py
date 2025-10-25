"""
support_ticket_pipeline.py
End-to-end ML + LLM workflow for classifying support tickets, plus batch predictions.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from openai import OpenAI

# ---------- CONFIG ----------
BASE_DIR = Path("/opt/airflow")  # inside container
DATA_DIR = BASE_DIR / "data"

RAW_CSV = DATA_DIR / "raw" / "tickets.csv"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
CANDIDATES_DIR = MODELS_DIR / "candidates"
PROD_DIR = MODELS_DIR / "prod"
METRICS_DIR = DATA_DIR / "metrics"
PRED_DIR = DATA_DIR / "predictions"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42

def ensure_dirs():
    for d in [DATA_DIR, INTERIM_DIR, PROCESSED_DIR, FEATURES_DIR, CANDIDATES_DIR, PROD_DIR, METRICS_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ---------- TASKS ----------
@task
def extract_tickets(raw_csv: str, out_path: str) -> str:
    df = pd.read_csv(raw_csv)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    pd.DataFrame(df).to_parquet(out_path, index=False)
    print(f"[extract] {len(df)} rows -> {out_path}")
    return out_path

@task
def clean_text(in_parquet: str, out_path: str) -> str:
    from string import punctuation
    df = pd.read_parquet(in_parquet)
    df["text"] = (
        df["text"].astype(str).str.lower()
        .str.replace(f"[{punctuation}]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["text"] != ""]
    df.to_parquet(out_path, index=False)
    print(f"[clean] {len(df)} rows -> {out_path}")
    return out_path

@task
def generate_embeddings(clean_parquet: str, out_npz: str, model_name: str = MODEL_NAME) -> str:
    df = pd.read_parquet(clean_parquet)
    texts = df["text"].astype(str).tolist()
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    if "id" in df.columns:
        try:
            ids = df["id"].astype("int64").to_numpy()
        except Exception:
            ids = np.asarray(df["id"].astype("string").fillna("").to_numpy(), dtype="U")
    else:
        ids = np.arange(len(df), dtype=np.int64)

    labels = None
    if "label" in df.columns:
        labels = np.asarray(df["label"].astype("string").fillna("").to_numpy(), dtype="U")

    if labels is None:
        np.savez_compressed(out_npz, embeddings=embs, ids=ids)
    else:
        np.savez_compressed(out_npz, embeddings=embs, ids=ids, labels=labels)

    print(f"[embed] shape={embs.shape} -> {out_npz}")
    return out_npz

@task
def train_small_model(emb_npz: str, model_out: str, metrics_out: str) -> str:
    npz = np.load(emb_npz, allow_pickle=False)
    X = npz["embeddings"]
    if "labels" not in npz:
        raise ValueError("No 'labels' in embeddings.")
    y = np.asarray(npz["labels"], dtype="U")

    classes, counts = np.unique(y, return_counts=True)
    clf = LogisticRegression(max_iter=2000, n_jobs=1, solver="saga", class_weight="balanced", random_state=SEED)

    metrics: Dict = {}
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    if counts.min() >= 2 and len(y) >= 6:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1w = f1_score(y_val, y_pred, average="weighted")
        report = classification_report(y_val, y_pred, output_dict=True)
        metrics.update({"accuracy": float(acc), "f1_weighted": float(f1w), "report": report})
    else:
        print(f"[warn] Tiny/imbalanced dataset: {dict(zip(classes, counts))} — training on all data, no validation.")
        clf.fit(X, y)
        metrics.update({"accuracy": None, "f1_weighted": None, "report": {}})

    import joblib
    joblib.dump(clf, model_out)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train] model={model_out}, f1={metrics.get('f1_weighted')}")
    return metrics_out

@task
def maybe_promote_to_prod(model_path: str, metrics_path: str, prod_model: str, threshold_f1: float = 0.7) -> str:
    with open(metrics_path) as f:
        metrics = json.load(f)
    f1 = metrics.get("f1_weighted")
    if f1 is None:
        print("[deploy] skipped: no f1 metric.")
        return prod_model
    if f1 >= threshold_f1:
        import shutil
        Path(prod_model).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, prod_model)
        print(f"[deploy] promoted {model_path} -> {prod_model}")
    else:
        print(f"[deploy] not promoted (f1={f1} < {threshold_f1})")
    return prod_model

@task
def batch_predict_latest(clean_parquet: str, prod_model_path: str, out_path: str) -> str:
    import joblib
    if not Path(prod_model_path).exists():
        print(f"[predict] No prod model at {prod_model_path}. Writing empty predictions.")
        pd.DataFrame(columns=["id","text","pred_label","pred_proba"]).to_parquet(out_path, index=False)
        return out_path

    df = pd.read_parquet(clean_parquet)
    texts = df["text"].astype(str).tolist()

    st_model = SentenceTransformer(MODEL_NAME)
    X = st_model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    clf = joblib.load(prod_model_path)
    pred = clf.predict(X)
    proba = clf.predict_proba(X).max(axis=1) if hasattr(clf, "predict_proba") else np.ones(len(pred), dtype=float)

    out = pd.DataFrame({
        "id": df["id"] if "id" in df.columns else range(len(df)),
        "text": df["text"],
        "pred_label": pred,
        "pred_proba": proba,
    })
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[predict] {len(out)} rows -> {out_path}")
    return out_path

@task.short_circuit
def llm_gate(run_llm: str = os.getenv("RUN_LLM", "false")) -> bool:
    on = str(run_llm).strip().lower() in ("true","1","yes","y")
    print(f"[gate] RUN_LLM={run_llm} -> {on}")
    return on

@task
def enrich_labels_with_llm(raw_csv: str, out_csv: str,
                           model: str = "mistralai/mistral-7b-instruct",
                           site_url: str = "https://yourproject.local",
                           site_title: str = "Support Ticket Classifier") -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[llm] missing OPENROUTER_API_KEY → skipped.")
        return raw_csv

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    df = pd.read_csv(raw_csv)
    if "label" not in df.columns:
        df["label"] = ""

    unlabeled = df[df["label"].isna() | (df["label"].astype(str).strip() == "")]
    print(f"[llm] Found {len(unlabeled)} unlabeled tickets.")

    for i, row in unlabeled.iterrows():
        text = row["text"]
        prompt = (
            "Classify this customer support ticket into one of: "
            "Billing, Technical Issue, Feature Request, or Other.\n"
            f'Ticket: "{text}"\n'
            "Respond with only the label."
        )
        try:
            completion = client.chat.completions.create(
                extra_headers={"HTTP-Referer": site_url, "X-Title": site_title},
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful classifier."},
                    {"role": "user", "content": prompt},
                ],
            )
            label = completion.choices[0].message.content.strip()
            df.loc[i, "label"] = label
            print(f"[llm] Row {i}: {label}")
        except Exception as e:
            print(f"[llm] failed on row {i}: {e}")
        time.sleep(1.0)

    df.to_csv(out_csv, index=False)
    print(f"[llm] Wrote enriched CSV to {out_csv}")
    return out_csv

# ---------- DAG ----------
@dag(
    dag_id="support_ticket_pipeline",
    description="ML + LLM pipeline for support ticket classification with batch predictions",
    start_date=days_ago(1),
    schedule_interval=None,   # manual for dev; change to "@daily" or "@yearly" later
    catchup=False,
    tags=["ml","llm","support"]
)
def support_ticket_pipeline():
    ensure_dirs()

    # Date-stamped outputs
    date_str = datetime.now().strftime("%Y%m%d")
    interim_path   = str(INTERIM_DIR / f"tickets_{date_str}.parquet")
    processed_path = str(PROCESSED_DIR / f"tickets_clean_{date_str}.parquet")
    emb_path       = str(FEATURES_DIR / f"embeddings_{date_str}.npz")
    model_path     = str(CANDIDATES_DIR / f"model_{date_str}.pkl")
    metrics_path   = str(METRICS_DIR / f"metrics_{date_str}.json")
    prod_model     = str(PROD_DIR / "model.pkl")
    preds_path     = str(PRED_DIR / f"preds_{date_str}.parquet")
    enriched_csv   = str(DATA_DIR / "raw" / "tickets_enriched.csv")

    # Core pipeline
    extracted = extract_tickets(str(RAW_CSV), interim_path)
    cleaned   = clean_text(extracted, processed_path)
    embedded  = generate_embeddings(cleaned, emb_path)
    trained_m = train_small_model(embedded, model_path, metrics_path)
    # Deploy + batch predictions
    prod = maybe_promote_to_prod(model_path, trained_m, prod_model)
    _pred = batch_predict_latest(cleaned, prod, preds_path)

    # Optional LLM enrichment (runtime gated)
    gate = llm_gate()
    llm = enrich_labels_with_llm(str(RAW_CSV), enriched_csv)
    cleaned >> gate >> llm

support_ticket_pipeline()