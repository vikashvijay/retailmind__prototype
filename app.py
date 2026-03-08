"""
RetailMind — Advanced AI Retail Platform
Groq LLM + AWS S3 + DynamoDB
5 AI Features: Decisions, Insights, Copilot, Forecast, Competitor Analysis
"""
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import os, uuid, json, io, time
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

load_dotenv()

from models.demand_model import DemandPredictor
from utils.ai_engine import AIEngine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "retailmind-groq-2025")

# ── AWS Config ────────────────────────────────────────────────────────────────
AWS_REGION     = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET      = os.environ.get("S3_BUCKET", "")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "retailmind-sessions")
USE_AWS        = bool(S3_BUCKET)

# ── AWS Clients ───────────────────────────────────────────────────────────────
s3       = None
dyn_tbl  = None

if USE_AWS:
    try:
        s3      = boto3.client("s3", region_name=AWS_REGION)
        dyn_tbl = boto3.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMODB_TABLE)
        print(f"✅ AWS Storage: S3={S3_BUCKET}, DynamoDB={DYNAMODB_TABLE}")
    except Exception as e:
        print(f"⚠️  AWS init failed: {e} — using memory fallback")
        USE_AWS = False

# ── Memory fallback (when no AWS) ─────────────────────────────────────────────
DATA_STORE  = {}
MODEL_STORE = {}

AI       = AIEngine()
REQUIRED = ["Product","Category","Units_Sold","Current_Stock","Price","Competitor_Price"]


def sid():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


# ── Storage helpers ───────────────────────────────────────────────────────────
def save_data(session_id, df):
    js = df.to_json(orient="records", date_format="iso")
    if USE_AWS:
        try:
            s3.put_object(Bucket=S3_BUCKET, Key=f"sessions/{session_id}/data.json",
                         Body=js.encode(), ContentType="application/json")
            dyn_tbl.put_item(Item={
                "session_id": session_id,
                "s3_key": f"sessions/{session_id}/data.json",
                "row_count": str(df.shape[0]),
                "ttl": str(int(time.time()) + 86400 * 7),
            })
            return
        except ClientError as e:
            print(f"AWS save error: {e}")
    DATA_STORE[session_id] = js


def load_data(session_id):
    if USE_AWS:
        try:
            meta = dyn_tbl.get_item(Key={"session_id": session_id})
            if "Item" not in meta:
                return None
            obj = s3.get_object(Bucket=S3_BUCKET, Key=meta["Item"]["s3_key"])
            return pd.read_json(io.StringIO(obj["Body"].read().decode()), orient="records")
        except ClientError:
            return None
    raw = DATA_STORE.get(session_id)
    return pd.read_json(io.StringIO(raw), orient="records") if raw else None


def safe_int(v):
    try: return int(v)
    except: return 0

def safe_float(v):
    try: return round(float(v), 2)
    except: return 0.0


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify({
        "status":    "ok",
        "aws_mode":  USE_AWS,
        "ai_engine": "Groq (LLaMA 70B)",
        "s3_bucket": S3_BUCKET if USE_AWS else None,
        "dynamo":    DYNAMODB_TABLE if USE_AWS else None,
        "features":  ["decisions","insights","copilot","forecast","competitor"],
    })


@app.route("/api/upload", methods=["POST"])
def upload():
    s = sid()
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files accepted"}), 400
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    # Save raw CSV to S3
    if USE_AWS and s3:
        try:
            f.seek(0)
            s3.put_object(Bucket=S3_BUCKET, Key=f"uploads/{s}/{f.filename}",
                         Body=f.read(), ContentType="text/csv")
        except: pass

    # Date features
    if "Date" in df.columns:
        df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
        df["Day"]   = df["Date"].dt.day.fillna(1).astype(int)
        df["Month"] = df["Date"].dt.month.fillna(1).astype(int)
        df["Year"]  = df["Date"].dt.year.fillna(2024).astype(int)
    else:
        df["Day"] = 1; df["Month"] = 1; df["Year"] = 2024

    pred = DemandPredictor()
    df   = pred.fit_predict(df)
    MODEL_STORE[s] = pred
    save_data(s, df)

    return jsonify({
        "success": True,
        "storage": "AWS S3 + DynamoDB" if USE_AWS else "Memory",
        "summary": {
            "rows":       safe_int(df.shape[0]),
            "products":   safe_int(df["Product"].nunique()),
            "categories": safe_int(df["Category"].nunique()),
            "has_date":   "Date" in df.columns,
        }
    })


@app.route("/api/dashboard")
def dashboard():
    s  = sid()
    df = load_data(s)
    if df is None: return jsonify({"error": "No data"}), 400

    prod = df.groupby("Product").agg(
        sold=("Units_Sold","sum"), stock=("Current_Stock","mean"),
        demand=("Predicted_Demand","mean"), price=("Price","mean"),
        comp=("Competitor_Price","mean"),
    ).reset_index()

    critical = int((prod["stock"] < prod["demand"] * 0.7).sum())
    overpriced = int((prod["price"] > prod["comp"] * 1.1).sum())

    kpis = {
        "total_rows":   safe_int(df.shape[0]),
        "products":     safe_int(prod.shape[0]),
        "categories":   safe_int(df["Category"].nunique()),
        "avg_stock":    safe_float(df["Current_Stock"].mean()),
        "avg_demand":   safe_float(df["Predicted_Demand"].mean()),
        "critical":     critical,
        "overpriced":   overpriced,
        "revenue_est":  safe_float((prod["sold"] * prod["price"]).sum()),
    }

    top20        = prod.sort_values("sold", ascending=False).head(20)
    demand_stock = top20[["Product","sold","stock","demand"]].to_dict(orient="records")
    cat          = df.groupby("Category")["Units_Sold"].sum().reset_index()
    cat.columns  = ["category","sales"]

    daily = []
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        d = df.dropna(subset=["Date"]).groupby(df["Date"].dt.date)["Units_Sold"].sum()
        daily = [{"date": str(k), "sales": int(v)} for k, v in d.items()]

    price_cmp = prod.sort_values("price").head(20)[["Product","price","comp"]].to_dict(orient="records")

    # Price health breakdown
    price_health = {
        "overpriced":     int((prod["price"] > prod["comp"] * 1.05).sum()),
        "underpriced":    int((prod["price"] < prod["comp"] * 0.95).sum()),
        "competitive":    int(((prod["price"] >= prod["comp"] * 0.95) & (prod["price"] <= prod["comp"] * 1.05)).sum()),
    }

    return jsonify({
        "kpis": kpis,
        "demand_stock": demand_stock,
        "category_sales": cat.to_dict(orient="records"),
        "daily_trend": daily,
        "price_comparison": price_cmp,
        "price_health": price_health,
    })


@app.route("/api/decisions")
def decisions():
    s  = sid()
    df = load_data(s)
    if df is None: return jsonify({"error": "No data"}), 400
    return jsonify(AI.generate_decisions(df))


@app.route("/api/insights")
def insights():
    s  = sid()
    df = load_data(s)
    if df is None: return jsonify({"error": "No data"}), 400
    return jsonify(AI.generate_insights(df))


@app.route("/api/copilot", methods=["POST"])
def copilot():
    s  = sid()
    df = load_data(s)
    if df is None: return jsonify({"error": "No data"}), 400
    q  = (request.get_json() or {}).get("question","").strip()
    if not q: return jsonify({"error": "Empty question"}), 400
    return jsonify({"answer": AI.copilot(q, df)})


@app.route("/api/forecast")
def forecast():
    """NEW: 30-day AI sales forecast"""
    s  = sid()
    df = load_data(s)
    if df is None: return jsonify({"error": "No data"}), 400
    return jsonify(AI.generate_forecast(df))


@app.route("/api/competitor")
def competitor():
    """NEW: AI competitor pricing analysis"""
    s  = sid()
    df = load_data(s)
    if df is None: return jsonify({"error": "No data"}), 400
    return jsonify(AI.generate_competitor_analysis(df))


@app.route("/api/raw_data")
def raw_data():
    s  = sid()
    df = load_data(s)
    if df is None: return jsonify({"error": "No data"}), 400
    cols = [c for c in df.columns if c not in ("Day","Month","Year")]
    return jsonify(df[cols].head(200).fillna("").to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
