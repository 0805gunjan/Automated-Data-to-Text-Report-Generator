import streamlit as st
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fpdf import FPDF
from pathlib import Path

# ==============================
# 1️⃣ Streamlit Page Config
# ==============================
st.set_page_config(page_title="Automated Report Generator", layout="wide")
st.title("📊 Automated Data-to-Text Report Generator")

# ==============================
# 2️⃣ File Upload Section
# ==============================
st.sidebar.header("📁 Upload CSV Files")
orders_file = st.sidebar.file_uploader("Upload Orders CSV", type=["csv"])
details_file = st.sidebar.file_uploader("Upload Details CSV (optional)", type=["csv"])

TONE = st.sidebar.selectbox("Select Report Tone", ["formal", "concise", "executive", "detailed"])

if not orders_file:
    st.warning("Please upload at least one CSV file to proceed.")
    st.stop()

orders = pd.read_csv(orders_file)

if details_file:
    details = pd.read_csv(details_file)
    # Try to merge automatically if common column found
    common_cols = list(set(orders.columns) & set(details.columns))
    if common_cols:
        df = pd.merge(orders, details, on=common_cols[0], how="outer")
    else:
        df = pd.concat([orders, details], axis=1)
else:
    df = orders.copy()

st.success("✅ Files uploaded successfully!")

# ==============================
# 3️⃣ Handle Date Columns
# ==============================
date_col = None
for col in df.columns:
    if "date" in col.lower():
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Week"] = df[date_col].dt.isocalendar().week
else:
    df["Week"] = None

# ==============================
# 4️⃣ Compute KPIs Dynamically
# ==============================
columns = [c.lower() for c in df.columns]

def find_col(possible_names):
    for name in possible_names:
        for col in df.columns:
            if name in col.lower():
                return col
    return None

amount_col = find_col(["amount", "revenue", "sales"])
profit_col = find_col(["profit", "margin"])
state_col = find_col(["state", "region"])
city_col = find_col(["city", "town"])
category_col = find_col(["category", "segment"])
subcat_col = find_col(["sub-category", "subcategory"])
payment_col = find_col(["payment", "mode", "method"])

total_revenue = df[amount_col].sum() if amount_col else None
total_profit = df[profit_col].sum() if profit_col else None
top_state = df.groupby(state_col)[amount_col].sum().idxmax() if state_col and amount_col else None
top_city = df.groupby(city_col)[amount_col].sum().idxmax() if city_col and amount_col else None
top_category = df.groupby(category_col)[amount_col].sum().idxmax() if category_col and amount_col else None
top_subcategory = df.groupby(subcat_col)[amount_col].sum().idxmax() if subcat_col and amount_col else None
payment_counts = df[payment_col].value_counts().to_dict() if payment_col else {}

# ==============================
# 5️⃣ Detect anomalies (optional)
# ==============================
anomalies = pd.DataFrame()
if amount_col:
    mean_amount = df[amount_col].mean()
    std_amount = df[amount_col].std()
    anomalies = df[df[amount_col] < mean_amount - 2 * std_amount]

# ==============================
# 6️⃣ Load Fine-tuned Model
# ==============================
MODEL_PATH = "models/fine_tuned_t5"
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ==============================
# 7️⃣ Prepare Prompt
# ==============================
prompt_parts = []
if total_revenue is not None:
    prompt_parts.append(f"total_revenue={total_revenue}")
if total_profit is not None:
    prompt_parts.append(f"total_profit={total_profit}")
if top_state:
    prompt_parts.append(f"top_state={top_state}")
if top_city:
    prompt_parts.append(f"top_city={top_city}")
if top_category:
    prompt_parts.append(f"top_category={top_category}")
if top_subcategory:
    prompt_parts.append(f"top_subcategory={top_subcategory}")
if payment_counts:
    prompt_parts.append(f"payment_distribution={payment_counts}")

input_text = ", ".join(prompt_parts)
if not anomalies.empty:
    input_text += ", anomalies_detected=True"

prompt = f"Tone: {TONE}. {input_text}"

# ==============================
# 8️⃣ Generate Report
# ==============================
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=300, num_beams=4)
report_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

if not anomalies.empty:
    report_text += f"\n\n⚠️ Anomalies detected:\n{anomalies.to_dict(orient='records')}"

st.subheader("📝 Generated Report")
st.write(report_text)

# ==============================
# 9️⃣ Visualizations
# ==============================
st.subheader("📈 Visual Insights")

if amount_col and category_col in df.columns:
    st.bar_chart(df.groupby(category_col)[amount_col].sum())

if amount_col and date_col:
    st.line_chart(df.groupby(df[date_col].dt.to_period("W"))[amount_col].sum())

if profit_col:
    st.area_chart(df.groupby("Week")[profit_col].sum())

# ==============================
# 🔟 Export PDF Report
# ==============================
if st.button("📥 Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)
    Path("reports").mkdir(parents=True, exist_ok=True)
    pdf_path = "reports/auto_generated_report.pdf"
    pdf.output(pdf_path)
    st.success(f"✅ Report saved to {pdf_path}")
    with open(pdf_path, "rb") as f:
        st.download_button("⬇️ Download PDF", f, file_name="report.pdf")
