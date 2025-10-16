# streamlit_app_flexible.py
import streamlit as st
import pandas as pd
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="AI-Powered Data Report Generator", layout="wide")
st.title("🤖 AI-Powered Data Report Generator")

st.write("""
Upload your CSV file(s) — the app will adapt automatically, 
generate visualizations for available columns, and create an AI-written report.
""")

# ====================
# 1️⃣ File Upload
# ====================
file1 = st.file_uploader("Upload first CSV file", type="csv")
file2 = st.file_uploader("Upload second CSV file (optional)", type="csv")

if file1:
    df1 = pd.read_csv(file1)
    st.success(f"✅ First CSV loaded with {df1.shape[0]} rows and {df1.shape[1]} columns.")
    st.dataframe(df1.head())
else:
    st.stop()

if file2:
    df2 = pd.read_csv(file2)
    common_cols = list(set(df1.columns) & set(df2.columns))
    if common_cols:
        df = pd.merge(df1, df2, on=common_cols[0], how="outer")
        st.info(f"🔗 Merged on common column: {common_cols[0]}")
    else:
        df = pd.concat([df1, df2], axis=0)
        st.warning("⚠️ No common column found — concatenated instead.")
else:
    df = df1.copy()

# ====================
# 2️⃣ Column Detection
# ====================
st.subheader("🧩 Column Mapping (Optional)")
st.write("Select columns if they exist; leave blank if not applicable.")

def select_if_exists(label):
    cols = ["<None>"] + list(df.columns)
    choice = st.selectbox(label, cols)
    return None if choice == "<None>" else choice

col_order_id = select_if_exists("Order ID column")
col_date = select_if_exists("Order Date column")
col_amount = select_if_exists("Amount / Revenue column")
col_profit = select_if_exists("Profit column")
col_state = select_if_exists("State column")
col_city = select_if_exists("City column")
col_category = select_if_exists("Category column")
col_payment = select_if_exists("Payment Mode column")

# ====================
# 3️⃣ Handle Dates
# ====================
if col_date:
    df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
    df[col_date] = df[col_date].ffill()
else:
    st.warning("No date column detected — skipping time-based filtering.")

# ====================
# 4️⃣ Filters
# ====================
st.sidebar.header("📅 Filters")
filtered_df = df.copy()

if col_date:
    min_date, max_date = df[col_date].min(), df[col_date].max()
    if pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        filtered_df = filtered_df[
            (filtered_df[col_date] >= pd.to_datetime(date_range[0])) &
            (filtered_df[col_date] <= pd.to_datetime(date_range[1]))
        ]

if col_state and col_state in df.columns:
    selected_states = st.sidebar.multiselect("Select States", df[col_state].unique())
    if selected_states:
        filtered_df = filtered_df[filtered_df[col_state].isin(selected_states)]

if filtered_df.empty:
    st.warning("⚠️ No data available after applying filters.")
    st.stop()

# ====================
# 5️⃣ KPI Calculations
# ====================
def safe_sum(col):
    return filtered_df[col].sum() if col and col in filtered_df.columns else "N/A"

total_revenue = safe_sum(col_amount)
total_profit = safe_sum(col_profit)

top_state = (
    filtered_df.groupby(col_state)[col_amount].sum().idxmax()
    if col_state and col_amount in filtered_df.columns else "N/A"
)
top_city = (
    filtered_df.groupby(col_city)[col_amount].sum().idxmax()
    if col_city and col_amount in filtered_df.columns else "N/A"
)

payment_counts = (
    filtered_df[col_payment].value_counts().to_dict()
    if col_payment in filtered_df.columns else {}
)

# ====================
# 6️⃣ Visualization Section
# ====================
st.subheader("📊 Visualizations")

if col_state and col_amount in df.columns:
    fig, ax = plt.subplots()
    sns.barplot(x=filtered_df[col_state], y=filtered_df[col_amount], ax=ax)
    ax.set_title("Revenue by State")
    plt.xticks(rotation=45)
    st.pyplot(fig)

if col_category and col_amount in df.columns:
    fig2, ax2 = plt.subplots()
    sns.barplot(x=filtered_df[col_category], y=filtered_df[col_amount], ax=ax2)
    ax2.set_title("Revenue by Category")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

# ====================
# 7️⃣ Load fine-tuned Model
# ====================
MODEL_PATH = "models/fine_tuned_t5"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# ====================
# 8️⃣ Generate Report
# ====================
TONE = st.selectbox("Select Tone for Report", ["formal", "concise", "executive", "detailed"])

summary_info = {
    "total_revenue": total_revenue,
    "total_profit": total_profit,
    "top_state": top_state,
    "top_city": top_city,
    "payment_distribution": payment_counts,
}

input_text = ", ".join(f"{k}={v}" for k, v in summary_info.items() if v != "N/A")
prompt = f"Tone: {TONE}. Summarize the following data insights: {input_text}"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=250, num_beams=4)
report_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

st.subheader("📝 Generated Report:")
st.text(report_text)

# ====================
# 9️⃣ Export PDF
# ====================
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, report_text)
Path("reports").mkdir(exist_ok=True, parents=True)
pdf_file = "reports/generated_report.pdf"
pdf.output(pdf_file)

st.download_button(
    "📥 Download PDF Report",
    data=open(pdf_file, "rb").read(),
    file_name="AI_Report.pdf",
    mime="application/pdf"
)

