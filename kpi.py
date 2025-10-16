import pandas as pd
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fpdf import FPDF

# ====================
# 1️⃣ Configuration
# ====================
ORDERS_FILE = "data/orders.csv"
DETAILS_FILE = "data/details.csv"
MODEL_DIR = "models/fine_tuned_t5"  # your local fine-tuned T5
PDF_FILE = "reports/weekly_report.pdf"
TONE = "formal"  # formal, concise, executive, detailed

# ====================
# 2️⃣ Load and Merge Data
# ====================
orders = pd.read_csv(ORDERS_FILE)
details = pd.read_csv(DETAILS_FILE)
df = pd.merge(orders, details, on="Order ID")

# ====================
# 3️⃣ Handle Order Date
# ====================
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Order Date'] = df['Order Date'].ffill()  # forward fill missing dates

if df['Order Date'].isna().any():
    print("⚠️ Some Order Date values could not be converted to datetime!")

df['Week'] = df['Order Date'].dt.isocalendar().week

# ====================
# 4️⃣ Compute KPIs
# ====================
total_revenue = df['Amount'].sum()
total_profit = df['Profit'].sum()

top_state = df.groupby('State')['Amount'].sum().idxmax()
top_city = df.groupby('City')['Amount'].sum().idxmax()

top_category = df.groupby('Category')['Amount'].sum().idxmax()
top_subcategory = df.groupby('Sub-Category')['Amount'].sum().idxmax()

payment_counts = df['PaymentMode'].value_counts().to_dict()

# ====================
# 5️⃣ Detect anomalies (Revenue below 2 std devs)
# ====================
threshold = 2
mean_amount = df['Amount'].mean()
std_amount = df['Amount'].std()
anomalies = df[df['Amount'] < mean_amount - threshold*std_amount]

# ====================
# 6️⃣ Load fine-tuned T5 model
# ====================
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Relative path (Windows-friendly)
MODEL_PATH = "models/fine_tuned_t5"

# Load model & tokenizer
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# Move model to GPU if available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)



# ====================
# 7️⃣ Prepare prompt for report
# ====================
input_text = (
    f"total_revenue={total_revenue}, total_profit={total_profit}, "
    f"top_state={top_state}, top_city={top_city}, "
    f"top_category={top_category}, top_subcategory={top_subcategory}, "
    f"payment_distribution={payment_counts}"
)

if not anomalies.empty:
    input_text += ", anomalies_detected=True"

prompt = f"Tone: {TONE}. {input_text}"


import matplotlib.pyplot as plt
import seaborn as sns

# Folder for temp images
img_dir = Path("reports/images")
img_dir.mkdir(exist_ok=True, parents=True)

# ---------- Revenue by State ----------
plt.figure(figsize=(6,4))
sns.barplot(x=df.groupby('State')['Amount'].sum().index,
            y=df.groupby('State')['Amount'].sum().values,
            palette="Blues_d")
plt.title("Revenue by State")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
state_img = img_dir / "revenue_by_state.png"
plt.savefig(state_img)
plt.close()

# ---------- Revenue by Category ----------
plt.figure(figsize=(6,4))
sns.barplot(x=df.groupby('Category')['Amount'].sum().index,
            y=df.groupby('Category')['Amount'].sum().values,
            palette="Greens_d")
plt.title("Revenue by Category")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
category_img = img_dir / "revenue_by_category.png"
plt.savefig(category_img)
plt.close()

# ---------- Payment Mode Distribution ----------
plt.figure(figsize=(6,4))
sns.barplot(x=list(payment_counts.keys()),
            y=list(payment_counts.values()),
            palette="Oranges_d")
plt.title("Payment Mode Distribution")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
payment_img = img_dir / "payment_distribution.png"
plt.savefig(payment_img)
plt.close()


# ====================
# 8️⃣ Generate report text
# ====================
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=250, num_beams=4)
report_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

if not anomalies.empty:
    report_text += f"\n\n⚠️ Anomalies detected:\n{anomalies.to_dict(orient='records')}"

print("📝 Generated Report:")
print(report_text)


# ====================
# 9️⃣ Export PDF
# ====================
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, report_text)

pdf.ln(10)  # add some space

# Add plots
for img_path in [state_img, category_img, payment_img]:
    pdf.image(str(img_path), w=180)  # width 180mm
    pdf.ln(10)  # space after image

Path(PDF_FILE).parent.mkdir(exist_ok=True, parents=True)  # create folder if missing

if not anomalies.empty:
    pdf.add_page()  # new page for anomalies
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "⚠️ Anomalies Detected", ln=True)
    pdf.ln(5)

    # Table header
    pdf.set_font("Arial", "B", 12)
    col_width = pdf.w / (len(anomalies.columns) + 1)  # auto width
    for col in anomalies.columns:
        pdf.cell(col_width, 10, str(col), border=1)
    pdf.ln()

    # Table rows
    pdf.set_font("Arial", "", 12)
    for _, row in anomalies.iterrows():
        for col in anomalies.columns:
            pdf.cell(col_width, 10, str(row[col]), border=1)
        pdf.ln()

pdf.output(PDF_FILE)
print(f"✅ Report with visualizations saved as {PDF_FILE}")






