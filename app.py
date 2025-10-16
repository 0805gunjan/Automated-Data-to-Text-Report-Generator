# streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

# ====================
# 1️⃣ Streamlit UI
# ====================
st.title("Automated Sales Report Generator")

st.write("Upload your Orders CSV and Details CSV files to generate a PDF report.")

orders_file = st.file_uploader("Upload Orders CSV", type="csv")
details_file = st.file_uploader("Upload Details CSV", type="csv")

TONE = st.selectbox("Select report tone", ["formal", "concise", "executive", "detailed"])

if orders_file and details_file:
    # ====================
    # 2️⃣ Load and merge CSVs
    # ====================
    orders = pd.read_csv(orders_file)
    details = pd.read_csv(details_file)
    df = pd.merge(orders, details, on="Order ID")

    # ====================
    # 3️⃣ Handle dates
    # ====================
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Order Date'] = df['Order Date'].ffill()

    # ====================
    # 4️⃣ Interactive filters
    # ====================
    st.sidebar.header("Filters")
    min_date = df['Order Date'].min()
    max_date = df['Order Date'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    
    states = st.sidebar.multiselect("Select States", df['State'].unique(), default=df['State'].unique())
    cities = st.sidebar.multiselect("Select Cities", df['City'].unique(), default=df['City'].unique())
    categories = st.sidebar.multiselect("Select Categories", df['Category'].unique(), default=df['Category'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['Order Date'] >= pd.to_datetime(date_range[0])) &
        (df['Order Date'] <= pd.to_datetime(date_range[1])) &
        (df['State'].isin(states)) &
        (df['City'].isin(cities)) &
        (df['Category'].isin(categories))
    ]

    if filtered_df.empty:
        st.warning("No data available for selected filters!")
    else:
        # ====================
        # 5️⃣ Compute KPIs
        # ====================
        total_revenue = filtered_df['Amount'].sum()
        total_profit = filtered_df['Profit'].sum()
        top_state = filtered_df.groupby('State')['Amount'].sum().idxmax()
        top_city = filtered_df.groupby('City')['Amount'].sum().idxmax()
        top_category = filtered_df.groupby('Category')['Amount'].sum().idxmax()
        top_subcategory = filtered_df.groupby('Sub-Category')['Amount'].sum().idxmax()
        payment_counts = filtered_df['PaymentMode'].value_counts().to_dict()

        # ====================
        # 6️⃣ Detect anomalies
        # ====================
        threshold = 2
        mean_amount = filtered_df['Amount'].mean()
        std_amount = filtered_df['Amount'].std()
        anomalies = filtered_df[filtered_df['Amount'] < mean_amount - threshold * std_amount]

        # ====================
        # 7️⃣ Load fine-tuned T5
        # ====================
        MODEL_PATH = "models/fine_tuned_t5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
        tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

        # ====================
        # 8️⃣ Prepare prompt
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

        # ====================
        # 9️⃣ Generate report text
        # ====================
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=250, num_beams=4)
        report_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not anomalies.empty:
            report_text += f"\n\n⚠️ Anomalies detected:\n{anomalies.to_dict(orient='records')}"

        st.subheader("Generated Report:")
        st.text(report_text)

        # ====================
        # 🔟 Visualizations
        # ====================
        st.subheader("Visualizations")

        # Revenue by State
        fig, ax = plt.subplots()
        state_rev = filtered_df.groupby("State")["Amount"].sum().sort_values(ascending=False)
        sns.barplot(x=state_rev.index, y=state_rev.values, ax=ax)
        plt.xticks(rotation=45)
        ax.set_title("Revenue by State")
        st.pyplot(fig)

        # Revenue by Category
        fig2, ax2 = plt.subplots()
        cat_rev = filtered_df.groupby("Category")["Amount"].sum()
        sns.barplot(x=cat_rev.index, y=cat_rev.values, ax=ax2)
        plt.xticks(rotation=45)
        ax2.set_title("Revenue by Category")
        st.pyplot(fig2)

        # Payment Mode distribution
        fig3, ax3 = plt.subplots()
        sns.barplot(x=list(payment_counts.keys()), y=list(payment_counts.values()), ax=ax3)
        ax3.set_title("Payment Mode Distribution")
        st.pyplot(fig3)

        # ====================
        # 1️⃣1️⃣ Export PDF
        # ====================
        pdf_file = "reports/filtered_report.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, report_text)

        # Anomalies table
        if not anomalies.empty:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "⚠️ Anomalies Detected", ln=True)
            pdf.ln(5)

            pdf.set_font("Arial", "B", 12)
            col_width = pdf.w / (len(anomalies.columns) + 1)
            for col in anomalies.columns:
                pdf.cell(col_width, 10, str(col), border=1)
            pdf.ln()

            pdf.set_font("Arial", "", 12)
            for _, row in anomalies.iterrows():
                for col in anomalies.columns:
                    pdf.cell(col_width, 10, str(row[col]), border=1)
                pdf.ln()

        Path(pdf_file).parent.mkdir(exist_ok=True, parents=True)
        pdf.output(pdf_file)

        st.success(f"✅ PDF report generated: {pdf_file}")
        st.download_button("📥 Download PDF", data=open(pdf_file, "rb").read(), file_name="filtered_report.pdf")
