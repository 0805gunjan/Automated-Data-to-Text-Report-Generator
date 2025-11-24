#  Automated Data-to-Text Report Generator (Streamlit + T5 Fine-Tuning)

This project is an end-to-end automated report generation system that converts **CSV data into natural-language business reports** using a **fine-tuned T5 transformer model** and an interactive **Streamlit dashboard**.

The system:
- Accepts one or more CSV input files  
- Automatically merges datasets  
- Detects important metrics (KPIs) dynamically  
- Generates clean and business-ready reports in multiple tones  
- Visualizes insights (charts for revenue, profit, category performance, etc.)  
- Detects anomalies  
- Allows exporting the generated report as a PDF  
- Uses a custom **fine-tuned T5 model** trained on your own business dataset  

---

##  Features

### **1. CSV Upload & Auto-Merging**
- Upload `orders.csv` (required)  
- Upload `details.csv` (optional)  
- If a common column exists (e.g., `Order ID`), the app **auto-merges** both files  
- If no common column exists, files are **concatenated side-by-side**

---

### **2. Automatic KPI Extraction**
The app dynamically detects important column names based on patterns:
- Revenue (`amount`, `sales`, `revenue`)  
- Profit (`profit`, `margin`)  
- State / Region  
- City  
- Category  
- Sub-Category  
- Payment Mode  

Then computes:  
- **Total Revenue**  
- **Total Profit**  
- **Top Performing State**  
- **Top Performing City**  
- **Top Category & Sub-Category**  
- **Payment method distribution**

---

### **3. Date Handling & Derived Metrics**
If a date column exists:
- Automatically parsed into datetime  
- Weekly grouping created  
- Used to plot revenue or profit trends  

---

### **4. Anomaly Detection (Optional)**
Detects outliers using **mean − 2 × std deviation** logic.  
These anomalies get added into the final report.

---

### **5. Fine-tuned T5 Transformer Model**
The core of the project is a **custom fine-tuned T5 model**, trained on your business data.

The model learns to:
- Read structured KPI inputs like  
  `total_revenue=10000, top_state=California, ...`  
- Generate a complete, fluent business report.

Supports multiple tones:
- **Formal**  
- **Concise**  
- **Executive**  
- **Detailed**

---

### **6. Natural-Language Report Generation**
After KPIs are extracted, the Streamlit app sends the prompt to the fine-tuned model and generates a readable report.

data_to_text/
│
├── data/
│ ├── sample_sales.csv
│ └── training_pairs.csv
│
├── src/
│ ├── dataloader.py
│ ├── analyzer.py
│ ├── anomaly.py
│ ├── nlg_finetune.py
│ ├── nlg_inference.py
│ ├── reporter.py
│ ├── scheduler.py
│ ├── auth.py
│ ├── monitor.py
│ ├── utils.py
│ └── app.py
│
├── reports/
│ └── weekly_report.pdf
│
├── Dockerfile
├── requirements.txt
└── README.md

