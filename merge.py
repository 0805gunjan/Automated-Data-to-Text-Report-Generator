import pandas as pd
import json

orders = pd.read_csv("data/orders.csv", parse_dates=["Order Date"])
details = pd.read_csv("data/details.csv")
df = pd.merge(orders, details, on="Order ID")

# Optional: forward fill missing dates
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Order Date'] = df['Order Date'].ffill()

# Create structured → text pairs
pairs = []
for _, row in df.iterrows():
    # Convert structured data to JSON-like string
    input_text = json.dumps({
        "OrderID": row["Order ID"],
        "Customer": row["CustomerName"],
        "State": row["State"],
        "City": row["City"],
        "Amount": row["Amount"],
        "Profit": row["Profit"],
        "Quantity": row["Quantity"],
        "Category": row["Category"],
        "SubCategory": row["Sub-Category"],
        "PaymentMode": row["PaymentMode"]
    })
    target_text = f"{row['CustomerName']} in {row['City']}, {row['State']} spent ${row['Amount']} with profit ${row['Profit']}."

    pairs.append({"input": input_text, "target": target_text})

# Save as JSONL for Hugging Face
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(json.dumps(pair) + "\n")