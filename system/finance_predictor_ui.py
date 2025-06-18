import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd

# Assume the following are already defined from your code:
# scaler, model_bill, model_expense, model_balance

# Example: Load or define your models here if not already done
from joblib import load
scaler = load('scaler.joblib')
model_bill = load('model_bill.joblib')
model_expense = load('model_expense.joblib')
model_balance = load('model_balance.joblib')

categories = ['groceries', 'rent', 'utilities', 'entertainment', 'salary', 'subscription']
merchants = ['Walmart', 'Netflix', 'Spotify', 'Amazon', 'Landlord', 'Employer']

def predict():
    try:
        # Collect user input
        amount = float(amount_var.get())
        category = category_var.get()
        merchant = merchant_var.get()
        day_of_month = int(day_of_month_var.get())
        day_of_week = int(day_of_week_var.get())
        hour = int(hour_var.get())
        balance = float(balance_var.get())

        # Prepare input DataFrame
        input_dict = {
            'amount': [amount],
            'day_of_month': [day_of_month],
            'day_of_week': [day_of_week],
            'hour': [hour],
            'balance': [balance]
        }
        for m in merchants[1:]:  # drop_first=True in get_dummies
            input_dict[f'mch_{m}'] = [1 if merchant == m else 0]
        input_df = pd.DataFrame(input_dict)

        # Align columns with training data
        # (Assumes you used pd.get_dummies with drop_first=True)
        # If you saved the columns order, use it here for input_df = input_df.reindex(columns=your_columns, fill_value=0)

        # Scale features
        X_input = scaler.transform(input_df)

        # Predict
        bill_due_pred = model_bill.predict(X_input)[0][0]
        expense_class_pred = np.argmax(model_expense.predict(X_input), axis=1)[0]
        low_balance_pred = model_balance.predict(X_input)[0][0]

        # Show results
        result = (
            f"Bill Due Prediction: {'Yes' if bill_due_pred > 0.5 else 'No'}\n"
            f"Expense Class: {categories[expense_class_pred]}\n"
            f"Low Balance Alert: {'Yes' if low_balance_pred > 0.5 else 'No'}"
        )
        messagebox.showinfo("Prediction Result", result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Finance Predictor")

# Variables
amount_var = tk.StringVar()
category_var = tk.StringVar(value=categories[0])
merchant_var = tk.StringVar(value=merchants[0])
day_of_month_var = tk.StringVar()
day_of_week_var = tk.StringVar()
hour_var = tk.StringVar()
balance_var = tk.StringVar()

# Layout
fields = [
    ("Amount", amount_var),
    ("Category", category_var),
    ("Merchant", merchant_var),
    ("Day of Month", day_of_month_var),
    ("Day of Week (0=Mon)", day_of_week_var),
    ("Hour (0-23)", hour_var),
    ("Balance", balance_var)
]

for idx, (label, var) in enumerate(fields):
    tk.Label(root, text=label).grid(row=idx, column=0, padx=5, pady=5, sticky='e')
    if label in ["Category", "Merchant"]:
        values = categories if label == "Category" else merchants
        ttk.Combobox(root, textvariable=var, values=values, state="readonly").grid(row=idx, column=1, padx=5, pady=5)
    else:
        tk.Entry(root, textvariable=var).grid(row=idx, column=1, padx=5, pady=5)

tk.Button(root, text="Predict", command=predict).grid(row=len(fields), column=0, columnspan=2, pady=10)

root.mainloop()