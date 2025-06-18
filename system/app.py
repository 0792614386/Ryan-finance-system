import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained models and scaler
model_bill = load_model("model_bill.h5")
model_expense = load_model("model_expense.h5")
model_balance = load_model("model_balance.h5")
scaler = joblib.load("scaler.pkl")

# Merchant options used during training
merchants = ['Amazon', 'Employer', 'Landlord', 'Netflix', 'Spotify', 'Walmart']
categories = ['groceries', 'rent', 'utilities', 'entertainment', 'salary', 'subscription']

def preprocess_input(amount, day, weekday, hour, balance, merchant):
    merchant_onehot = [1 if merchant == m else 0 for m in merchants]
    raw_features = np.array([[amount, day, weekday, hour, balance] + merchant_onehot])
    return scaler.transform(raw_features)

st.title("💵 AI Personal Finance Assistant")

# User Inputs
amount = st.number_input("💰 Transaction Amount", min_value=0.0, value=100.0)
day = st.slider("📅 Day of Month", 1, 31, 15)
weekday = st.slider("🗓️ Day of Week (0 = Sunday)", 0, 6, 2)
hour = st.slider("⏰ Hour of Day", 0, 23, 14)
balance = st.number_input("🏦 Account Balance", min_value=0.0, value=500.0)
merchant = st.selectbox("🏬 Merchant", merchants)

if st.button("Run Prediction"):
    X = preprocess_input(amount, day, weekday, hour, balance, merchant)

    # Predictions
    bill_due = model_bill.predict(X)[0][0] > 0.5
    expense_class = np.argmax(model_expense.predict(X))
    low_balance = model_balance.predict(X)[0][0] > 0.5

    # Display
    st.subheader("📊 Predictions")
    st.write(f"📌 **Bill Due Soon?** {'✅ Yes' if bill_due else '❌ No'}")
    st.write(f"📌 **Predicted Expense Category:** `{categories[expense_class]}`")
    st.write(f"📌 **Low Balance Warning:** {'⚠️ Yes' if low_balance else '✅ No'}")

# finance_system.py

import datetime

# Sample data
bills = [
    {"name": "Electricity", "due_date": "2025-06-15", "amount": 100},
    {"name": "Internet", "due_date": "2025-06-20", "amount": 50},
]

expenses = [
    {"description": "Grocery shopping", "amount": 80},
    {"description": "Movie tickets", "amount": 30},
    {"description": "Bus fare", "amount": 10},
    {"description": "Restaurant", "amount": 60},
]

expense_categories = {
    "grocery": "Food",
    "movie": "Entertainment",
    "bus": "Transport",
    "restaurant": "Food",
}

balance = 200  # Synthetic balance

# 1. Bill Reminders
def bill_reminders(bills):
    today = datetime.date.today()
    print("Upcoming Bill Reminders:")
    for bill in bills:
        due = datetime.datetime.strptime(bill["due_date"], "%Y-%m-%d").date()
        days_left = (due - today).days
        if days_left <= 7:
            print(f"  - {bill['name']} bill of ${bill['amount']} is due in {days_left} days ({bill['due_date']})")

# 2. Expense Classification
def classify_expenses(expenses):
    print("\nExpense Classification:")
    for exp in expenses:
        desc = exp["description"].lower()
        category = "Other"
        for key in expense_categories:
            if key in desc:
                category = expense_categories[key]
                break
        print(f"  - {exp['description']}: {category}")

# 3. Low Balance Simulation
def simulate_balance(balance, expenses, bills):
    total_expenses = sum(e["amount"] for e in expenses)
    total_bills = sum(b["amount"] for b in bills)
    projected_balance = balance - total_expenses - total_bills
    print(f"\nProjected balance after expenses and bills: ${projected_balance}")
    if projected_balance < 50:
        print("Warning: Low balance! Consider reducing expenses.")

if __name__ == "__main__":
    bill_reminders(bills)
    classify_expenses(expenses)
    simulate_balance(balance, expenses, bills)