"""
command.py - Telegram Tennis AI Prediction Bot
--------------------------------------------------
Purpose: Predict tennis match totals and provide premium statistics.
Features:
1. Validate user input for all required stats.
2. Predict first set + full match totals.
3. Compute home/away win probability %.
4. Compare to Over/Under line (value edge).
5. Apply bias adjustment from past results.
6. Track model accuracy %.
7. Premium formatted report.
8. Store past matches in SQLite for adaptive learning.
Commands:
    /start   - Welcome message
    /predict - Ask user to input match stats
    /result  - Record actual match score and update model
    /help    - Support info
"""

import logging
import sqlite3
import os
import numpy as np
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# ---------------- CONFIG ----------------
TOKEN = os.getenv("BOT_TOKEN")  # Railway environment variable
logging.basicConfig(level=logging.INFO)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("tennis_ai.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    home_set1 REAL,
    away_set1 REAL,
    home_win REAL,
    away_win REAL,
    home_form REAL,
    away_form REAL,
    h2h_home REAL,
    h2h_away REAL,
    over_line REAL,
    predicted_total REAL,
    actual_total REAL,
    error REAL
)
""")
conn.commit()

# ---------------- MACHINE LEARNING ----------------
weights = np.zeros(10)
learning_rate = 0.0001

def train_model():
    cursor.execute("SELECT * FROM matches WHERE actual_total IS NOT NULL")
    rows = cursor.fetchall()
    if len(rows) < 5:
        return len(rows)
    global weights
    for _ in range(300):
        for row in rows:
            x = np.array(row[1:11])
            y = row[12]
            prediction = np.dot(weights, x)
            error = prediction - y
            weights -= learning_rate * error * x
    return len(rows)

def bias_adjustment():
    cursor.execute("SELECT AVG(error) FROM matches WHERE error IS NOT NULL")
    result = cursor.fetchone()[0]
    return result if result else 0

def model_accuracy():
    cursor.execute("SELECT COUNT(*) FROM matches WHERE error IS NOT NULL")
    total = cursor.fetchone()[0]
    if total == 0:
        return 0
    cursor.execute("SELECT COUNT(*) FROM matches WHERE ABS(error) <= 2")
    correct = cursor.fetchone()[0]
    return round((correct / total) * 100, 2)

def predict_total(features):
    count = train_model()
    if count < 5:
        base = (features[0] + features[1]) * 2
    else:
        base = np.dot(weights, features)
    bias = bias_adjustment()
    prediction = base + bias
    return round(prediction, 2), count, bias

# ---------------- VALIDATION ----------------
REQUIRED_FIELDS = [
    "home_set1","away_set1","home_win","away_win",
    "home_form","away_form","h2h_home","h2h_away","over_line"
]

def validate_data(data):
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False
    return True

# ---------------- COMMAND HANDLERS ----------------
def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "🏆 ELITE TENNIS AI PREDICTOR BOT\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "Adaptive Learning + Bias Correction\n"
        "Use /predict to input match data.\n"
        "Use /help for support."
    )

def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Need help? Contact owner:\n"
        "🇬🇭 +233531386553\n"
        "Email: samuelkelvin950@gmail.com"
    )

def predict(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Send match stats line by line in this format:\n"
        "home_set1=5.2\n"
        "away_set1=4.8\n"
        "home_win=65\n"
        "away_win=55\n"
        "home_form=70\n"
        "away_form=60\n"
        "h2h_home=3\n"
        "h2h_away=2\n"
        "over_line=21.5"
    )

def handle_message(update: Update, context: CallbackContext):
    text = update.message.text
    data = {}
    try:
        for line in text.strip().split("\n"):
            key, value = line.split("=")
            data[key.strip()] = float(value.strip())
    except:
        update.message.reply_text("❌ Invalid format. Use key=value per line.")
        return

    if not validate_data(data):
        update.message.reply_text("❌ Missing required inputs.")
        return

    features = [
        data["home_set1"], data["away_set1"], data["home_win"], data["away_win"],
        data["home_form"], data["away_form"], data["h2h_home"], data["h2h_away"],
        data["over_line"], 1
    ]

    prediction, sample_count, bias = predict_total(features)
    first_set = data["home_set1"] + data["away_set1"]

    # Win probability
    total_win = data["home_win"] + data["away_win"]
    home_prob = round(data["home_win"]/total_win*100,2)
    away_prob = round(data["away_win"]/total_win*100,2)

    # Over/Under comparison
    line = data["over_line"]
    edge = round(prediction - line, 2)
    ou_direction = "OVER Value Edge" if edge > 0 else "UNDER Value Edge"

    accuracy = model_accuracy()

    context.user_data["features"] = features
    context.user_data["prediction"] = prediction

    update.message.reply_text(
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 PREMIUM TENNIS REPORT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"First Set Projection: {round(first_set,2)} games\n"
        f"Full Match Projection: {prediction} games\n"
        f"Market Line: {line}\n"
        f"Value Edge: {edge} ({ou_direction})\n\n"
        f"WIN PROBABILITY\n🏠 Home: {home_prob}%\n✈ Away: {away_prob}%\n"
        f"Model Accuracy: {accuracy}%\nLearning Samples: {sample_count}\nBias Adjustment: {round(bias,2)}\n\n"
        f"Send /result <score> after match ends"
    )

def result(update: Update, context: CallbackContext):
    if "prediction" not in context.user_data:
        update.message.reply_text("❌ No prediction found.")
        return
    try:
        actual_total = float(context.args[0])
    except:
        update.message.reply_text("Use: /result 22")
        return

    prediction = context.user_data["prediction"]
    features = context.user_data["features"]
    error = actual_total - prediction

    cursor.execute("""
    INSERT INTO matches (
        home_set1, away_set1, home_win, away_win,
        home_form, away_form, h2h_home, h2h_away,
        over_line, predicted_total, actual_total, error
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (*features[:-1], prediction, actual_total, error))
    conn.commit()

    update.message.reply_text("✅ Result stored. Model updated.")

# ---------------- MAIN ----------------
def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("result", result))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
