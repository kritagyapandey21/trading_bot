from flask import Flask
import threading, time
import trading_bot2  # your main bot file

app = Flask(__name__)

@app.route("/")
def home():
    return "🤖 Quotex Bot is Live!"

def run_bot():
    trading_bot.start_bot()  # this should be your bot’s main function

threading.Thread(target=run_bot).start()
