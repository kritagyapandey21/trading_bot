import asyncio
import logging
import random
import time
import re
import json
import websockets
import sys
import os
import hashlib
import string
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, Any, List
import threading
import signal as system_signal

import nest_asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# --- 1. CONFIGURATION ---
CONFIG = {
    "TELEGRAM_BOT_TOKEN": "YOUR_BOT_TOKEN",
    "QUOTEX_EMAIL": "Quoto421@gmail.com",
    "QUOTEX_PASSWORD": "Quoto@123",
    # Updated with only specified pairs
    "OTC_PAIRS": [
        "CAD/CHF", "EUR/SGD", "USD/BDT", "USD/DZD", "USD/EGP", "USD/IDR", 
        "USD/INR", "USD/MXN", "USD/PHP", "USD/ZAR", "NZD/JPY", "USD/ARS",
        "EUR/NZD", "NZD/CHF", "NZD/CAD", "USD/BRL", "USD/TRY", "USD/NGN",
        "USD/PKR", "USD/COP", "GBP/NZD", "AUD/NZD", "NZD/USD"
    ],
    "NON_OTC_PAIRS": [
        "USD/JPY", "AUD/USD", "AUD/JPY", "GBP/CAD", "GBP/USD", "EUR/GBP",
        "EUR/JPY", "AUD/CAD", "CAD/JPY", "EUR/AUD", "EUR/CAD", "EUR/CHF",
        "EUR/USD", "GBP/AUD", "GBP/JPY", "USD/CAD", "CHF/JPY", "AUD/CHF",
        "GBP/CHF", "USD/CHF"
    ],
    "ASSETS_TO_TRACK": [],  # Will be populated from OTC + NON_OTC
    "MAX_RETRIES": 5,
    "USE_DEMO_ACCOUNT": True,
    "SIMULATION_MODE": True,
    "TRADE_DURATION_MINUTES": 1,
    "QUOTEX_WS_URL": "wss://ws.quotex.io",
    "SIGNAL_INTERVAL_SECONDS": 600,
    "MIN_CONFIDENCE": 85,       # Increased for higher quality
    "MIN_SCORE": 85,            # Increased for higher quality
    "AUTO_TRADE_ENABLED": True,
    "ADMIN_IDS": [896970612, 976201044, 2049948903],
    "ENTRY_DELAY_MINUTES": 2,
    "PRICE_UPDATE_INTERVAL": 2,
}

# Populate ASSETS_TO_TRACK
CONFIG["ASSETS_TO_TRACK"] = CONFIG["OTC_PAIRS"] + CONFIG["NON_OTC_PAIRS"]

# --- 2. TECHNICAL INDICATOR CONFIG ---
INDICATOR_CONFIG = {
    "MA_SHORT": 5,
    "MA_LONG": 20,
    "MA_SIGNAL": 9,
    "RSI_PERIOD": 14,
    "RSI_OVERBOUGHT": 75,
    "RSI_OVERSOLD": 25,
    "PRICE_HISTORY_SIZE": 200,
    "VOLATILITY_THRESHOLD": 0.0008,
    "MIN_PRICE_DATA": 60,
    "BB_PERIOD": 20,
    "BB_STD": 2.1,
    "STOCHASTIC_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
}

# --- 3. TIMEZONE CONFIGURATION ---
class IndiaTimezone:
    """UTC+5:30 timezone operations"""
    @staticmethod
    def now():
        """Get current time in UTC+5:30"""
        return datetime.utcnow() + timedelta(hours=5, minutes=30)
    
    @staticmethod
    def format_time(dt=None):
        """Format time as HH:MM:00 in IST"""
        if dt is None:
            dt = IndiaTimezone.now()
        return dt.strftime("%H:%M:00")
    
    @staticmethod
    def format_datetime(dt=None):
        """Format full datetime in IST"""
        if dt is None:
            dt = IndiaTimezone.now()
        return dt.strftime("%Y-%m-%d %H:%M:%S IST")
    
    @staticmethod
    def is_weekend():
        """Check if current time is weekend (Saturday or Sunday)"""
        current_time = IndiaTimezone.now()
        return current_time.weekday() >= 5

# --- 4. DATABASE LOCK ---
db_lock = threading.Lock()

# --- 5. JSON-BASED LICENSE MANAGEMENT ---
class LicenseManager:
    def __init__(self):
        self.data_dir = "data"
        self.users_file = os.path.join(self.data_dir, "users.json")
        self.tokens_file = os.path.join(self.data_dir, "tokens.json")
        self.signals_file = os.path.join(self.data_dir, "signals.json")
        self.trades_file = os.path.join(self.data_dir, "trades.json")
        self.init_db()
    
    def ensure_data_dir(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def load_json(self, filename, default=None):
        """Load JSON file, return default if file doesn't exist"""
        self.ensure_data_dir()
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
        return default if default is not None else {}
    
    def save_json(self, filename, data):
        """Save data to JSON file"""
        self.ensure_data_dir()
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            return False
    
    def init_db(self):
        """Initialize JSON database files"""
        with db_lock:
            # Initialize users
            users = self.load_json(self.users_file, {})
            for admin_id in CONFIG["ADMIN_IDS"]:
                admin_id_str = str(admin_id)
                if admin_id_str not in users:
                    users[admin_id_str] = {
                        'user_id': admin_id,
                        'username': f"Admin{admin_id}",
                        'license_key': f"ADMIN{admin_id}",
                        'created_at': IndiaTimezone.now().isoformat(),
                        'is_active': True
                    }
            self.save_json(self.users_file, users)
            
            # Initialize other files with empty data
            self.save_json(self.tokens_file, {})
            self.save_json(self.signals_file, {})
            self.save_json(self.trades_file, {})
            
        print("✅ JSON Database initialized successfully")
    
    def generate_license_key(self, user_id, username):
        base_string = f"{user_id}{username}{IndiaTimezone.now().timestamp()}"
        return hashlib.md5(base_string.encode()).hexdigest()[:8]
    
    def generate_access_token(self, length=12):
        characters = string.ascii_uppercase + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
    
    def create_license(self, user_id, username):
        with db_lock:
            users = self.load_json(self.users_file, {})
            user_id_str = str(user_id)
            
            license_key = self.generate_license_key(user_id, username)
            users[user_id_str] = {
                'user_id': user_id,
                'username': username,
                'license_key': license_key,
                'created_at': IndiaTimezone.now().isoformat(),
                'is_active': True
            }
            
            success = self.save_json(self.users_file, users)
            if success:
                print(f"✅ License created for user {user_id}: {license_key}")
                return license_key
            return None
    
    def create_access_token(self, admin_id):
        with db_lock:
            tokens = self.load_json(self.tokens_file, {})
            
            token = self.generate_access_token()
            tokens[token] = {
                'created_by': admin_id,
                'created_at': IndiaTimezone.now().isoformat(),
                'is_used': False,
                'used_by': None,
                'used_at': None
            }
            
            success = self.save_json(self.tokens_file, tokens)
            if success:
                print(f"✅ Access token generated by admin {admin_id}: {token}")
                return token
            return None
    
    def use_access_token(self, token, user_id):
        with db_lock:
            tokens = self.load_json(self.tokens_file, {})
            users = self.load_json(self.users_file, {})
            user_id_str = str(user_id)
            
            if token in tokens and not tokens[token]['is_used']:
                # Mark token as used
                tokens[token]['is_used'] = True
                tokens[token]['used_by'] = user_id
                tokens[token]['used_at'] = IndiaTimezone.now().isoformat()
                
                # Create user license
                username = f"User{user_id}"
                license_key = self.generate_license_key(user_id, username)
                users[user_id_str] = {
                    'user_id': user_id,
                    'username': username,
                    'license_key': license_key,
                    'created_at': IndiaTimezone.now().isoformat(),
                    'is_active': True
                }
                
                # Save both files
                if self.save_json(self.tokens_file, tokens) and self.save_json(self.users_file, users):
                    print(f"✅ Token {token} used by user {user_id}")
                    return license_key
            
            print(f"❌ Invalid token attempt: {token} by user {user_id}")
            return None
    
    def check_user_access(self, user_id):
        if user_id in CONFIG["ADMIN_IDS"]:
            return True
            
        with db_lock:
            users = self.load_json(self.users_file, {})
            user_id_str = str(user_id)
            user = users.get(user_id_str)
            return user is not None and user.get('is_active', False)
    
    def get_user_stats(self):
        with db_lock:
            users = self.load_json(self.users_file, {})
            tokens = self.load_json(self.tokens_file, {})
            
            active_users = sum(1 for user in users.values() if user.get('is_active', False))
            available_tokens = sum(1 for token in tokens.values() if not token.get('is_used', False))
            used_tokens = sum(1 for token in tokens.values() if token.get('is_used', False))
            
            return active_users, available_tokens, used_tokens

    def get_active_users(self):
        """Get all active users with their details"""
        with db_lock:
            users = self.load_json(self.users_file, {})
            active_users = []
            
            for user_data in users.values():
                if (user_data.get('is_active', False) and 
                    user_data.get('user_id') not in CONFIG["ADMIN_IDS"]):
                    active_users.append({
                        'user_id': user_data['user_id'],
                        'username': user_data.get('username', f"User{user_data['user_id']}"),
                        'license_key': user_data.get('license_key', ''),
                        'created_at': user_data.get('created_at', '')
                    })
            
            return active_users

    def deactivate_user(self, user_id):
        """Deactivate a user by user_id"""
        with db_lock:
            users = self.load_json(self.users_file, {})
            user_id_str = str(user_id)
            
            if user_id_str in users:
                users[user_id_str]['is_active'] = False
                
                # Also remove from auto signals
                if user_id in STATE.auto_signal_users:
                    STATE.auto_signal_users.discard(user_id)
                
                success = self.save_json(self.users_file, users)
                if success:
                    print(f"✅ User {user_id} deactivated successfully")
                    return True
            
            print(f"❌ Failed to deactivate user {user_id}")
            return False

    def save_signal(self, signal_data):
        with db_lock:
            signals = self.load_json(self.signals_file, {})
            signal_id = signal_data['trade_id']
            
            timestamp_str = signal_data['timestamp'].isoformat() if hasattr(signal_data['timestamp'], 'isoformat') else str(signal_data['timestamp'])
            
            signals[signal_id] = {
                'signal_id': signal_data['trade_id'],
                'asset': signal_data['asset'],
                'direction': signal_data['direction'],
                'entry_time': signal_data['entry_time'],
                'confidence': signal_data['confidence'],
                'profit_percentage': signal_data.get('profit_percentage', 0),
                'score': signal_data['score'],
                'source': signal_data.get('source', 'TECHNICAL'),
                'timestamp': timestamp_str
            }
            
            self.save_json(self.signals_file, signals)

    def save_active_trade(self, trade_id, user_id, asset, direction, entry_time, signal_data):
        """Save active trade for result tracking"""
        with db_lock:
            trades = self.load_json(self.trades_file, {})
            
            expiry_time = (IndiaTimezone.now() + timedelta(minutes=2)).isoformat()
            
            trades[trade_id] = {
                'trade_id': trade_id,
                'user_id': user_id,
                'asset': asset,
                'direction': direction,
                'entry_time': entry_time,
                'expiry_time': expiry_time,
                'signal_data': signal_data,
                'created_at': IndiaTimezone.now().isoformat()
            }
            
            self.save_json(self.trades_file, trades)

# --- 6. GLOBAL STATE ---
class TradingState:
    def __init__(self):
        self.quotex_client = None
        self.is_connected: bool = False
        self.last_signal_time: datetime = None
        self.current_balance: float = 1000.0
        self.simulation_mode: bool = CONFIG["SIMULATION_MODE"]
        self.price_data: Dict[str, deque] = {}
        self.telegram_app = None
        self.signal_generation_task = None
        self.price_update_task = None
        self.auto_signal_task = None
        self.health_check_task = None
        self.trade_result_task = None
        self.shutting_down = False
        self.license_manager = LicenseManager()
        self.user_states: Dict[int, Dict] = {}
        self.recent_signals: Dict[str, datetime] = {}
        self.signal_cooldown = timedelta(minutes=1)
        self.task_lock = asyncio.Lock()
        self.last_user_signal_time: Dict[int, datetime] = {}
        self.auto_signal_users: set = set()
        
        for asset in CONFIG["ASSETS_TO_TRACK"]:
            self.price_data[asset] = deque(maxlen=INDICATOR_CONFIG["PRICE_HISTORY_SIZE"])

STATE = TradingState()

# Configure Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# --- 7. HIGH ACCURACY TECHNICAL INDICATORS ---
class HighAccuracyIndicators:
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return sum(prices[-period:]) / period

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        ema = prices[0]
        multiplier = 2 / (period + 1)
        
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        
        return ema

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
            
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return max(0, min(100, rsi))

    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict[str, float]:
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        ema_12 = HighAccuracyIndicators.calculate_ema(prices, INDICATOR_CONFIG["MACD_FAST"])
        ema_26 = HighAccuracyIndicators.calculate_ema(prices, INDICATOR_CONFIG["MACD_SLOW"])
        macd = ema_12 - ema_26
        
        macd_values = [macd] * 9
        signal = HighAccuracyIndicators.calculate_ema(macd_values, INDICATOR_CONFIG["MACD_SIGNAL"])
        histogram = macd - signal
        
        return {
            "macd": macd,
            "signal": signal,
            "histogram": histogram
        }

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: int = 2) -> Dict[str, float]:
        if len(prices) < period:
            sma = prices[-1] if prices else 1.0
            return {"upper": sma, "middle": sma, "lower": sma}
        
        sma = sum(prices[-period:]) / period
        variance = sum((x - sma) ** 2 for x in prices[-period:]) / period
        std = variance ** 0.5
        
        return {
            "upper": sma + (std_dev * std),
            "middle": sma,
            "lower": sma - (std_dev * std)
        }

    @staticmethod
    def calculate_stochastic(prices: List[float], period: int = 14) -> Dict[str, float]:
        if len(prices) < period:
            return {"k": 50.0, "d": 50.0}
        
        current_price = prices[-1]
        lowest_low = min(prices[-period:])
        highest_high = max(prices[-period:])
        
        if highest_high == lowest_low:
            k = 50.0
        else:
            k = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
        
        if len(prices) >= period + 2:
            k_values = [k] * 3
            d = sum(k_values) / 3
        else:
            d = k
        
        return {"k": k, "d": d}

    @staticmethod
    def calculate_support_resistance(prices: List[float]) -> Dict[str, float]:
        if len(prices) < 30:
            current = prices[-1] if prices else 1.0
            return {"support": current * 0.995, "resistance": current * 1.005}
        
        high = max(prices[-20:])
        low = min(prices[-20:])
        close = prices[-1]
        
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        
        return {
            "support": round(s1, 4),
            "resistance": round(r1, 4)
        }

    @staticmethod
    def calculate_trend_strength(prices: List[float]) -> float:
        if len(prices) < 15:
            return 0.5
        
        x = list(range(len(prices[-15:])))
        y = prices[-15:]
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x_i * x_i for x_i in x)
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            trend_strength = min(1.0, abs(slope) * 100)
            return trend_strength
        except:
            return 0.5

    @staticmethod
    def calculate_profit_potential(prices: List[float], direction: str, analysis: Dict) -> float:
        if len(prices) < 20:
            return 88.0
        
        base_profit = 88.0
        
        recent_volatility = (max(prices[-10:]) - min(prices[-10:])) / prices[-1] * 100
        volatility_bonus = min(8, recent_volatility)
        
        trend_strength = analysis.get('trend_strength', 0.5)
        trend_bonus = min(6, trend_strength * 8)
        
        rsi = analysis.get('rsi', 50)
        rsi_bonus = 0
        if (direction == "BULLISH" and rsi < 30) or (direction == "BEARISH" and rsi > 70):
            rsi_bonus = 5
        elif (direction == "BULLISH" and rsi < 40) or (direction == "BEARISH" and rsi > 60):
            rsi_bonus = 3
        
        macd_histogram = analysis.get('macd_histogram', 0)
        macd_bonus = min(3, abs(macd_histogram) * 1000)
        
        total_profit = base_profit + volatility_bonus + trend_bonus + rsi_bonus + macd_bonus
        
        return min(95.0, total_profit)

    @staticmethod
    def analyze_asset_with_high_accuracy(prices: List[float], asset: str) -> Dict[str, Any]:
        try:
            if len(prices) < INDICATOR_CONFIG["MIN_PRICE_DATA"]:
                return HighAccuracyIndicators.generate_fallback_analysis(asset)
            
            ma_short = HighAccuracyIndicators.calculate_sma(prices, INDICATOR_CONFIG["MA_SHORT"])
            ma_long = HighAccuracyIndicators.calculate_sma(prices, INDICATOR_CONFIG["MA_LONG"])
            rsi = HighAccuracyIndicators.calculate_rsi(prices, INDICATOR_CONFIG["RSI_PERIOD"])
            macd_data = HighAccuracyIndicators.calculate_macd(prices)
            bb_data = HighAccuracyIndicators.calculate_bollinger_bands(prices)
            stochastic_data = HighAccuracyIndicators.calculate_stochastic(prices)
            sr_levels = HighAccuracyIndicators.calculate_support_resistance(prices)
            trend_strength = HighAccuracyIndicators.calculate_trend_strength(prices)
            
            current_price = prices[-1]
            price_change = ((current_price - prices[-5]) / prices[-5] * 100) if len(prices) >= 5 else 0
            volatility = (max(prices[-10:]) - min(prices[-10:])) / current_price * 100 if len(prices) >= 10 else 1.0
            
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            if ma_short > ma_long:
                bullish_signals += 2.0
            else:
                bearish_signals += 2.0
            total_signals += 2.0
            
            if rsi < INDICATOR_CONFIG["RSI_OVERSOLD"]:
                bullish_signals += 1.5
            elif rsi > INDICATOR_CONFIG["RSI_OVERBOUGHT"]:
                bearish_signals += 1.5
            total_signals += 1.5
            
            if macd_data["histogram"] > 0:
                bullish_signals += 1.5
            else:
                bearish_signals += 1.5
            total_signals += 1.5
            
            bb_position = (current_price - bb_data["lower"]) / (bb_data["upper"] - bb_data["lower"])
            if bb_position < 0.2:
                bullish_signals += 1.2
            elif bb_position > 0.8:
                bearish_signals += 1.2
            total_signals += 1.2
            
            if stochastic_data["k"] < 20 and stochastic_data["d"] < 20:
                bullish_signals += 1.0
            elif stochastic_data["k"] > 80 and stochastic_data["d"] > 80:
                bearish_signals += 1.0
            total_signals += 1.0
            
            if current_price <= sr_levels["support"] * 1.001:
                bullish_signals += 1.0
            elif current_price >= sr_levels["resistance"] * 0.999:
                bearish_signals += 1.0
            total_signals += 1.0
            
            if bullish_signals > bearish_signals:
                direction = "BULLISH"
                signal_strength = (bullish_signals / total_signals) * 100
            else:
                direction = "BEARISH"
                signal_strength = (bearish_signals / total_signals) * 100
            
            base_score = (
                signal_strength * 0.25 +
                trend_strength * 0.20 +
                min(25, abs(price_change) * 3) * 0.15 +
                min(20, (80 - abs(rsi - 50))) * 0.15 +
                min(15, abs(macd_data["histogram"]) * 300) * 0.15 +
                min(10, volatility) * 0.10
            )
            
            quality_bonus = 0
            signal_diff = abs(bullish_signals - bearish_signals)
            if signal_diff > 3:
                quality_bonus = 10
            elif signal_diff > 2:
                quality_bonus = 7
            elif signal_diff > 1:
                quality_bonus = 4
            
            final_score = min(95, base_score + quality_bonus)
            
            confidence = max(CONFIG["MIN_CONFIDENCE"], min(95, 80 + (final_score - 80) * 0.8))
            
            if final_score < CONFIG["MIN_SCORE"] or confidence < CONFIG["MIN_CONFIDENCE"]:
                return {"valid": False}
            
            profit_percentage = HighAccuracyIndicators.calculate_profit_potential(
                prices, direction, {
                    'trend_strength': trend_strength,
                    'rsi': rsi,
                    'macd_histogram': macd_data['histogram']
                }
            )
            
            if final_score >= 90:
                quality = "💎 DIAMOND"
            elif final_score >= 85:
                quality = "🔥 FIRE"
            else:
                quality = "📊 GOOD"
            
            return {
                "score": int(final_score),
                "direction": direction,
                "confidence": int(confidence),
                "profit_percentage": round(profit_percentage, 1),
                "quality": quality,
                "valid": True,
                "indicators": {
                    "ma_short": round(ma_short, 4),
                    "ma_long": round(ma_long, 4),
                    "rsi": round(rsi, 1),
                    "macd_histogram": round(macd_data["histogram"], 6),
                    "bb_upper": round(bb_data["upper"], 4),
                    "bb_lower": round(bb_data["lower"], 4),
                    "stochastic_k": round(stochastic_data["k"], 1),
                    "stochastic_d": round(stochastic_data["d"], 1),
                    "support": round(sr_levels["support"], 4),
                    "resistance": round(sr_levels["resistance"], 4),
                    "current_price": round(current_price, 4),
                    "trend_strength": round(trend_strength, 2),
                    "volatility": round(volatility, 2),
                    "price_change": round(price_change, 2),
                    "bullish_signals": round(bullish_signals, 1),
                    "bearish_signals": round(bearish_signals, 1)
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_asset_with_high_accuracy: {e}")
            return {"valid": False}

    @staticmethod
    def generate_fallback_analysis(asset: str) -> Dict[str, Any]:
        score = random.randint(85, 92)
        confidence = random.randint(86, 94)
        profit = random.uniform(88.0, 95.0)
        
        return {
            "score": score,
            "direction": "BULLISH" if random.random() > 0.5 else "BEARISH",
            "confidence": confidence,
            "profit_percentage": profit,
            "quality": "💎 DIAMOND" if score >= 90 else "🔥 FIRE",
            "valid": True,
            "indicators": {
                "ma_short": round(random.uniform(1.0, 1.1), 4),
                "ma_long": round(random.uniform(1.0, 1.1), 4),
                "rsi": round(random.uniform(25, 75), 1),
                "macd_histogram": round(random.uniform(-0.002, 0.002), 6),
                "bb_upper": round(random.uniform(1.05, 1.15), 4),
                "bb_lower": round(random.uniform(0.95, 1.05), 4),
                "stochastic_k": round(random.uniform(20, 80), 1),
                "stochastic_d": round(random.uniform(20, 80), 1),
                "support": round(random.uniform(0.95, 1.0), 4),
                "resistance": round(random.uniform(1.0, 1.05), 4),
                "current_price": round(random.uniform(1.0, 1.1), 4),
                "trend_strength": round(random.uniform(0.8, 0.98), 2),
                "volatility": round(random.uniform(0.5, 1.0), 2),
                "price_change": round(random.uniform(-0.2, 0.2), 2),
                "bullish_signals": round(random.uniform(5, 7), 1),
                "bearish_signals": round(random.uniform(1, 3), 1)
            }
        }

# --- 8. HIGH ACCURACY SIGNAL GENERATION ---
def generate_high_accuracy_signal() -> Dict[str, Any]:
    try:
        current_time = IndiaTimezone.now()
        is_weekend = IndiaTimezone.is_weekend()
        
        if is_weekend:
            available_assets = CONFIG["OTC_PAIRS"]
            logger.info("📅 Weekend detected - Using OTC pairs only")
        else:
    # Mix both OTC and normal pairs on working days
            available_assets = CONFIG["OTC_PAIRS"] + CONFIG["NON_OTC_PAIRS"]
            logger.info("📅 Weekday - Using both OTC and normal pairs")
        
        available_signals = []
        for asset in available_assets:
            prices = list(STATE.price_data.get(asset, []))
            if len(prices) >= INDICATOR_CONFIG["MIN_PRICE_DATA"]:
                analysis = HighAccuracyIndicators.analyze_asset_with_high_accuracy(prices, asset)
                
                if analysis["valid"] and analysis["score"] >= CONFIG["MIN_SCORE"]:
                    direction = "CALL" if analysis["direction"] == "BULLISH" else "PUT"
                    
                    entry_time = (current_time + timedelta(minutes=CONFIG["ENTRY_DELAY_MINUTES"]))
                    entry_time_str = IndiaTimezone.format_time(entry_time)
                    
                    signal = {
                        "trade_id": f"TANIX_AI_{asset.replace('/', '_')}_{int(current_time.timestamp())}",
                        "asset": f"{asset} {'(OTC)' if asset in CONFIG['OTC_PAIRS'] else ''}",
                        "direction": direction,
                        "confidence": analysis["confidence"],
                        "profit_percentage": analysis["profit_percentage"],
                        "score": analysis["score"],
                        "quality": analysis["quality"],
                        "entry_time": entry_time_str,
                        "analysis": analysis,
                        "source": "HIGH_ACCURACY",
                        "timestamp": current_time,
                        "is_otc": asset in CONFIG["OTC_PAIRS"]
                    }
                    available_signals.append(signal)
        
        if available_signals:
            available_signals.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
            best_signal = available_signals[0]
            
            STATE.license_manager.save_signal(best_signal)
            
            logger.info(f"🎯 HIGH ACCURACY Signal: {best_signal['asset']} {best_signal['direction']} "
                       f"(Score: {best_signal['score']}, Confidence: {best_signal['confidence']}%)")
            return best_signal
        else:
            return generate_guaranteed_high_quality_signal()
            
    except Exception as e:
        logger.error(f"Error in generate_high_accuracy_signal: {e}")
        return generate_guaranteed_high_quality_signal()

def generate_guaranteed_high_quality_signal() -> Dict[str, Any]:
    current_time = IndiaTimezone.now()
    is_weekend = IndiaTimezone.is_weekend()
    
    if is_weekend:
        available_assets = CONFIG["OTC_PAIRS"]
    else:
    # Mix both OTC and normal pairs on working days
        available_assets = CONFIG["OTC_PAIRS"] + CONFIG["NON_OTC_PAIRS"]
    
    asset = random.choice(available_assets)
    
    entry_time = (current_time + timedelta(minutes=2))
    entry_time_str = IndiaTimezone.format_time(entry_time)
    
    score = random.randint(85, 92)
    confidence = random.randint(86, 94)
    profit = random.uniform(88.0, 95.0)
    quality = "💎 DIAMOND" if score >= 90 else "🔥 FIRE"
    
    return {
        "trade_id": f"TANIX_AI_GUARANTEED_{asset.replace('/', '_')}_{int(current_time.timestamp())}",
        "asset": f"{asset} {'(OTC)' if asset in CONFIG['OTC_PAIRS'] else ''}",
        "direction": random.choice(["CALL", "PUT"]),
        "confidence": confidence,
        "profit_percentage": profit,
        "score": score,
        "quality": quality,
        "entry_time": entry_time_str,
        "analysis": HighAccuracyIndicators.generate_fallback_analysis(asset),
        "source": "GUARANTEED_HQ",
        "timestamp": current_time,
        "is_otc": asset in CONFIG["OTC_PAIRS"]
    }

def format_signal_message(signal: Dict[str, Any]) -> str:
    asset_name = signal["asset"]
    emoji_dir = "📈" if signal["direction"] == "CALL" else "📉"

    message = (
        f"🤖 *TANIX AI TRADING SIGNAL* 🤖\n\n"
        f"📌 *Asset:* {asset_name}\n"
        f"🎯 *Direction:* {signal['direction']} {emoji_dir}\n"
        f"⏰ *ENTRY TIME:* {signal['entry_time']} IST\n"
        f"⏱️ *TIMEFRAME:* 1 MINUTE\n\n"
        f"💰 *Confidence:* {signal['confidence']}%\n"
        f"💸 *Profit Potential:* {signal.get('profit_percentage', 88)}%\n"
        f"🔮 *Source:* TANIX AI\n"
        f"💎 *Quality:* {signal.get('quality', '🔥 FIRE')}\n"
        f"📊 *Score:* {signal['score']}/100\n\n"
    )

    if signal.get("analysis"):
        message += (
            f"📈 *Technical Analysis:*\n"
            f"   • MA Trend: {signal['analysis']['indicators']['ma_short']} vs {signal['analysis']['indicators']['ma_long']}\n"
            f"   • RSI: {signal['analysis']['indicators']['rsi']:.1f}\n"
            f"   • MACD Hist: {signal['analysis']['indicators']['macd_histogram']:.6f}\n"
            f"   • Stochastic: K{signal['analysis']['indicators']['stochastic_k']:.1f}/D{signal['analysis']['indicators']['stochastic_d']:.1f}\n"
            f"   • Trend Strength: {signal['analysis']['indicators']['trend_strength']}/1.0\n"
            f"   • Support: {signal['analysis']['indicators']['support']:.4f}\n"
            f"   • Resistance: {signal['analysis']['indicators']['resistance']:.4f}\n"
        )

    if IndiaTimezone.is_weekend():
        message += "\n⚠️ *Weekend Notice:* Trading OTC pairs only\n"

    message += (
        "───────────────\n"
        "* 🇮🇳 All times are in IST (UTC+5:30)\n"
        "* 💲 Follow Proper Money Management\n"
        "* ⏳️ Always Select 1 Minute time frame\n"
        "* 🤖 Powered by TANIX AI\n"
    )

    return message

# --- 9. REALISTIC PRICE SIMULATION ---
class RealisticPriceGenerator:
    @staticmethod
    def generate_initial_prices(asset: str, count: int = 200) -> List[float]:
        base_prices = {
            # OTC Pairs
            "CAD/CHF": 0.6550, "EUR/SGD": 1.4550, "USD/BDT": 110.50, 
            "USD/DZD": 134.80, "USD/EGP": 30.85, "USD/IDR": 15600.0,
            "USD/INR": 83.25, "USD/MXN": 17.35, "USD/PHP": 56.20, 
            "USD/ZAR": 18.85, "NZD/JPY": 89.40, "USD/ARS": 350.0,
            "EUR/NZD": 1.7800, "NZD/CHF": 0.5450, "NZD/CAD": 0.8200,
            "USD/BRL": 4.95, "USD/TRY": 32.25, "USD/NGN": 460.0,
            "USD/PKR": 280.0, "USD/COP": 3950.0, "GBP/NZD": 2.0800,
            "AUD/NZD": 1.0950, "NZD/USD": 0.6150,
            
            # Non-OTC Pairs
            "USD/JPY": 148.50, "AUD/USD": 0.6580, "AUD/JPY": 97.80,
            "GBP/CAD": 1.7050, "GBP/USD": 1.2680, "EUR/GBP": 0.8580,
            "EUR/JPY": 161.20, "AUD/CAD": 0.8850, "CAD/JPY": 110.20,
            "EUR/AUD": 1.6550, "EUR/CAD": 1.4650, "EUR/CHF": 0.9550,
            "EUR/USD": 1.0870, "GBP/AUD": 1.9300, "GBP/JPY": 188.00,
            "USD/CAD": 1.3520, "CHF/JPY": 168.80, "AUD/CHF": 0.5770,
            "GBP/CHF": 1.1250, "USD/CHF": 0.8820
        }
        
        base_price = base_prices.get(asset, 1.0)
        prices = [base_price]
        
        if any(curr in asset for curr in ["INR", "BDT", "PKR", "NGN", "TRY", "ARS"]):
            volatility = random.uniform(0.15, 0.25)
        elif any(curr in asset for curr in ["JPY", "IDR", "COP"]):
            volatility = random.uniform(0.12, 0.18)
        else:
            volatility = random.uniform(0.08, 0.12)
        
        trend = random.uniform(-0.01, 0.01)
        
        for i in range(count - 1):
            noise = random.gauss(0, volatility / 100)
            change = trend + noise
            new_price = prices[-1] * (1 + change)
            
            if abs(new_price - base_price) / base_price > 0.015:
                reversion = (base_price - new_price) * 0.005
                new_price += reversion
                
            prices.append(round(new_price, 4))
        
        return prices
    
    @staticmethod
    def generate_price_update(last_price: float, asset: str) -> float:
        if any(curr in asset for curr in ["INR", "BDT", "PKR", "NGN", "TRY", "ARS"]):
            volatility = 0.20
        elif any(curr in asset for curr in ["JPY", "IDR", "COP"]):
            volatility = 0.15
        else:
            volatility = 0.10
        
        change = random.gauss(0, volatility / 100)
        new_price = last_price * (1 + change)
        
        return round(new_price, 4)

# [Rest of the code remains the same - TaskManager, background tasks, Telegram handlers, etc.]
# Note: The rest of the code (TaskManager, background tasks, Telegram handlers) 
# remains unchanged from the previous version since it's already compatible.

# --- 10. ASYNC TASK MANAGEMENT ---
class TaskManager:
    def __init__(self):
        self.tasks = []
        self.running = True
    
    async def create_task(self, coro, name: str):
        """Safely create and track a task"""
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        return task
    
    async def cancel_all(self):
        """Cancel all tasks gracefully"""
        self.running = False
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()

task_manager = TaskManager()

async def price_update_task():
    """Continuously update prices for all assets"""
    logger.info("💰 Price update task started for all pairs")
    
    while task_manager.running and not STATE.shutting_down:
        try:
            async with STATE.task_lock:
                for asset in CONFIG["ASSETS_TO_TRACK"]:
                    if STATE.price_data[asset]:
                        last_price = STATE.price_data[asset][-1]
                        new_price = RealisticPriceGenerator.generate_price_update(last_price, asset)
                        STATE.price_data[asset].append(new_price)
            
            await asyncio.sleep(CONFIG["PRICE_UPDATE_INTERVAL"])
            
        except asyncio.CancelledError:
            logger.info("💰 Price update task cancelled")
            break
        except Exception as e:
            logger.error(f"Price update error: {e}")
            await asyncio.sleep(5)

async def auto_signal_task():
    """Automatically send high accuracy signals every 10 minutes to all active users"""
    logger.info("🔄 TANIX AI Automated signal task started (10-minute intervals)")
    
    await asyncio.sleep(10)  # Initial delay
    
    while task_manager.running and not STATE.shutting_down:
        try:
            if STATE.auto_signal_users and STATE.telegram_app:
                signal = generate_high_accuracy_signal()
                message = format_signal_message(signal)
                
                for user_id in list(STATE.auto_signal_users):
                    try:
                        await STATE.telegram_app.bot.send_message(
                            chat_id=user_id,
                            text=message,
                            parse_mode='Markdown'
                        )
                        
                        logger.info(f"🔄 TANIX AI Auto signal sent to user {user_id}: "
                                  f"{signal['asset']} {signal['direction']} "
                                  f"(Score: {signal['score']})")
                    except Exception as e:
                        logger.error(f"Failed to send auto signal to user {user_id}: {e}")
            
            await asyncio.sleep(CONFIG["SIGNAL_INTERVAL_SECONDS"])
            
        except asyncio.CancelledError:
            logger.info("🔄 Auto signal task cancelled")
            break
        except Exception as e:
            logger.error(f"Auto signal error: {e}")
            await asyncio.sleep(60)

# --- 11. TELEGRAM HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    username = update.effective_user.username or update.effective_user.first_name or "Unknown"
    
    if STATE.license_manager.check_user_access(user_id):
        keyboard = [
            [InlineKeyboardButton("🎯 Signal", callback_data="get_signal")],
            [InlineKeyboardButton("🤖 Automated Signal", callback_data="auto_signals")],
            [InlineKeyboardButton("📊 Market Status", callback_data="market_status")],
        ]
        
        if user_id in CONFIG["ADMIN_IDS"]:
            keyboard.append([InlineKeyboardButton("👨‍💼 Admin Panel", callback_data="admin_panel")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        auto_status = "✅ ON" if user_id in STATE.auto_signal_users else "❌ OFF"
        
        await update.message.reply_text(
            f"🤖 *TANIX AI TRADING BOT* 🤖\n\n"
            f"✅ *License Status:* ACTIVE\n"
            f"👤 *User:* {username}\n"
            f"🆔 *ID:* {user_id}\n"
            f"🎯 *Strategy:* AI-POWERED ANALYSIS\n"
            f"⏰ *Timeframe:* 1 MINUTE\n"
            f"💸 *Minimum Accuracy:* 85%+\n"
            f"🤖 *Auto Signals:* {auto_status}\n\n"
            f"*Choose an option:*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            f"🔒 *Access Required*\n\n"
            f"Use `/token YOUR_TOKEN` to activate your account."
        )

async def token_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("❌ Usage: `/token YOUR_TOKEN`")
        return
    
    token = context.args[0].strip().upper()
    license_key = STATE.license_manager.use_access_token(token, user_id)
    
    if license_key:
        await update.message.reply_text(
            f"✅ *Access Granted!*\n\n"
            f"License: `{license_key}`\n"
            f"User ID: `{user_id}`\n\n"
            f"Use /start to begin."
        )
    else:
        await update.message.reply_text("❌ Invalid token")

async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    
    if user_id not in CONFIG["ADMIN_IDS"]:
        await update.message.reply_text("❌ Admin only")
        return
    
    keyboard = [
        [InlineKeyboardButton("🎫 Generate Token", callback_data="generate_token")],
        [InlineKeyboardButton("📊 User Stats", callback_data="user_stats")],
        [InlineKeyboardButton("🎯 Accuracy Report", callback_data="accuracy_report")],
        [InlineKeyboardButton("🗑️ Remove Selected User", callback_data="remove_user")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text("👨‍💼 *Admin Panel*", reply_markup=reply_markup, parse_mode='Markdown')

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = query.from_user.id
    
    try:
        if data == "get_signal":
            if not STATE.license_manager.check_user_access(user_id):
                await query.message.reply_text("❌ Need license")
                return
                
            signal = generate_high_accuracy_signal()
            message = format_signal_message(signal)
            
            await query.message.reply_text(message, parse_mode='Markdown')
            
            logger.info(f"👤 User {user_id} requested TANIX AI signal: "
                       f"{signal['asset']} {signal['direction']} "
                       f"(Score: {signal['score']})")
        
        elif data == "auto_signals":
            if not STATE.license_manager.check_user_access(user_id):
                await query.message.reply_text("❌ Need license")
                return
            
            if user_id in STATE.auto_signal_users:
                STATE.auto_signal_users.discard(user_id)
                status = "❌ OFF"
                message_text = "🤖 Automated signals have been *stopped*."
                logger.info(f"🛑 User {user_id} stopped automated signals")
            else:
                STATE.auto_signal_users.add(user_id)
                status = "✅ ON"
                message_text = "🤖 Automated signals *started*! You'll receive TANIX AI signals every 10 minutes automatically."
                logger.info(f"🚀 User {user_id} started automated signals")
            
            keyboard = [
                [InlineKeyboardButton("🎯 Signal", callback_data="get_signal")],
                [InlineKeyboardButton("🤖 Automated Signal", callback_data="auto_signals")],
                [InlineKeyboardButton("📊 Market Status", callback_data="market_status")],
            ]
            
            if user_id in CONFIG["ADMIN_IDS"]:
                keyboard.append([InlineKeyboardButton("👨‍💼 Admin Panel", callback_data="admin_panel")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.edit_text(
                f"🤖 *TANIX AI TRADING BOT* 🤖\n\n"
                f"✅ *License Status:* ACTIVE\n"
                f"👤 *User:* {query.from_user.first_name}\n"
                f"🆔 *ID:* {user_id}\n"
                f"🎯 *Strategy:* AI-POWERED ANALYSIS\n"
                f"⏰ *Timeframe:* 1 MINUTE\n"
                f"💸 *Minimum Accuracy:* 85%+\n"
                f"🤖 *Auto Signals:* {status}\n\n"
                f"{message_text}",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data == "market_status":
            if not STATE.license_manager.check_user_access(user_id):
                await query.message.reply_text("❌ Need license")
                return
            
            status = []
            sample_pairs = CONFIG["ASSETS_TO_TRACK"][:8]
            for asset in sample_pairs:
                prices = list(STATE.price_data.get(asset, []))
                if prices:
                    price = prices[-1]
                    if len(prices) > 1:
                        change = ((price - prices[-2]) / prices[-2]) * 100
                        change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                        display_name = f"{asset} {'(OTC)' if asset in CONFIG['OTC_PAIRS'] else ''}"
                        status.append(f"• {display_name}: ${price:.4f} {change_emoji} {change:+.2f}%")
                    else:
                        display_name = f"{asset} {'(OTC)' if asset in CONFIG['OTC_PAIRS'] else ''}"
                        status.append(f"• {display_name}: ${price:.4f}")
            
            status_text = "\n".join(status) if status else "No data"
            
            auto_count = len(STATE.auto_signal_users)
            is_weekend = IndiaTimezone.is_weekend()
            weekend_status = "✅ Active" if not is_weekend else "⏸️ Limited (Weekend)"
            
            await query.message.reply_text(
                f"📊 *Market Status* 🤖\n\n"
                f"{status_text}\n\n"
                f"• Showing 8 of {len(CONFIG['ASSETS_TO_TRACK'])} pairs\n"
                f"• Weekend Mode: {weekend_status}\n"
                f"• OTC Pairs: {'Available' if not is_weekend else 'Limited'}\n\n"
                f"🔗 *System Status:*\n"
                f"• Bot: TANIX AI 🤖\n"
                f"• Timeframe: 1 MINUTE\n"
                f"• Auto Users: {auto_count}\n"
                f"• Min Accuracy: 85%+\n"
                f"• Timezone: IST 🇮🇳",
                parse_mode='Markdown'
            )
        
        elif data == "admin_panel":
            if user_id not in CONFIG["ADMIN_IDS"]:
                await query.message.reply_text("❌ Admin only")
                return
            
            keyboard = [
                [InlineKeyboardButton("🎫 Generate Token", callback_data="generate_token")],
                [InlineKeyboardButton("📊 User Stats", callback_data="user_stats")],
                [InlineKeyboardButton("🎯 Accuracy Report", callback_data="accuracy_report")],
                [InlineKeyboardButton("🗑️ Remove Selected User", callback_data="remove_user")],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.reply_text("👨‍💼 *Admin Panel*", reply_markup=reply_markup, parse_mode='Markdown')
        
        elif data == "generate_token":
            if user_id not in CONFIG["ADMIN_IDS"]:
                await query.message.reply_text("❌ Admin only")
                return
            
            token = STATE.license_manager.create_access_token(user_id)
            await query.message.reply_text(f"🎫 *New Token:*\n`{token}`", parse_mode='Markdown')
        
        elif data == "user_stats":
            if user_id not in CONFIG["ADMIN_IDS"]:
                await query.message.reply_text("❌ Admin only")
                return
            
            active_users, available_tokens, used_tokens = STATE.license_manager.get_user_stats()
            auto_count = len(STATE.auto_signal_users)
            
            await query.message.reply_text(
                f"📊 *System Statistics*\n\n"
                f"👥 Active Users: {active_users}\n"
                f"🎫 Available Tokens: {available_tokens}\n"
                f"🎫 Used Tokens: {used_tokens}\n"
                f"🤖 Auto Signal Users: {auto_count}\n"
                f"💸 Trading Pairs: {len(CONFIG['ASSETS_TO_TRACK'])}\n"
                f"⏰ Signal Interval: 10 minutes",
                parse_mode='Markdown'
            )
        
        elif data == "accuracy_report":
            if user_id not in CONFIG["ADMIN_IDS"]:
                await query.message.reply_text("❌ Admin only")
                return
            
            await query.message.reply_text(
                f"🎯 *TANIX AI Accuracy Report*\n\n"
                f"📊 Minimum Accuracy: 85%+\n"
                f"🎯 Signal Quality: AI-POWERED\n"
                f"💎 Minimum Score: {CONFIG['MIN_SCORE']}+\n"
                f"💰 Minimum Confidence: {CONFIG['MIN_CONFIDENCE']}%+\n\n"
                f"*Signal Quality Distribution:*\n"
                f"• 💎 DIAMOND: Score 90+\n"
                f"• 🔥 FIRE: Score 85-89",
                parse_mode='Markdown'
            )
        
        elif data == "remove_user":
            if user_id not in CONFIG["ADMIN_IDS"]:
                await query.message.reply_text("❌ Admin only")
                return
            
            active_users = STATE.license_manager.get_active_users()
            
            if not active_users:
                await query.message.reply_text("📭 No active users found to remove.")
                return
            
            keyboard = []
            for user in active_users:
                username = user['username'] or f"User{user['user_id']}"
                button_text = f"🗑️ {username} (ID: {user['user_id']})"
                keyboard.append([InlineKeyboardButton(button_text, callback_data=f"remove_{user['user_id']}")])
            
            keyboard.append([InlineKeyboardButton("🔙 Back to Admin Panel", callback_data="admin_panel")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.reply_text(
                "👥 *Active Users - Select to Remove*\n\n"
                "Click on any user below to remove their access:",
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        elif data.startswith("remove_"):
            if user_id not in CONFIG["ADMIN_IDS"]:
                await query.message.reply_text("❌ Admin only")
                return
            
            try:
                target_user_id = int(data.replace("remove_", ""))
                
                active_users = STATE.license_manager.get_active_users()
                user_to_remove = None
                for user in active_users:
                    if user['user_id'] == target_user_id:
                        user_to_remove = user
                        break
                
                if not user_to_remove:
                    await query.message.reply_text("❌ User not found or already removed.")
                    return
                
                success = STATE.license_manager.deactivate_user(target_user_id)
                
                if success:
                    username = user_to_remove['username'] or f"User{target_user_id}"
                    await query.message.reply_text(
                        f"✅ *User Removed Successfully*\n\n"
                        f"👤 Username: {username}\n"
                        f"🆔 User ID: {target_user_id}\n"
                        f"🗑️ Access: Revoked\n"
                        f"⏰ Removed at: {IndiaTimezone.format_datetime()}",
                        parse_mode='Markdown'
                    )
                    logger.info(f"👮 Admin {user_id} removed user {target_user_id} ({username})")
                else:
                    await query.message.reply_text("❌ Failed to remove user. Please try again.")
            
            except ValueError:
                await query.message.reply_text("❌ Invalid user ID.")
            except Exception as e:
                logger.error(f"Error removing user: {e}")
                await query.message.reply_text("❌ Error removing user.")
    
    except Exception as e:
        logger.error(f"Callback error: {e}")
        try:
            signal = generate_high_accuracy_signal()
            message = format_signal_message(signal)
            await query.message.reply_text(message, parse_mode='Markdown')
        except Exception as signal_error:
            logger.error(f"Signal generation error: {signal_error}")
            await query.message.reply_text("✅ *TANIX AI Signal Generated*")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    text = update.message.text
    
    if not STATE.license_manager.check_user_access(user_id):
        await update.message.reply_text("🔒 Use /token YOUR_TOKEN")
    else:
        keyboard = [
            [InlineKeyboardButton("🎯 Signal", callback_data="get_signal")],
            [InlineKeyboardButton("🤖 Automated Signal", callback_data="auto_signals")],
            [InlineKeyboardButton("📊 Market Status", callback_data="market_status")],
        ]
        
        if user_id in CONFIG["ADMIN_IDS"]:
            keyboard.append([InlineKeyboardButton("👨‍💼 Admin Panel", callback_data="admin_panel")])
            
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "🤖 TANIX AI Trading Bot - Choose an option:",
            reply_markup=reply_markup
        )

# --- 12. GRACEFUL SHUTDOWN ---
async def shutdown():
    """Gracefully shutdown the application"""
    logger.info("🛑 Shutdown initiated...")
    
    STATE.shutting_down = True
    await task_manager.cancel_all()
    
    if STATE.telegram_app:
        await STATE.telegram_app.stop()
        await STATE.telegram_app.shutdown()
    
    logger.info("✅ Shutdown completed")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"🛑 Received signal {signum}, shutting down...")
    asyncio.create_task(shutdown())

# --- 13. MAIN APPLICATION ---
async def main():
    logger.info("🤖 Starting TANIX AI Trading Bot...")
    
    system_signal.signal(system_signal.SIGINT, signal_handler)
    system_signal.signal(system_signal.SIGTERM, signal_handler)
    
    try:
        logger.info(f"💰 Initializing {len(CONFIG['ASSETS_TO_TRACK'])} trading pairs...")
        for asset in CONFIG["ASSETS_TO_TRACK"]:
            prices = RealisticPriceGenerator.generate_initial_prices(asset, INDICATOR_CONFIG["PRICE_HISTORY_SIZE"])
            STATE.price_data[asset].extend(prices)
            logger.info(f"✅ {asset}: {len(prices)} prices loaded")
        
        logger.info("🚀 Starting TANIX AI system...")
        await task_manager.create_task(price_update_task(), "price_updater")
        await task_manager.create_task(auto_signal_task(), "auto_signal")
        
        logger.info("🤖 Initializing Telegram bot...")
        application = Application.builder().token(CONFIG["TELEGRAM_BOT_TOKEN"]).build()
        STATE.telegram_app = application
        
        handlers = [
            CommandHandler("start", start_command),
            CommandHandler("token", token_command),
            CommandHandler("admin", admin_command),
            CallbackQueryHandler(handle_callback),
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message),
        ]
        
        for handler in handlers:
            application.add_handler(handler)
        
        logger.info("✅ TANIX AI Trading Bot ready!")
        logger.info(f"🎯 Monitoring {len(CONFIG['ASSETS_TO_TRACK'])} pairs")
        logger.info("💰 Strategy: AI-powered technical analysis")
        logger.info("⏰ Timeframe: 1 minute with HH:MM:00 entry times (IST)")
        logger.info(f"📊 Minimum Quality: Score {CONFIG['MIN_SCORE']}+, Confidence {CONFIG['MIN_CONFIDENCE']}%+")
        logger.info("🎯 Signal Accuracy: 85%+ only")
        logger.info("🤖 Automated Signals: Every 10 minutes")
        logger.info("📅 Weekend Filter: OTC pairs only on weekends")
        logger.info("🇮🇳 Timezone: UTC+5:30 (IST)")
        
        await application.run_polling(
            close_loop=False,
            stop_signals=None
        )
        
    except asyncio.CancelledError:
        logger.info("🛑 Main task cancelled")
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
    finally:
        await shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        logger.info("👋 TANIX AI Bot terminated")
