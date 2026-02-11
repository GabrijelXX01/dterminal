"""
QQQ/SPY ULTIMATE TRADING TERMINAL V2.1
FIXED: UI bugs, News, Settings, Themes
Author: Gabrijel Franjicevic
Python 3.10+
"""

import threading
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button, TextBox
import matplotlib.patches as mpatches
import yfinance as yf
from datetime import datetime, timedelta
import feedparser
from scipy import stats
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

# ==================== SETTINGS ====================
class Settings:
    def __init__(self):
        self.theme = "dark"
        self.nq_conversion = False
        
        self.themes = {
            "dark": {
                "bg": "#0a0e12",
                "panel": "#161b22",
                "panel_light": "#1c2128",
                "text": "#e6edf3",
                "text_dim": "#7d8590",
                "accent": "#58a6ff",
                "bullish": "#3fb950",
                "bearish": "#f85149",
                "neutral": "#8b949e",
                "warning": "#d29922"
            },
            "light": {
                "bg": "#ffffff",
                "panel": "#f6f8fa",
                "panel_light": "#ffffff",
                "text": "#24292f",
                "text_dim": "#57606a",
                "accent": "#0969da",
                "bullish": "#1a7f37",
                "bearish": "#cf222e",
                "neutral": "#656d76",
                "warning": "#9a6700"
            },
            "navy": {
                "bg": "#001f3f",
                "panel": "#003366",
                "panel_light": "#004080",
                "text": "#ffffff",
                "text_dim": "#8fa8c7",
                "accent": "#39cccc",
                "bullish": "#2ecc40",
                "bearish": "#ff4136",
                "neutral": "#aaaaaa",
                "warning": "#ff851b"
            }
        }
    
    def get_colors(self):
        return self.themes[self.theme]

SETTINGS = Settings()

# ==================== ANALYZER ====================
class SmartAnalyzer:
    
    def __init__(self, symbol="QQQ"):
        self.symbol = symbol.upper()
        self.ticker = None
        self.lock = threading.Lock()
        
        # Price
        self.spot_price = 0
        self.price_history = []
        
        # Options
        self.iv_dict = {}
        self.id_map = {}
        self.expirations = []
        self.strikes = []
        
        # IV
        self.current_iv = 0
        self.iv_history = []
        self.iv_ema = None
        self.ema_alpha = 0.3
        self.iv_rank = 0
        self.iv_percentile = 0
        self.vix_proxy = 0
        
        # IV Environments (HIGH/LOW IV zones)
        self.high_iv_environment = []  # Strike ranges with high IV
        self.low_iv_environment = []   # Strike ranges with low IV
        
        # Levels
        self.resistance_levels = []
        self.support_levels = []
        
        # Gamma (±15 points)
        self.gamma_levels = []
        self.total_gex = 0
        self.gamma_flip = 0
        
        # Flow
        self.call_volume = 0
        self.put_volume = 0
        self.put_call_ratio = 0
        
        # VWAP
        self.vwap = 0
        self.vwap_upper = 0
        self.vwap_lower = 0
        self.vwap_position = "NEUTRAL"  # ABOVE/BELOW/AT
        
        # News
        self.news_items = []
        self.news_sentiment = 0
        self.flash_news = []
        
        # Predictions
        self.bias_now = "NEUTRAL"
        self.bias_30min = "NEUTRAL"
        self.bias_eod = "NEUTRAL"
        self.confidence = 0
        self.smart_reasoning = []
        
    def fetch_data(self):
        """Fetch data"""
        try:
            print(f"\n{'='*70}")
            print(f"  📊 LOADING {self.symbol}")
            print(f"{'='*70}\n")
            
            self.ticker = yf.Ticker(self.symbol)
            
            # Price
            hist = self.ticker.history(period="1d")
            if hist.empty:
                print(f"✗ TICKER NOT AVAILABLE: {self.symbol}")
                return False
            
            self.spot_price = hist['Close'].iloc[-1]
            print(f"✓ Spot: ${self.spot_price:.2f}")
            
            # Price history
            hist_long = self.ticker.history(period="3mo")
            self.price_history = hist_long['Close'].tolist()
            
            # Options
            self.expirations = self.ticker.options
            if not self.expirations:
                print(f"✗ No options for {self.symbol}")
                return False
            
            print(f"✓ Expirations: {len(self.expirations)}")
            
            # Historical IV
            self._fetch_historical_iv()
            
            # VWAP
            self._calculate_vwap()
            
            # News (FIXED - try harder)
            self._fetch_news_aggressive()
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def _fetch_historical_iv(self):
        """Fetch IV history"""
        try:
            hist = self.ticker.history(period="6mo")
            if len(hist) > 30:
                returns = np.log(hist['Close'] / hist['Close'].shift(1))
                
                vol_5 = returns.rolling(5).std() * np.sqrt(252)
                vol_20 = returns.rolling(20).std() * np.sqrt(252)
                vol_60 = returns.rolling(60).std() * np.sqrt(252)
                
                combined = (vol_5 * 0.2 + vol_20 * 0.5 + vol_60 * 0.3)
                self.iv_history = combined.dropna().tolist()
                
                print(f"✓ IV History: {len(self.iv_history)} points")
        except Exception as e:
            print(f"⚠️  IV History error: {e}")
    
    def _calculate_vwap(self):
        """Calculate VWAP"""
        try:
            intraday = self.ticker.history(period="1d", interval="5m")
            if len(intraday) > 0:
                tp = (intraday['High'] + intraday['Low'] + intraday['Close']) / 3
                self.vwap = (tp * intraday['Volume']).sum() / intraday['Volume'].sum()
                
                dev = tp - self.vwap
                std = np.sqrt((dev**2 * intraday['Volume']).sum() / intraday['Volume'].sum())
                
                self.vwap_upper = self.vwap + std
                self.vwap_lower = self.vwap - std
                
                # Position
                if self.spot_price > self.vwap_upper:
                    self.vwap_position = "OVERBOUGHT"
                elif self.spot_price < self.vwap_lower:
                    self.vwap_position = "OVERSOLD"
                elif self.spot_price > self.vwap:
                    self.vwap_position = "ABOVE"
                else:
                    self.vwap_position = "BELOW"
                
                print(f"✓ VWAP: ${self.vwap:.2f} | Position: {self.vwap_position}")
        except Exception as e:
            print(f"⚠️  VWAP error: {e}")
    
    def _fetch_news_aggressive(self):
        """Aggressive news fetching - FIXED"""
        try:
            self.news_items = []
            self.flash_news = []
            
            # Multiple feed URLs
            feeds = [
                f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.symbol}&region=US&lang=en-US",
                f"https://finance.yahoo.com/rss/headline?s={self.symbol}",
                f"https://news.google.com/rss/search?q={self.symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            ]
            
            flash_kw = ['breaking', 'alert', 'tariff', 'fed', 'powell', 'rate',
                       'china', 'war', 'crash', 'surge', 'emergency', 'trump',
                       'earnings', 'beat', 'miss', 'guidance']
            
            for feed_url in feeds:
                try:
                    print(f"  Trying: {feed_url[:50]}...")
                    feed = feedparser.parse(feed_url, request_headers={'User-Agent': 'Mozilla/5.0'})
                    
                    if hasattr(feed, 'entries') and feed.entries:
                        print(f"  ✓ Got {len(feed.entries)} items")
                        
                        for entry in feed.entries[:25]:
                            title = entry.get('title', '')
                            if not title:
                                continue
                            
                            title_lower = title.lower()
                            
                            # Sentiment
                            try:
                                blob = TextBlob(title)
                                sentiment = blob.sentiment.polarity
                            except:
                                sentiment = 0
                            
                            is_flash = any(kw in title_lower for kw in flash_kw)
                            
                            news = {
                                'title': title,
                                'sentiment': sentiment,
                                'is_flash': is_flash,
                                'time': entry.get('published', 'Recent')
                            }
                            
                            self.news_items.append(news)
                            
                            if is_flash:
                                self.flash_news.append(news)
                        
                        if self.news_items:
                            break  # Got news, stop
                except Exception as e:
                    print(f"  ✗ Feed failed: {e}")
                    continue
            
            # Calculate sentiment
            if self.news_items:
                sentiments = [n['sentiment'] for n in self.news_items]
                self.news_sentiment = np.mean(sentiments)
                print(f"✓ News: {len(self.news_items)} items | Sentiment: {self.news_sentiment:+.2f}")
                if self.flash_news:
                    print(f"  ⚡ Flash news: {len(self.flash_news)} items")
            else:
                print(f"⚠️  No news found (feeds might be down)")
                
        except Exception as e:
            print(f"✗ News error: {e}")
    
    def update_options(self):
        """Update options - GAMMA ±15 RANGE"""
        if not self.ticker:
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        target_exps = [exp for exp in self.expirations if exp >= today][:5]
        
        new_iv_dict = {}
        new_id_map = {}
        all_ivs = []
        atm_ivs = []
        
        gamma_by_strike = {}
        iv_by_strike = {}
        
        total_call_vol = 0
        total_put_vol = 0
        
        req_id = 1000
        
        for exp in target_exps:
            try:
                chain = self.ticker.option_chain(exp)
                calls = chain.calls
                puts = chain.puts
                
                spot = self.spot_price
                
                # GAMMA RANGE: ±15 points
                gamma_min = spot - 15
                gamma_max = spot + 15
                
                # IV SURFACE: ±10 points (tighter for display)
                surf_min = spot - 10
                surf_max = spot + 10
                
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                days = max((exp_date - datetime.now()).days, 1)
                years = days / 365.0
                
                # Calls
                for _, row in calls.iterrows():
                    strike = row['strike']
                    
                    # For surface (display)
                    if surf_min <= strike <= surf_max:
                        iv = row.get('impliedVolatility', None)
                        
                        if iv and 0 < iv < 2:
                            new_id_map[req_id] = (exp, strike, 'CALL')
                            new_iv_dict[req_id] = iv
                            all_ivs.append(iv)
                            req_id += 1
                            
                            if abs(strike - spot) < 2:
                                atm_ivs.append(iv)
                            
                            # Track IV by strike
                            if strike not in iv_by_strike:
                                iv_by_strike[strike] = []
                            iv_by_strike[strike].append(iv)
                    
                    # For gamma (wider range)
                    if gamma_min <= strike <= gamma_max:
                        iv = row.get('impliedVolatility', None)
                        vol = row.get('volume', 0) or 0
                        oi = row.get('openInterest', 0) or 0
                        
                        total_call_vol += vol
                        
                        if iv and 0 < iv < 2 and years > 0:
                            gamma = self._calc_gamma(spot, strike, years, iv)
                            gex = gamma * oi * 100 * spot * spot / 100
                            
                            if strike not in gamma_by_strike:
                                gamma_by_strike[strike] = 0
                            gamma_by_strike[strike] += gex
                
                # Puts
                for _, row in puts.iterrows():
                    strike = row['strike']
                    
                    if surf_min <= strike <= surf_max:
                        iv = row.get('impliedVolatility', None)
                        
                        if iv and 0 < iv < 2:
                            new_id_map[req_id] = (exp, strike, 'PUT')
                            new_iv_dict[req_id] = iv
                            all_ivs.append(iv)
                            req_id += 1
                            
                            if abs(strike - spot) < 2:
                                atm_ivs.append(iv)
                            
                            if strike not in iv_by_strike:
                                iv_by_strike[strike] = []
                            iv_by_strike[strike].append(iv)
                    
                    if gamma_min <= strike <= gamma_max:
                        iv = row.get('impliedVolatility', None)
                        vol = row.get('volume', 0) or 0
                        oi = row.get('openInterest', 0) or 0
                        
                        total_put_vol += vol
                        
                        if iv and 0 < iv < 2 and years > 0:
                            gamma = self._calc_gamma(spot, strike, years, iv)
                            gex = -gamma * oi * 100 * spot * spot / 100
                            
                            if strike not in gamma_by_strike:
                                gamma_by_strike[strike] = 0
                            gamma_by_strike[strike] += gex
                    
            except Exception as e:
                print(f"✗ Error {exp}: {e}")
                continue
        
        # Calculate metrics
        if all_ivs:
            self._calculate_metrics(all_ivs, atm_ivs, gamma_by_strike,
                                   iv_by_strike, total_call_vol, total_put_vol)
        
        with self.lock:
            self.iv_dict = new_iv_dict
            self.id_map = new_id_map
            if new_id_map:
                self.strikes = sorted(list(set([s for _, s, _ in new_id_map.values()])))
    
    def _calc_gamma(self, spot, strike, years, iv):
        """Gamma"""
        try:
            d1 = (np.log(spot/strike) + (0.5*iv*iv)*years) / (iv*np.sqrt(years))
            return stats.norm.pdf(d1) / (spot * iv * np.sqrt(years))
        except:
            return 0
    
    def _calculate_metrics(self, all_ivs, atm_ivs, gamma_strikes, iv_strikes,
                          call_vol, put_vol):
        """Calculate metrics"""
        
        # Current IV
        raw_iv = np.median(atm_ivs) if atm_ivs else np.median(all_ivs)
        
        if self.iv_ema is None:
            self.iv_ema = raw_iv
        else:
            self.iv_ema = self.ema_alpha * raw_iv + (1 - self.ema_alpha) * self.iv_ema
        
        self.current_iv = self.iv_ema
        
        # IV metrics
        if self.iv_history and len(self.iv_history) > 10:
            hist_min = min(self.iv_history)
            hist_max = max(self.iv_history)
            
            if hist_max > hist_min:
                self.iv_rank = ((self.current_iv - hist_min) / (hist_max - hist_min)) * 100
            
            below = sum(1 for x in self.iv_history if x < self.current_iv)
            self.iv_percentile = (below / len(self.iv_history)) * 100
        
        self.vix_proxy = (np.mean(atm_ivs) if atm_ivs else self.current_iv) * 100
        
        # Flow
        self.call_volume = call_vol
        self.put_volume = put_vol
        if call_vol > 0:
            self.put_call_ratio = put_vol / call_vol
        
        # GEX
        if gamma_strikes:
            self.total_gex = sum(gamma_strikes.values())
            
            # Top gamma levels (sorted by abs value)
            self.gamma_levels = sorted(gamma_strikes.items(), 
                                      key=lambda x: abs(x[1]), 
                                      reverse=True)[:8]
            
            # Gamma flip
            sorted_strikes = sorted(gamma_strikes.items())
            for i in range(len(sorted_strikes)-1):
                if sorted_strikes[i][1] * sorted_strikes[i+1][1] < 0:
                    self.gamma_flip = (sorted_strikes[i][0] + sorted_strikes[i+1][0]) / 2
                    break
        
        # IV ENVIRONMENTS (HIGH/LOW IV zones)
        self._detect_iv_environments(iv_strikes)
        
        # Support/Resistance
        self._calculate_levels(iv_strikes)
        
        # Predictions
        self._smart_predictions()
    
    def _detect_iv_environments(self, iv_strikes):
        """Detect HIGH IV and LOW IV environments"""
        if not iv_strikes:
            return
        
        # Average IV per strike
        avg_iv = {k: np.mean(v) for k, v in iv_strikes.items()}
        
        if not avg_iv:
            return
        
        spot = self.spot_price
        
        # Calculate mean and std
        mean_iv = np.mean(list(avg_iv.values()))
        std_iv = np.std(list(avg_iv.values()))
        
        high_threshold = mean_iv + 0.5 * std_iv
        low_threshold = mean_iv - 0.5 * std_iv
        
        # Find zones
        high_zones = []
        low_zones = []
        
        for strike, iv in sorted(avg_iv.items()):
            if iv > high_threshold:
                high_zones.append((strike, iv))
            elif iv < low_threshold:
                low_zones.append((strike, iv))
        
        # Group into ranges
        self.high_iv_environment = self._group_into_ranges(high_zones, spot)
        self.low_iv_environment = self._group_into_ranges(low_zones, spot)
    
    def _group_into_ranges(self, zones, spot):
        """Group strikes into ranges"""
        if not zones:
            return []
        
        ranges = []
        current_range = [zones[0][0]]
        
        for i in range(1, len(zones)):
            if zones[i][0] - zones[i-1][0] < 3:  # Within 3 points
                current_range.append(zones[i][0])
            else:
                if len(current_range) >= 2:
                    ranges.append((min(current_range), max(current_range)))
                current_range = [zones[i][0]]
        
        if len(current_range) >= 2:
            ranges.append((min(current_range), max(current_range)))
        
        return ranges[:3]  # Top 3 ranges
    
    def _calculate_levels(self, iv_strikes):
        """Calculate support/resistance"""
        if not iv_strikes:
            return
        
        avg_iv = {k: np.mean(v) for k, v in iv_strikes.items()}
        spot = self.spot_price
        
        mean_iv = np.mean(list(avg_iv.values()))
        std_iv = np.std(list(avg_iv.values()))
        
        high_iv_strikes = []
        for strike, iv in avg_iv.items():
            if iv > mean_iv + 0.3 * std_iv:
                high_iv_strikes.append((strike, iv))
        
        # Resistance = above spot
        self.resistance_levels = [(s, iv) for s, iv in high_iv_strikes if s > spot][:3]
        
        # Support = below spot
        self.support_levels = [(s, iv) for s, iv in high_iv_strikes if s < spot][:3]
        
        # Fallback to gamma
        if not self.resistance_levels and self.gamma_levels:
            for strike, gex in self.gamma_levels:
                if strike > spot:
                    self.resistance_levels.append((strike, 0))
                    if len(self.resistance_levels) >= 3:
                        break
        
        if not self.support_levels and self.gamma_levels:
            for strike, gex in self.gamma_levels:
                if strike < spot:
                    self.support_levels.append((strike, 0))
                    if len(self.support_levels) >= 3:
                        break
    
    def _smart_predictions(self):
        """Smart predictions"""
        self.smart_reasoning = []
        
        bull = 0
        bear = 0
        
        # Price momentum
        if len(self.price_history) > 5:
            change = (self.spot_price - self.price_history[-5]) / self.price_history[-5]
            
            if change > 0.02:
                bull += 3
                self.smart_reasoning.append("Strong upward momentum")
            elif change > 0.005:
                bull += 1
                self.smart_reasoning.append("Positive momentum")
            elif change < -0.02:
                bear += 3
                self.smart_reasoning.append("Strong downward momentum")
            elif change < -0.005:
                bear += 1
                self.smart_reasoning.append("Negative momentum")
        
        # VWAP
        if self.vwap_position == "OVERBOUGHT":
            bear += 2
            self.smart_reasoning.append("Overbought vs VWAP")
        elif self.vwap_position == "OVERSOLD":
            bull += 2
            self.smart_reasoning.append("Oversold vs VWAP")
        elif self.vwap_position == "ABOVE":
            bull += 1
        else:
            bear += 1
        
        # News
        if self.news_sentiment > 0.2:
            bull += 3
            self.smart_reasoning.append(f"Strong bullish news ({self.news_sentiment:+.2f})")
        elif self.news_sentiment > 0.05:
            bull += 1
            self.smart_reasoning.append("Positive news")
        elif self.news_sentiment < -0.2:
            bear += 3
            self.smart_reasoning.append(f"Strong bearish news ({self.news_sentiment:+.2f})")
        elif self.news_sentiment < -0.05:
            bear += 1
            self.smart_reasoning.append("Negative news")
        
        # Flash news
        if self.flash_news:
            for news in self.flash_news[:3]:
                if news['sentiment'] > 0.15:
                    bull += 2
                    self.smart_reasoning.append(f"Breaking: {news['title'][:35]}...")
                elif news['sentiment'] < -0.15:
                    bear += 2
                    self.smart_reasoning.append(f"Breaking: {news['title'][:35]}...")
        
        # IV regime
        if self.iv_rank > 70:
            bear += 2
            self.smart_reasoning.append(f"High IV (rank {self.iv_rank:.0f})")
        elif self.iv_rank < 30:
            bull += 1
        
        # GEX
        if self.gamma_flip > 0:
            if self.spot_price < self.gamma_flip:
                bear += 2
                self.smart_reasoning.append("Below gamma flip")
            else:
                bull += 1
        
        # P/C
        if self.put_call_ratio > 1.5:
            bull += 1
            self.smart_reasoning.append(f"Heavy put buying (P/C {self.put_call_ratio:.2f})")
        elif self.put_call_ratio < 0.6:
            bear += 1
        
        # Calculate
        total = bull + bear
        if total > 0:
            self.confidence = abs(bull - bear) / total * 100
        
        if bull > bear + 2:
            self.bias_now = "BULLISH"
        elif bear > bull + 2:
            self.bias_now = "BEARISH"
        else:
            self.bias_now = "NEUTRAL"
        
        # 30min
        if bull > bear + 1:
            self.bias_30min = "BULLISH"
        elif bear > bull + 1:
            self.bias_30min = "BEARISH"
        else:
            self.bias_30min = "NEUTRAL"
        
        # EOD
        if bull > bear + 1:
            self.bias_eod = "BULLISH"
        elif bear > bull + 1:
            self.bias_eod = "BEARISH"
        else:
            self.bias_eod = "NEUTRAL"


# ==================== TERMINAL ====================

class Terminal:
    def __init__(self):
        self.app = None
        self.ticker = "QQQ"
        self.paused = False
        self.show_settings = False
        
    def start(self, ticker="QQQ"):
        self.ticker = ticker.upper()
        self.app = SmartAnalyzer(self.ticker)
        
        print(f"\n{'='*70}")
        print(f"  🚀 TERMINAL V2.1")
        print(f"  by Gabrijel Franjicevic")
        print(f"{'='*70}\n")
        
        if not self.app.fetch_data():
            return False
        
        self.app.update_options()
        return True


def update_thread(terminal, interval=1):
    """Background updates"""
    count = 0
    while True:
        try:
            if not terminal.paused and terminal.app:
                app = terminal.app
                
                hist = app.ticker.history(period="1d")
                if not hist.empty:
                    app.spot_price = hist['Close'].iloc[-1]
                
                app.update_options()
                
                if count % 10 == 0:
                    app._calculate_vwap()
                
                if count % 60 == 0:
                    app._fetch_news_aggressive()
                
                count += 1
            
            time.sleep(interval)
        except:
            time.sleep(interval)


def create_ui(terminal):
    """Create UI - FIXED"""
    
    plt.ion()
    fig = plt.figure(figsize=(24, 13))
    fig.canvas.manager.set_window_title(f"Trading Terminal - {terminal.ticker}")
    
    c = SETTINGS.get_colors()
    fig.patch.set_facecolor(c['bg'])
    
    # Grid
    ax_3d = plt.subplot2grid((5, 6), (1, 0), colspan=4, rowspan=2, projection='3d')
    ax_bias_big = plt.subplot2grid((5, 6), (1, 4), colspan=2)
    ax_metrics = plt.subplot2grid((5, 6), (2, 4))
    ax_levels = plt.subplot2grid((5, 6), (2, 5))
    
    ax_gamma = plt.subplot2grid((5, 6), (3, 0), colspan=2)
    ax_vwap_panel = plt.subplot2grid((5, 6), (3, 2))
    ax_iv_env = plt.subplot2grid((5, 6), (3, 3))
    ax_predictions = plt.subplot2grid((5, 6), (3, 4), colspan=2)
    
    ax_news = plt.subplot2grid((5, 6), (4, 0), colspan=6)
    
    # Ticker + Price display (FIXED - shows price)
    ax_ticker = plt.axes([0.05, 0.96, 0.15, 0.025])
    ax_ticker.set_facecolor(c['panel'])
    ticker_box = TextBox(ax_ticker, '', initial=terminal.ticker)
    ticker_box.label.set_text("TICKER:")
    ticker_box.label.set_color(c['text'])
    ticker_box.text_disp.set_color(c['accent'])
    ticker_box.text_disp.set_fontsize(12)
    ticker_box.text_disp.set_weight('bold')
    
    def submit_ticker(text):
        if text.strip():
            terminal.paused = True
            time.sleep(0.5)
            if terminal.start(text.strip().upper()):
                ticker_box.set_val(terminal.ticker)
                terminal.paused = False
            else:
                terminal.start(terminal.ticker)
                terminal.paused = False
    
    ticker_box.on_submit(submit_ticker)
    
    # Theme toggle (WORKING)
    ax_theme = plt.axes([0.82, 0.96, 0.06, 0.025])
    btn_theme = Button(ax_theme, 'THEME', color=c['panel'], hovercolor=c['accent'])
    btn_theme.label.set_color(c['text'])
    
    def toggle_theme(event):
        themes = ["dark", "light", "navy"]
        current_idx = themes.index(SETTINGS.theme)
        SETTINGS.theme = themes[(current_idx + 1) % 3]
        print(f"Theme: {SETTINGS.theme}")
    
    btn_theme.on_clicked(toggle_theme)
    
    # Settings (WORKING - shows panel)
    ax_settings = plt.axes([0.89, 0.96, 0.06, 0.025])
    btn_settings = Button(ax_settings, 'SETTINGS', color=c['panel'], hovercolor=c['accent'])
    btn_settings.label.set_color(c['text'])
    
    def toggle_settings(event):
        terminal.show_settings = not terminal.show_settings
        print(f"Settings: {terminal.show_settings}")
    
    btn_settings.on_clicked(toggle_settings)
    
    # Pause
    ax_pause = plt.axes([0.46, 0.01, 0.06, 0.025])
    btn_pause = Button(ax_pause, 'PAUSE', color=c['panel'], hovercolor=c['accent'])
    btn_pause.label.set_color(c['bullish'])
    
    def toggle_pause(event):
        terminal.paused = not terminal.paused
        btn_pause.label.set_text("RESUME" if terminal.paused else "PAUSE")
        plt.draw()
    
    btn_pause.on_clicked(toggle_pause)
    
    # Watermark
    fig.text(0.99, 0.005, "Gabrijel Franjicevic", 
            ha='right', va='bottom', fontsize=7, 
            color=c['text'], alpha=0.3)
    
    # Thread
    update_th = threading.Thread(target=update_thread, args=(terminal, 1), daemon=True)
    update_th.start()
    
    print(f"\n{'='*70}")
    print(f"  ✓ DASHBOARD ACTIVE")
    print(f"{'='*70}\n")
    
    # Main loop
    try:
        while True:
            if not terminal.paused and terminal.app:
                app = terminal.app
                c = SETTINGS.get_colors()
                
                # Update background color
                fig.patch.set_facecolor(c['bg'])
                
                # Price display next to ticker
                fig.text(0.21, 0.975, f"${app.spot_price:.2f}", 
                        ha='left', va='center', fontsize=16, 
                        color=c['accent'], weight='bold')
                
                # Get data
                current_data = []
                with app.lock:
                    for rid in list(app.iv_dict.keys()):
                        iv = app.iv_dict[rid]
                        exp, strike, _ = app.id_map[rid]
                        current_data.append({
                            'Expiry': exp,
                            'Strike': strike,
                            'IV': iv
                        })
                
                if len(current_data) > 5:
                    df = pd.DataFrame(current_data)
                    pivot = df.pivot_table(index='Expiry', columns='Strike', values='IV')
                    pivot = pivot.sort_index().sort_index(axis=1)
                    pivot = pivot.interpolate(method='linear', limit_direction='both').bfill().ffill()
                    
                    X, Y_idx = np.meshgrid(pivot.columns, np.arange(len(pivot.index)))
                    Z = pivot.values
                    
                    elev, azim = ax_3d.elev, ax_3d.azim
                    
                    # === 3D ===
                    ax_3d.clear()
                    ax_3d.set_facecolor(c['bg'])
                    ax_3d.plot_surface(X, Y_idx, Z, cmap='viridis', 
                                      edgecolor='none', alpha=0.9, antialiased=True)
                    
                    ax_3d.set_yticks(np.arange(len(pivot.index)))
                    ax_3d.set_yticklabels(pivot.index, fontsize=7, color=c['text'])
                    ax_3d.set_title(f"{app.symbol} IV SURFACE | {time.strftime('%H:%M:%S')}", 
                                   color=c['accent'], fontsize=11, weight='bold')
                    ax_3d.view_init(elev=elev, azim=azim)
                    ax_3d.tick_params(colors=c['text_dim'], labelsize=6)
                    
                    # === BIG BIAS ===
                    ax_bias_big.clear()
                    ax_bias_big.set_facecolor(c['panel_light'])
                    ax_bias_big.axis('off')
                    
                    bias_col = c['bullish'] if app.bias_now == "BULLISH" else c['bearish'] if app.bias_now == "BEARISH" else c['neutral']
                    
                    y = 0.88
                    ax_bias_big.text(0.5, y, "BIAS", ha='center', va='top',
                                    fontsize=11, color=c['text_dim'], weight='bold',
                                    transform=ax_bias_big.transAxes)
                    
                    y -= 0.22
                    ax_bias_big.text(0.5, y, app.bias_now, ha='center', va='center',
                                    fontsize=32, color=bias_col, weight='bold',
                                    transform=ax_bias_big.transAxes)
                    
                    y -= 0.28
                    ax_bias_big.text(0.5, y, f"{app.confidence:.0f}% CONF", ha='center', va='top',
                                    fontsize=9, color=c['text_dim'],
                                    transform=ax_bias_big.transAxes)
                    
                    y -= 0.12
                    # 30min + EOD side by side
                    col_30 = c['bullish'] if app.bias_30min == "BULLISH" else c['bearish'] if app.bias_30min == "BEARISH" else c['neutral']
                    col_eod = c['bullish'] if app.bias_eod == "BULLISH" else c['bearish'] if app.bias_eod == "BEARISH" else c['neutral']
                    
                    ax_bias_big.text(0.25, y, "30M", ha='center', va='top',
                                    fontsize=7, color=c['text_dim'],
                                    transform=ax_bias_big.transAxes)
                    ax_bias_big.text(0.25, y-0.07, app.bias_30min[:4], ha='center', va='top',
                                    fontsize=9, color=col_30, weight='bold',
                                    transform=ax_bias_big.transAxes)
                    
                    ax_bias_big.text(0.75, y, "EOD", ha='center', va='top',
                                    fontsize=7, color=c['text_dim'],
                                    transform=ax_bias_big.transAxes)
                    ax_bias_big.text(0.75, y-0.07, app.bias_eod[:4], ha='center', va='top',
                                    fontsize=9, color=col_eod, weight='bold',
                                    transform=ax_bias_big.transAxes)
                    
                    # === METRICS ===
                    ax_metrics.clear()
                    ax_metrics.set_facecolor(c['panel'])
                    ax_metrics.axis('off')
                    
                    y = 0.96
                    ax_metrics.text(0.5, y, "METRICS", ha='center', va='top',
                                   fontsize=8, color=c['accent'], weight='bold',
                                   transform=ax_metrics.transAxes)
                    y -= 0.14
                    
                    metrics = [
                        ("IV", f"{app.current_iv:.1%}"),
                        ("Rank", f"{app.iv_rank:.0f}"),
                        ("VIX", f"{app.vix_proxy:.1f}"),
                        ("P/C", f"{app.put_call_ratio:.2f}"),
                    ]
                    
                    for label, val in metrics:
                        ax_metrics.text(0.1, y, label, ha='left', va='top',
                                       fontsize=7, color=c['text_dim'],
                                       transform=ax_metrics.transAxes)
                        ax_metrics.text(0.9, y, val, ha='right', va='top',
                                       fontsize=8, color=c['text'], weight='bold',
                                       transform=ax_metrics.transAxes)
                        y -= 0.12
                    
                    # === LEVELS ===
                    ax_levels.clear()
                    ax_levels.set_facecolor(c['panel'])
                    ax_levels.axis('off')
                    
                    y = 0.96
                    ax_levels.text(0.5, y, "LEVELS", ha='center', va='top',
                                  fontsize=8, color=c['accent'], weight='bold',
                                  transform=ax_levels.transAxes)
                    y -= 0.14
                    
                    ax_levels.text(0.5, y, "RESIST", ha='center', va='top',
                                  fontsize=7, color=c['bearish'], weight='bold',
                                  transform=ax_levels.transAxes)
                    y -= 0.09
                    
                    if app.resistance_levels:
                        for strike, _ in app.resistance_levels[:3]:
                            ax_levels.text(0.5, y, f"${strike:.0f}", ha='center', va='top',
                                          fontsize=7, color=c['bearish'],
                                          transform=ax_levels.transAxes)
                            y -= 0.08
                    else:
                        ax_levels.text(0.5, y, "None", ha='center', va='top',
                                      fontsize=6, color=c['text_dim'],
                                      transform=ax_levels.transAxes)
                        y -= 0.08
                    
                    y -= 0.04
                    ax_levels.text(0.5, y, "SUPPORT", ha='center', va='top',
                                  fontsize=7, color=c['bullish'], weight='bold',
                                  transform=ax_levels.transAxes)
                    y -= 0.09
                    
                    if app.support_levels:
                        for strike, _ in app.support_levels[:3]:
                            ax_levels.text(0.5, y, f"${strike:.0f}", ha='center', va='top',
                                          fontsize=7, color=c['bullish'],
                                          transform=ax_levels.transAxes)
                            y -= 0.08
                    else:
                        ax_levels.text(0.5, y, "None", ha='center', va='top',
                                      fontsize=6, color=c['text_dim'],
                                      transform=ax_levels.transAxes)
                    
                    # === GAMMA (FIXED - easier to read) ===
                    ax_gamma.clear()
                    ax_gamma.set_facecolor(c['panel'])
                    ax_gamma.set_title("GAMMA (±15pts)", color=c['text'], fontsize=9, weight='bold', pad=4)
                    
                    if app.gamma_levels:
                        strikes = [s for s, g in app.gamma_levels[:8]]
                        gex = [g/1e9 for s, g in app.gamma_levels[:8]]
                        colors_g = [c['bullish'] if g > 0 else c['bearish'] for g in gex]
                        
                        ax_gamma.barh(range(len(strikes)), gex, color=colors_g, alpha=0.8, height=0.6)
                        ax_gamma.set_yticks(range(len(strikes)))
                        ax_gamma.set_yticklabels([f"${s:.0f}" for s in strikes], fontsize=7, color=c['text'])
                        ax_gamma.axvline(0, color=c['text_dim'], linewidth=1)
                        ax_gamma.set_xlabel('GEX (B)', fontsize=7, color=c['text'])
                        ax_gamma.grid(True, alpha=0.15, color=c['text'], axis='x')
                        ax_gamma.tick_params(colors=c['text_dim'], labelsize=6)
                    
                    # === VWAP PANEL (FIXED - shows numbers) ===
                    ax_vwap_panel.clear()
                    ax_vwap_panel.set_facecolor(c['panel'])
                    ax_vwap_panel.axis('off')
                    
                    y = 0.96
                    ax_vwap_panel.text(0.5, y, "VWAP", ha='center', va='top',
                                      fontsize=8, color=c['accent'], weight='bold',
                                      transform=ax_vwap_panel.transAxes)
                    y -= 0.16
                    
                    vwap_data = [
                        ("VWAP", f"${app.vwap:.2f}"),
                        ("Upper", f"${app.vwap_upper:.2f}"),
                        ("Lower", f"${app.vwap_lower:.2f}"),
                        ("Status", app.vwap_position),
                    ]
                    
                    for label, val in vwap_data:
                        ax_vwap_panel.text(0.05, y, label, ha='left', va='top',
                                          fontsize=7, color=c['text_dim'],
                                          transform=ax_vwap_panel.transAxes)
                        
                        val_color = c['text']
                        if label == "Status":
                            if "OVER" in val:
                                val_color = c['bearish']
                            elif "UNDER" in val or val == "OVERSOLD":
                                val_color = c['bullish']
                        
                        ax_vwap_panel.text(0.95, y, val, ha='right', va='top',
                                          fontsize=7, color=val_color, weight='bold',
                                          transform=ax_vwap_panel.transAxes)
                        y -= 0.13
                    
                    # === IV ENVIRONMENTS (NEW!) ===
                    ax_iv_env.clear()
                    ax_iv_env.set_facecolor(c['panel'])
                    ax_iv_env.axis('off')
                    
                    y = 0.96
                    ax_iv_env.text(0.5, y, "IV ZONES", ha='center', va='top',
                                  fontsize=8, color=c['accent'], weight='bold',
                                  transform=ax_iv_env.transAxes)
                    y -= 0.16
                    
                    ax_iv_env.text(0.5, y, "HIGH IV", ha='center', va='top',
                                  fontsize=7, color=c['bearish'], weight='bold',
                                  transform=ax_iv_env.transAxes)
                    y -= 0.10
                    
                    if app.high_iv_environment:
                        for low, high in app.high_iv_environment[:2]:
                            ax_iv_env.text(0.5, y, f"${low:.0f}-${high:.0f}", ha='center', va='top',
                                          fontsize=6, color=c['bearish'],
                                          transform=ax_iv_env.transAxes)
                            y -= 0.09
                    else:
                        ax_iv_env.text(0.5, y, "None", ha='center', va='top',
                                      fontsize=6, color=c['text_dim'],
                                      transform=ax_iv_env.transAxes)
                        y -= 0.09
                    
                    y -= 0.06
                    ax_iv_env.text(0.5, y, "LOW IV", ha='center', va='top',
                                  fontsize=7, color=c['bullish'], weight='bold',
                                  transform=ax_iv_env.transAxes)
                    y -= 0.10
                    
                    if app.low_iv_environment:
                        for low, high in app.low_iv_environment[:2]:
                            ax_iv_env.text(0.5, y, f"${low:.0f}-${high:.0f}", ha='center', va='top',
                                          fontsize=6, color=c['bullish'],
                                          transform=ax_iv_env.transAxes)
                            y -= 0.09
                    else:
                        ax_iv_env.text(0.5, y, "None", ha='center', va='top',
                                      fontsize=6, color=c['text_dim'],
                                      transform=ax_iv_env.transAxes)
                    
                    # === PREDICTIONS ===
                    ax_predictions.clear()
                    ax_predictions.set_facecolor(c['panel'])
                    ax_predictions.axis('off')
                    
                    y = 0.96
                    ax_predictions.text(0.5, y, "ANALYSIS", ha='center', va='top',
                                       fontsize=8, color=c['accent'], weight='bold',
                                       transform=ax_predictions.transAxes)
                    y -= 0.14
                    
                    for reason in app.smart_reasoning[:7]:
                        ax_predictions.text(0.03, y, f"• {reason}", ha='left', va='top',
                                           fontsize=6, color=c['text'],
                                           transform=ax_predictions.transAxes, wrap=True)
                        y -= 0.10
                    
                    if not app.smart_reasoning:
                        ax_predictions.text(0.5, 0.5, "Analyzing...", ha='center', va='center',
                                           fontsize=7, color=c['text_dim'],
                                           transform=ax_predictions.transAxes)
                    
                    # === NEWS (FIXED) ===
                    ax_news.clear()
                    ax_news.set_facecolor(c['panel'])
                    ax_news.axis('off')
                    
                    y = 0.94
                    ax_news.text(0.5, y, "NEWS FEED", ha='center', va='top',
                                fontsize=9, color=c['accent'], weight='bold',
                                transform=ax_news.transAxes)
                    y -= 0.16
                    
                    if app.flash_news:
                        ax_news.text(0.01, y, "⚡ BREAKING:", ha='left', va='top',
                                    fontsize=8, color=c['bearish'], weight='bold',
                                    transform=ax_news.transAxes)
                        y -= 0.14
                        
                        for news in app.flash_news[:2]:
                            title = news['title'][:85] + '...' if len(news['title']) > 85 else news['title']
                            col = c['bullish'] if news['sentiment'] > 0.1 else c['bearish'] if news['sentiment'] < -0.1 else c['text']
                            
                            ax_news.text(0.01, y, f"• {title}", ha='left', va='top',
                                        fontsize=6, color=col,
                                        transform=ax_news.transAxes)
                            y -= 0.11
                        
                        y -= 0.04
                    
                    if app.news_items:
                        ax_news.text(0.01, y, f"Latest ({len(app.news_items)} items):", ha='left', va='top',
                                    fontsize=7, color=c['text_dim'],
                                    transform=ax_news.transAxes)
                        y -= 0.14
                        
                        for news in app.news_items[:6]:
                            if news in app.flash_news:
                                continue
                            
                            title = news['title'][:85] + '...' if len(news['title']) > 85 else news['title']
                            col = c['bullish'] if news['sentiment'] > 0.1 else c['bearish'] if news['sentiment'] < -0.1 else c['text']
                            emoji = '🟢' if news['sentiment'] > 0.1 else '🔴' if news['sentiment'] < -0.1 else '⚪'
                            
                            ax_news.text(0.01, y, f"{emoji} {title}", ha='left', va='top',
                                        fontsize=6, color=col,
                                        transform=ax_news.transAxes)
                            y -= 0.11
                    else:
                        ax_news.text(0.5, 0.5, "No news available (feeds may be down)", ha='center', va='center',
                                    fontsize=7, color=c['text_dim'],
                                    transform=ax_news.transAxes)
                    
                    # Settings overlay
                    if terminal.show_settings:
                        ax_news.text(0.5, 0.3, "⚙️ SETTINGS", ha='center', va='center',
                                    fontsize=14, color=c['accent'], weight='bold',
                                    transform=ax_news.transAxes,
                                    bbox=dict(boxstyle='round', facecolor=c['panel_light'], alpha=0.95, pad=0.5))
                        
                        ax_news.text(0.5, 0.15, f"Theme: {SETTINGS.theme.upper()}", ha='center', va='center',
                                    fontsize=9, color=c['text'],
                                    transform=ax_news.transAxes)
                        
                        ax_news.text(0.5, 0.05, "Click SETTINGS again to close", ha='center', va='center',
                                    fontsize=7, color=c['text_dim'],
                                    transform=ax_news.transAxes)
            
            plt.pause(0.15)
    
    except KeyboardInterrupt:
        plt.close()
        print("\n✓ Terminal closed")


# ==================== MAIN ====================

if __name__ == '__main__':
    terminal = Terminal()
    
    if terminal.start("QQQ"):
        print("\n🚀 Starting in 2 seconds...\n")
        time.sleep(2)
        create_ui(terminal)
    else:
        print("\n✗ Failed to start")
