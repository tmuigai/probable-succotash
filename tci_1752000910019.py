import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import time
import logging
import os
from dotenv import load_dotenv
import schedule
import redis
from fastapi import FastAPI
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from alpha_vantage.forex import Forex
from finnhub import Client as FinnhubClient
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange
import pytz
import backtrader as bt
import logging
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
import warnings
from typing import Optional, Union
import pandas as pd
import dask.dataframe as dd
import logging
import plotly.graph_objects as go
from typing import Dict, Callable, Optional, Tuple
from pandas.tseries.offsets import CustomBusinessHour

# === CONFIGURATION ===
load_dotenv()
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "your_app_id")
DERIV_TOKEN = os.getenv("DERIV_TOKEN", "your_api_token")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "your_alpha_vantage_key")
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "your_finnhub_key")
SYMBOL = "frxEURUSD"
LOT_SIZE = 1.00
MAX_TICKS = 5000
RISK_REWARD_RATIO = 2.0
MAX_TRADES_PER_DAY = 5
SESSION_FILTER = True
TRADE_EXECUTION = False  # Set to True for live trading
MIN_SWING_PIPS = 40
PO3_LOOKBACK = 20
OTE_MIN_RETRACEMENT = 0.62
OTE_MAX_RETRACEMENT = 0.79
OTE_SWEET_SPOT = 0.705
MIN_FVG_SIZE = 0.0003
INSTITUTIONAL_LEVELS = [0.0000, 0.0020, 0.0050, 0.0080]
KILL_ZONES = {
    'Asian': (0, 5),      # 00:00-05:00 UTC
    'London_Open': (2, 5), # 02:00-05:00 UTC
    'NY_AM': (7, 10),     # 07:00-10:00 UTC
    'NY_PM': (13, 16)     # 13:00-16:00 UTC
}
NO_TRADE_WINDOWS = ['news', 'low_volume']  # Placeholder for news API

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ict_smart_money_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === DATA STRUCTURES ===
ticks_data = deque(maxlen=MAX_TICKS)
trade_history = []
daily_stats = {
    'trades_today': 0,
    'last_trade_time': None,
    'wins': 0,
    'losses': 0,
    'max_drawdown': 0
}
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# === DATA FETCHING ===
def get_alpha_vantage_data(symbol, timeframe='daily'):
    """Fetch HTF data from Alpha Vantage (Compendium Section 12)."""
    try:
        forex = Forex(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        data, _ = forex.get_currency_exchange_intraday(
            from_symbol=symbol[:3], to_symbol=symbol[3:], interval=timeframe
        )
        data.columns = ['open', 'high', 'low', 'close']
        data['time'] = data.index
        data.reset_index(drop=True, inplace=True)
        return data
    except Exception as e:
        logger.error(f"Alpha Vantage data fetch failed: {str(e)}")
        return pd.DataFrame()

def get_finnhub_data(symbol, timeframe='1'):
    """Fetch LTF data from Finnhub (Compendium Section 12)."""
    try:
        finnhub_client = FinnhubClient(api_key=FINNHUB_KEY)
        now = int(time.time())
        start = now - 3600 * 24  # Last 24 hours
        data = finnhub_client.forex_candles(
            symbol=symbol, resolution=timeframe, _from=start, to=now
        )
        df = pd.DataFrame({
            'time': pd.to_datetime(data['t'], unit='s'),
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        return df
    except Exception as e:
        logger.error(f"Finnhub data fetch failed: {str(e)}")
        return pd.DataFrame()

def convert_ticks_to_ohlc(ticks, timeframe='1min'):
    """Convert Deriv tick data to OHLC (Compendium Section 11)."""
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    ohlc = df['bid'].resample(timeframe).ohlc()
    ohlc['ask'] = df['ask'].resample(timeframe).last()
    ohlc['volume'] = df['volume'].resample(timeframe).sum() if 'volume' in df else 0
    return ohlc.reset_index()

# === ICT CORE FUNCTIONS ===
logger = logging.getLogger(__name__)

def detect_internal_structure(
    df_m1: pd.DataFrame,
    m15_index: pd.DatetimeIndex,
    swing_lookback: int = 3,
    atr_period: int = 14,
    min_swing_atr: float = 0.5
) -> pd.DataFrame:
    """
    Detect M1 internal structure and Internal Order Blocks within an M15 leg.

    Parameters:
    -----------
    df_m1 : pd.DataFrame
        M1 DataFrame with OHLC columns.
    m15_index : pd.DatetimeIndex
        Index of the M15 candle for analysis.
    swing_lookback : int, optional (default=3)
        Bars for swing high/low detection.
    atr_period : int, optional (default=14)
        Period for ATR calculation.
    min_swing_atr : float, optional (default=0.5)
        Minimum swing size as fraction of ATR.

    Returns:
    --------
    pd.DataFrame
        M1 DataFrame with columns:
        - internal_swing_high, internal_swing_low: bool, M1 swing points
        - internal_ob_bullish, internal_ob_bearish: bool, Internal Order Blocks
        - internal_structure: str, 'bullish', 'bearish', 'neutral'
    """
    df_m1 = df_m1[(df_m1.index >= m15_index) & (df_m1.index < m15_index + pd.Timedelta(minutes=15))].copy()
    if len(df_m1) < 5:
        logger.debug(f"Insufficient M1 data for {m15_index}")
        return df_m1

    df_m1['internal_swing_high'] = False
    df_m1['internal_swing_low'] = False
    df_m1['internal_ob_bullish'] = False
    df_m1['internal_ob_bearish'] = False
    df_m1['internal_structure'] = 'neutral'

    atr = AverageTrueRange(df_m1['high'], df_m1['low'], df_m1['close'], window=atr_period).average_true_range()

    df_m1['internal_swing_high'] = (
        (df_m1['high'] > df_m1['high'].shift(1)) &
        (df_m1['high'] > df_m1['high'].shift(-1)) &
        (df_m1['high'] - pd.concat([df_m1['high'].shift(1), df_m1['high'].shift(-1)], axis=1).max(axis=1) > atr * min_swing_atr)
    )
    df_m1['internal_swing_low'] = (
        (df_m1['low'] < df_m1['low'].shift(1)) &
        (df_m1['low'] < df_m1['low'].shift(-1)) &
        (pd.concat([df_m1['low'].shift(1), df_m1['low'].shift(-1)], axis=1).min(axis=1) - df_m1['low'] > atr * min_swing_atr)
    )

    # Internal Order Blocks
    for i in range(2, len(df_m1)):
        if (df_m1['close'].iloc[i-1] < df_m1['open'].iloc[i-1] and
            df_m1['close'].iloc[i] > df_m1['high'].iloc[i-1]):
            df_m1.at[df_m1.index[i-1], 'internal_ob_bullish'] = True
        if (df_m1['close'].iloc[i-1] > df_m1['open'].iloc[i-1] and
            df_m1['close'].iloc[i] < df_m1['low'].iloc[i-1]):
            df_m1.at[df_m1.index[i-1], 'internal_ob_bearish'] = True

    # Internal Structure Bias
    if len(df_m1) >= 5:
        for i in range(5, len(df_m1)):
            if (df_m1['high'].iloc[i] > df_m1['high'].iloc[i-5] and
                df_m1['low'].iloc[i] > df_m1['low'].iloc[i-5]):
                df_m1.at[df_m1.index[i], 'internal_structure'] = 'bullish'
            elif (df_m1['high'].iloc[i] < df_m1['high'].iloc[i-5] and
                  df_m1['low'].iloc[i] < df_m1['low'].iloc[i-5]):
                df_m1.at[df_m1.index[i], 'internal_structure'] = 'bearish'

    return df_m1

def detect_market_structure(
    df: pd.DataFrame,
    swing_lookback: int = 3,
    atr_multiplier: float = 2.0,
    structure_lookback: int = 20,
    atr_period: int = 14,
    min_swing_atr: float = 0.5,
    use_volume: bool = False,
    volume_threshold: float = 1.5,
    use_rsi: bool = False,
    rsi_period: int = 14,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    fvg_threshold: float = 0.5,
    df_m1: pd.DataFrame = None,
    df_htf: pd.DataFrame = None,
    plot: bool = False,
    ohlc_columns: dict = None
) -> pd.DataFrame:
    """
    Detect market structure patterns and all ICT Order Blocks as per Compendium Section 1.

    Parameters:
    -----------
    df : pd.DataFrame
        M15 DataFrame with OHLC columns and optionally 'volume'.
    swing_lookback : int, optional (default=3)
        Bars for swing high/low detection.
    atr_multiplier : float, optional (default=2.0)
        ATR multiplier for displacement.
    structure_lookback : int, optional (default=20)
        Recent bars for market structure bias.
    atr_period : int, optional (default=14)
        Period for ATR calculation.
    min_swing_atr : float, optional (default=0.5)
        Minimum swing size as fraction of ATR.
    use_volume : bool, optional (default=False)
        Require high volume for signals.
    volume_threshold : float, optional (default=1.5)
        Volume multiplier for signal confirmation.
    use_rsi : bool, optional (default=False)
        Use RSI for signal confirmation.
    rsi_period : int, optional (default=14)
        Period for RSI calculation.
    rsi_overbought : float, optional (default=70)
        RSI level for overbought condition.
    rsi_oversold : float, optional (default=30)
        RSI level for oversold condition.
    fvg_threshold : float, optional (default=0.5)
        ATR fraction for fair value gap detection.
    df_m1 : pd.DataFrame, optional (default=None)
        M1 DataFrame for internal structure analysis.
    df_htf : pd.DataFrame, optional (default=None)
        Higher timeframe (e.g., D1/W1) DataFrame for HTF OB detection.
    plot : bool, optional (default=False)
        Generate Plotly chart of results.
    ohlc_columns : dict, optional (default=None)
        Mapping of OHLC column names (e.g., {'open': 'Open', 'high': 'High'}).

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - swing_high, swing_low, ith, itl, bos, choch, mss, true_break, displacement
        - bullish_ob, bearish_ob, breaker_block, mitigation_block, refined_ob_bullish, refined_ob_bearish
        - ob_chain_bullish, ob_chain_bearish, continuation_ob_bullish, continuation_ob_bearish
        - inverse_ob_bullish, inverse_ob_bearish, internal_ob_bullish, internal_ob_bearish
        - htf_ob_bullish, htf_ob_bearish, institutional_candle, liquidity_ob_bullish, liquidity_ob_bearish
        - engulfing_ob_bullish, engulfing_ob_bearish
        - market_structure, internal_structure, signal
    """
    # Handle custom OHLC column names
    if ohlc_columns is None:
        ohlc_columns = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    required_columns = [ohlc_columns[col]ხ]


logger = logging.getLogger(__name__)

def detect_internal_structure(
    df_m1: pd.DataFrame,
    m15_index: pd.DatetimeIndex,
    swing_lookback: int = 3,
    atr_period: int = 14,
    min_swing_atr: float = 0.5
) -> pd.DataFrame:
    """
    Detect M1 internal structure and Internal Order Blocks within an M15 leg.

    Parameters:
    -----------
    df_m1 : pd.DataFrame
        M1 DataFrame with OHLC columns.
    m15_index : pd.DatetimeIndex
        Index of the M15 candle for analysis.
    swing_lookback : int, optional (default=3)
        Bars for swing high/low detection.
    atr_period : int, optional (default=14)
        Period for ATR calculation.
    min_swing_atr : float, optional (default=0.5)
        Minimum swing size as fraction of ATR.

    Returns:
    --------
    pd.DataFrame
        M1 DataFrame with columns:
        - internal_swing_high, internal_swing_low: bool, M1 swing points
        - internal_ob_bullish, internal_ob_bearish: bool, Internal Order Blocks
        - internal_structure: str, 'bullish', 'bearish', 'neutral'
    """
    df_m1 = df_m1[(df_m1.index >= m15_index) & (df_m1.index < m15_index + pd.Timedelta(minutes=15))].copy()
    if len(df_m1) < 5:
        logger.debug(f"Insufficient M1 data for {m15_index}")
        return df_m1

    df_m1['internal_swing_high'] = False
    df_m1['internal_swing_low'] = False
    df_m1['internal_ob_bullish'] = False
    df_m1['internal_ob_bearish'] = False
    df_m1['internal_structure'] = 'neutral'

    atr = AverageTrueRange(df_m1['high'], df_m1['low'], df_m1['close'], window=atr_period).average_true_range()

    df_m1['internal_swing_high'] = (
        (df_m1['high'] > df_m1['high'].shift(1)) &
        (df_m1['high'] > df_m1['high'].shift(-1)) &
        (df_m1['high'] - pd.concat([df_m1['high'].shift(1), df_m1['high'].shift(-1)], axis=1).max(axis=1) > atr * min_swing_atr)
    )
    df_m1['internal_swing_low'] = (
        (df_m1['low'] < df_m1['low'].shift(1)) &
        (df_m1['low'] < df_m1['low'].shift(-1)) &
        (pd.concat([df_m1['low'].shift(1), df_m1['low'].shift(-1)], axis=1).min(axis=1) - df_m1['low'] > atr * min_swing_atr)
    )

    # Internal Order Blocks
    for i in range(2, len(df_m1)):
        if (df_m1['close'].iloc[i-1] < df_m1['open'].iloc[i-1] and
            df_m1['close'].iloc[i] > df_m1['high'].iloc[i-1]):
            df_m1.at[df_m1.index[i-1], 'internal_ob_bullish'] = True
        if (df_m1['close'].iloc[i-1] > df_m1['open'].iloc[i-1] and
            df_m1['close'].iloc[i] < df_m1['low'].iloc[i-1]):
            df_m1.at[df_m1.index[i-1], 'internal_ob_bearish'] = True

    # Internal Structure Bias
    if len(df_m1) >= 5:
        for i in range(5, len(df_m1)):
            if (df_m1['high'].iloc[i] > df_m1['high'].iloc[i-5] and
                df_m1['low'].iloc[i] > df_m1['low'].iloc[i-5]):
                df_m1.at[df_m1.index[i], 'internal_structure'] = 'bullish'
            elif (df_m1['high'].iloc[i] < df_m1['high'].iloc[i-5] and
                  df_m1['low'].iloc[i] < df_m1['low'].iloc[i-5]):
                df_m1.at[df_m1.index[i], 'internal_structure'] = 'bearish'

    return df_m1

def detect_market_structure(
    df: pd.DataFrame,
    swing_lookback: int = 3,
    atr_multiplier: float = 2.0,
    structure_lookback: int = 20,
    atr_period: int = 14,
    min_swing_atr: float = 0.5,
    use_volume: bool = False,
    volume_threshold: float = 1.5,
    use_rsi: bool = False,
    rsi_period: int = 14,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    fvg_threshold: float = 0.5,
    df_m1: pd.DataFrame = None,
    df_htf: pd.DataFrame = None,
    correlated_asset: pd.DataFrame = None,
    plot: bool = False,
    ohlc_columns: dict = None
) -> pd.DataFrame:
    """
    Detect market structure patterns and all ICT Order Blocks as per Compendium Section 1.

    Parameters:
    -----------
    df : pd.DataFrame
        M15 DataFrame with OHLC columns and optionally 'volume'.
    swing_lookback : int, optional (default=3)
        Bars for swing high/low detection.
    atr_multiplier : float, optional (default=2.0)
        ATR multiplier for displacement.
    structure_lookback : int, optional (default=20)
        Recent bars for market structure bias.
    atr_period : int, optional (default=14)
        Period for ATR calculation.
    min_swing_atr : float, optional (default=0.5)
        Minimum swing size as fraction of ATR.
    use_volume : bool, optional (default=False)
        Require high volume for signals.
    volume_threshold : float, optional (default=1.5)
        Volume multiplier for signal confirmation.
    use_rsi : bool, optional (default=False)
        Use RSI for signal confirmation.
    rsi_period : int, optional (default=14)
        Period for RSI calculation.
    rsi_overbought : float, optional (default=70)
        RSI level for overbought condition.
    rsi_oversold : float, optional (default=30)
        RSI level for oversold condition.
    fvg_threshold : float, optional (default=0.5)
        ATR fraction for fair value gap detection.
    df_m1 : pd.DataFrame, optional (default=None)
        M1 DataFrame for internal structure analysis.
    df_htf : pd.DataFrame, optional (default=None)
        Higher timeframe (e.g., D1/W1) DataFrame for HTF OB detection.
    correlated_asset : pd.DataFrame, optional (default=None)
        DataFrame for correlated asset to detect SMT divergence.
    plot : bool, optional (default=False)
        Generate Plotly chart of results.
    ohlc_columns : dict, optional (default=None)
        Mapping of OHLC column names (e.g., {'open': 'Open', 'high': 'High'}).

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns for swing points, BOS, CHoCH, MSS, true break, displacement,
        all 13 ICT Order Block types, market structure, internal structure, and signals.
    """
    # Handle custom OHLC column names
    if ohlc_columns is None:
        ohlc_columns = {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}
    required_columns = [ohlc_columns[col] for col in ['open', 'high', 'low', 'close']]
    optional_columns = [ohlc_columns['volume']] if use_volume else []

    # Input validation
    if len(df) < max(5, atr_period, rsi_period):
        logger.warning(f"Insufficient data: at least {max(5, atr_period, rsi_period)} rows required")
        return df
    if not all(col in df.columns for col in required_columns + optional_columns):
        logger.error(f"Missing required columns: {required_columns + optional_columns}")
        return df
    if not df[required_columns].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
        logger.error("OHLC columns must be numeric")
        return df
    if df[required_columns].isna().any().any():
        logger.warning("NaN values in OHLC columns; filling with forward fill")
        df = df.fillna(method='ffill')

    # Initialize DataFrame
    df = df.copy()
    ob_types = [
        'bullish_ob', 'bearish_ob', 'breaker_block', 'mitigation_block',
        'refined_ob_bullish', 'refined_ob_bearish', 'ob_chain_bullish', 'ob_chain_bearish',
        'continuation_ob_bullish', 'continuation_ob_bearish', 'inverse_ob_bullish', 'inverse_ob_bearish',
        'internal_ob_bullish', 'internal_ob_bearish', 'htf_ob_bullish', 'htf_ob_bearish',
        'institutional_candle', 'liquidity_ob_bullish', 'liquidity_ob_bearish',
        'engulfing_ob_bullish', 'engulfing_ob_bearish'
    ]
    for col in ['swing_high', 'swing_low', 'ith', 'itl', 'bos', 'choch', 'mss', 'true_break', 'displacement'] + ob_types:
        df[col] = False
    df['market_structure'] = 'neutral'
    df['internal_structure'] = 'neutral'
    df['signal'] = 'hold'
    df['smt_divergence'] = False

    # Map OHLC columns
    df_ohlc = df[required_columns + optional_columns].rename(columns={
        ohlc_columns['open']: 'open',
        ohlc_columns['high']: 'high',
        ohlc_columns['low']: 'low',
        ohlc_columns['close']: 'close',
        ohlc_columns.get('volume', 'volume'): 'volume'
    })

    # Calculate ATR and RSI
    atr = AverageTrueRange(df_ohlc['high'], df_ohlc['low'], df_ohlc['close'], window=atr_period).average_true_range()
    rsi = RSIIndicator(df_ohlc['close'], window=rsi_period).rsi() if use_rsi else pd.Series(np.nan, index=df.index)

    # Swing High/Low with ATR filter
    df['swing_high'] = (
        (df_ohlc['high'] > df_ohlc['high'].shift(1)) &
        (df_ohlc['high'] > df_ohlc['high'].shift(-1)) &
        (df_ohlc['high'] - pd.concat([df_ohlc['high'].shift(1), df_ohlc['high'].shift(-1)], axis=1).max(axis=1) > atr * min_swing_atr)
    )
    df['swing_low'] = (
        (df_ohlc['low'] < df_ohlc['low'].shift(1)) &
        (df_ohlc['low'] < df_ohlc['low'].shift(-1)) &
        (pd.concat([df_ohlc['low'].shift(1), df_ohlc['low'].shift(-1)], axis=1).min(axis=1) - df_ohlc['low'] > atr * min_swing_atr)
    )

    # Intermediate-term swings
    for i in range(swing_lookback, len(df) - swing_lookback):
        if df['swing_high'].iloc[i]:
            left = df_ohlc.iloc[i - swing_lookback:i]
            right = df_ohlc.iloc[i + 1:i + swing_lookback + 1]
            if (df['swing_high'].iloc[i - swing_lookback:i].any() and
                df_ohlc['high'].iloc[i] > left['high'].max() and
                df['swing_high'].iloc[i + 1:i + swing_lookback + 1].any() and
                df_ohlc['high'].iloc[i] > right['high'].max()):
                df.at[df.index[i], 'ith'] = True
        if df['swing_low'].iloc[i]:
            left = df_ohlc.iloc[i - swing_lookback:i]
            right = df_ohlc.iloc[i + 1:i + swing_lookback + 1]
            if (df['swing_low'].iloc[i - swing_lookback:i].any() and
                df_ohlc['low'].iloc[i] < left['low'].min() and
                df['swing_low'].iloc[i + 1:i + swing_lookback + 1].any() and
                df_ohlc['low'].iloc[i] < right['low'].min()):
                df.at[df.index[i], 'itl'] = True

    # Volume and RSI confirmation
    if use_volume:
        avg_volume = df_ohlc['volume'].rolling(window=atr_period).mean()
        df['high_volume'] = df_ohlc['volume'] > avg_volume * volume_threshold
    else:
        df['high_volume'] = True
    df['rsi_confirm'] = (
        (~use_rsi) |
        ((df['signal'] == 'buy') & (rsi > rsi_oversold)) |
        ((df['signal'] == 'sell') & (rsi < rsi_overbought))
    )

    # Liquidity Sweeps
    df['liquidity_sweep_high'] = (df_ohlc['high'] == df_ohlc['high'].shift(1)) & (df_ohlc['high'] == df_ohlc['high'].shift(2))
    df['liquidity_sweep_low'] = (df_ohlc['low'] == df_ohlc['low'].shift(1)) & (df_ohlc['low'] == df_ohlc['low'].shift(2))

    # SMT Divergence (basic: price moving opposite to correlated asset)
    if correlated_asset is not None and 'close' in correlated_asset.columns:
        df['smt_divergence'] = (
            ((df_ohlc['close'] > df_ohlc['close'].shift(1)) & (correlated_asset['close'] < correlated_asset['close'].shift(1))) |
            ((df_ohlc['close'] < df_ohlc['close'].shift(1)) & (correlated_asset['close'] > correlated_asset['close'].shift(1)))
        )

    # Order Blocks Detection
    ob_chain_bullish_count = 0
    ob_chain_bearish_count = 0
    for i in range(2, len(df)):
        # Bullish/Bearish OB
        if (df_ohlc['close'].iloc[i-1] < df_ohlc['open'].iloc[i-1] and
            df_ohlc['close'].iloc[i] > df_ohlc['high'].iloc[i-1]):
            df.at[df.index[i-1], 'bullish_ob'] = True
            if df['displacement'].iloc[i]:
                df.at[df.index[i-1], 'liquidity_ob_bullish'] = True
            if (df_ohlc['close'].iloc[i] > df_ohlc['open'].iloc[i] and
                df_ohlc['close'].iloc[i] > df_ohlc['high'].iloc[i-2]):
                df.at[df.index[i-1], 'engulfing_ob_bullish'] = True
            if not df['swing_high'].iloc[i-1]:
                df.at[df.index[i-1], 'internal_ob_bullish'] = True
            if (i < len(df) - 1 and df_ohlc['low'].iloc[i+1] <= df_ohlc['high'].iloc[i-1]):
                df.at[df.index[i-1], 'mitigation_block'] = True
            ob_chain_bullish_count += 1
            if ob_chain_bullish_count >= 2:
                df.at[df.index[i-1], 'ob_chain_bearish'] = True

        # Breaker Block: OB broken and retested
        if (i < len(df) - 1 and df['bullish_ob'].iloc[i-1] and
            df_ohlc['low'].iloc[i] <= df_ohlc['high'].iloc[i-1] and
            df_ohlc['close'].iloc[i+1] >= df_ohlc['high'].iloc[i-1]):
            df.at[df.index[i-1], 'breaker_block'] = True
        if (i < len(df) - 1 and df['bearish_ob'].iloc[i-1] and
            df_ohlc['high'].iloc[i] >= df_ohlc['low'].iloc[i-1] and
            df_ohlc['close'].iloc[i+1] <= df_ohlc['low'].iloc[i-1]):
            df.at[df.index[i-1], 'breaker_block'] = True

        # Inverse/Inducement OB
        if (df['bullish_ob'].iloc[i-1] and df['liquidity_sweep_high'].iloc[i]):
            df.at[df.index[i-1], 'inverse_ob_bullish'] = True
        if (df['bearish_ob'].iloc[i-1] and df['liquidity_sweep_low'].iloc[i]):
            df.at[df.index[i-1], 'inverse_ob_bearish'] = True

        # Refined OB: Body or midpoint
        if df['bullish_ob'].iloc[i-1]:
            body_range = df_ohlc['close'].iloc[i-1] - df_ohlc['open'].iloc[i-1]
            if abs(body_range) <= atr.iloc[i-1] * 0.5:
                df.at[df.index[i-1], 'refined_ob_bullish'] = True
        if df['bearish_ob'].iloc[i-1]:
            body_range = df_ohlc['open'].iloc[i-1] - df_ohlc['close'].iloc[i-1]
            if abs(body_range) <= atr.iloc[i-1] * 0.5:
                df.at[df.index[i-1], 'refined_ob_bearish'] = True

        # Continuation OB: Within displacement leg
        if (df['bullish_ob'].iloc[i-1] and df['bos'].iloc[i]):
            df.at[df.index[i-1], 'continuation_ob_bullish'] = True
        if (df['bearish_ob'].iloc[i-1] and df['bos'].iloc[i]):
            df.at[df.index[i-1], 'continuation_ob_bearish'] = True

        # Institutional Candle: High volume, displacement
        if (df['displacement'].iloc[i] and df['high_volume'].iloc[i]):
            df.at[df.index[i], 'institutional_candle'] = True

    # HTF Order Blocks
    if df_htf is not None and 'high' in df_htf.columns and 'low' in df_htf.columns:
        for i in range(2, len(df)):
            htf_idx = df_htf.index[df_htf.index <= df.index[i]][-1] if len(df_htf.index[df_htf.index <= df.index[i]]) > 0 else None
            if htf_idx is not None:
                htf_ohlc = df_htf.loc[htf_idx]
                if (htf_ohlc['close'] < htf_ohlc['open'] and
                    df_ohlc['close'].iloc[i] > htf_ohlc['high']):
                    df.at[df.index[i], 'htf_ob_bullish'] = True
                if (htf_ohlc['close'] > htf_ohlc['open'] and
                    df_ohlc['close'].iloc[i] < htf_ohlc['low']):
                    df.at[df.index[i], 'htf_ob_bearish'] = True

    # BOS, CHoCH, MSS, True Break, Displacement
    for i in range(max(5, atr_period), len(df)):
        swing_highs = df[df['swing_high'] & (df.index < df.index[i])]['high']
        swing_lows = df[df['swing_low'] & (df.index < df.index[i])]['low']
        prev_swing_high = swing_highs.iloc[-2] if len(swing_highs) >= 2 else None
        prev_swing_low = swing_lows.iloc[-2] if len(swing_lows) >= 2 else None

        if prev_swing_high is None or prev_swing_low is None:
            logger.debug(f"Skipping index {i}: Insufficient swing points")
            continue

        # Propagate market structure
        if i > max(5, atr_period) and not df['choch'].iloc[i] and not df['mss'].iloc[i]:
            df.at[df.index[i], 'market_structure'] = df['market_structure'].iloc[i-1]

        # BOS and True Break
        if (df['market_structure'].iloc[i-1] == 'bullish' and
            df_ohlc['close'].iloc[i] > prev_swing_high and
            df['high_volume'].iloc[i] and
            (not use_rsi or rsi.iloc[i] > rsi_oversold)):
            df.at[df.index[i], 'bos'] = True
            if df_ohlc['close'].iloc[i] > df_ohlc['open'].iloc[i]:
                df.at[df.index[i], 'true_break'] = True
                if (df['bullish_ob'].iloc[i-1] or df['refined_ob_bullish'].iloc[i-1] or
                    df['liquidity_ob_bullish'].iloc[i-1] or df['engulfing_ob_bullish'].iloc[i-1]):
                    df.at[df.index[i], 'signal'] = 'buy'
        elif (df['market_structure'].iloc[i-1] == 'bearish' and
              df_ohlc['close'].iloc[i] < prev_swing_low and
              df['high_volume'].iloc[i] and
              (not use_rsi or rsi.iloc[i] < rsi_overbought)):
            df.at[df.index[i], 'bos'] = True
            if df_ohlc['close'].iloc[i] < df_ohlc['open'].iloc[i]:
                df.at[df.index[i], 'true_break'] = True
                if (df['bearish_ob'].iloc[i-1] or df['refined_ob_bearish'].iloc[i-1] or
                    df['liquidity_ob_bearish'].iloc[i-1] or df['engulfing_ob_bearish'].iloc[i-1]):
                    df.at[df.index[i], 'signal'] = 'sell'

        # CHoCH and MSS
        if (df['market_structure'].iloc[i-1] == 'bullish' and
            df_ohlc['close'].iloc[i] < prev_swing_low and
            df['high_volume'].iloc[i] and
            (not use_rsi or rsi.iloc[i] < rsi_overbought)):
            df.at[df.index[i], 'choch'] = True
            if df['displacement'].iloc[i] or df['institutional_candle'].iloc[i]:
                df.at[df.index[i], 'mss'] = True
                df.at[df.index[i], 'market_structure'] = 'bearish'
                if (df['bearish_ob'].iloc[i-1] or df['refined_ob_bearish'].iloc[i-1] or
                    df['liquidity_ob_bearish'].iloc[i-1] or df['engulfing_ob_bearish'].iloc[i-1]):
                    df.at[df.index[i], 'signal'] = 'sell'
        elif (df['market_structure'].iloc[i-1] == 'bearish' and
              df_ohlc['close'].iloc[i] > prev_swing_high and
              df['high_volume'].iloc[i] and
              (not use_rsi or rsi.iloc[i] > rsi_oversold)):
            df.at[df.index[i], 'choch'] = True
            if df['displacement'].iloc[i] or df['institutional_candle'].iloc[i]:
                df.at[df.index[i], 'mss'] = True
                df.at[df.index[i], 'market_structure'] = 'bullish'
                if (df['bullish_ob'].iloc[i-1] or df['refined_ob_bullish'].iloc[i-1] or
                    df['liquidity_ob_bullish'].iloc[i-1] or df['engulfing_ob_bullish'].iloc[i-1]):
                    df.at[df.index[i], 'signal'] = 'buy'

        # Displacement
        if pd.notna(atr.iloc[i]) and (df_ohlc['high'].iloc[i] - df_ohlc['low'].iloc[i]) > (atr_multiplier * atr.iloc[i]):
            df.at[df.index[i], 'displacement'] = True

        # Internal Structure Analysis
        if df_m1 is not None and df.index[i] in df_m1.index.floor('15T'):
            internal_df = detect_internal_structure(df_m1, df.index[i], swing_lookback, atr_period, min_swing_atr)
            if not internal_df.empty:
                df.at[df.index[i], 'internal_structure'] = internal_df['internal_structure'].iloc[-1]
                df.at[df.index[i], 'internal_ob_bullish'] |= internal_df['internal_ob_bullish'].iloc[-1]
                df.at[df.index[i], 'internal_ob_bearish'] |= internal_df['internal_ob_bearish'].iloc[-1]
                # Precision entry: Signal only if internal structure aligns
                if df['signal'].iloc[i] == 'buy' and df['internal_structure'].iloc[i] != 'bullish':
                    df.at[df.index[i], 'signal'] = 'hold'
                if df['signal'].iloc[i] == 'sell' and df['internal_structure'].iloc[i] != 'bearish':
                    df.at[df.index[i], 'signal'] = 'hold'

    # Market Structure Bias (recent bars)
    if len(df) >= structure_lookback:
        for i in range(max(5, len(df) - structure_lookback), len(df)):
            if i >= 5:
                if (df_ohlc['high'].iloc[i] > df_ohlc['high'].iloc[max(0, i-5)] and
                    df_ohlc['low'].iloc[i] > df_ohlc['low'].iloc[max(0, i-5)]):
                    df.at[df.index[i], 'market_structure'] = 'bullish'
                elif (df_ohlc['high'].iloc[i] < df_ohlc['high'].iloc[max(0, i-5)] and
                      df_ohlc['low'].iloc[i] < df_ohlc['low'].iloc[max(0, i-5)]):
                    df.at[df.index[i], 'market_structure'] = 'bearish'
                    
                    

def detect_internal_structure(df: pd.DataFrame, timeframe: str = '1min', parent_timeframe: str = '15min', 
                             plot: bool = False, email_alert: str = None, trade_symbol: str = None) -> pd.DataFrame:
    """
    Analyze M1 substructure within M15 legs as per Compendium Section 1, with liquidity zones, backtesting, 
    trading integration, ML signal refinement, and alerts.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame with tick data (columns: timestamp, price, optional volume).
    - timeframe (str): Substructure timeframe (e.g., '1min' for M1).
    - parent_timeframe (str): Parent leg timeframe (e.g., '15min' for M15).
    - plot (bool): If True, generate a candlestick chart with structure annotations.
    - email_alert (str): Email address for sending precision entry alerts (optional).
    - trade_symbol (str): Symbol for trading platform integration (e.g., 'BTCUSD') (optional).
    
    Returns:
    - pd.DataFrame: M1 OHLC DataFrame with market structure annotations and signal probabilities.
    
    Raises:
    - ValueError: If input DataFrame or timeframes are invalid.
    """
    # Validate input
    if not isinstance(df, pd.DataFrame) or 'timestamp' not in df.columns or 'price' not in df.columns:
        raise ValueError("Input DataFrame must contain 'timestamp' and 'price' columns.")
    if df['timestamp'].isna().any():
        raise ValueError("Timestamp column contains missing values.")
    if not df['timestamp'].is_monotonic_increasing:
        df = df.sort_values('timestamp').copy()
    
    # Map timeframes
    timeframe_map: Dict[str, str] = {'1min': '1T', '5min': '5T', '15min': '15T', '1H': '1H', '1D': '1D'}
    if timeframe not in timeframe_map or parent_timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe. Use: {list(timeframe_map.keys())}")
    
    # Convert tick data to M1 and M15 OHLC
    df_m1 = convert_ticks_to_ohlc(df, timeframe_map[timeframe])
    df_m15 = convert_ticks_to_ohlc(df, timeframe_map[parent_timeframe])
    
    # Detect market structure on M15 (parent legs)
    df_m15 = detect_market_structure(df_m15)
    
    # Map M15 structure to M1
    df_m1 = map_parent_structure(df_m1, df_m15)
    
    # Detect M1 internal structure
    df_m1 = detect_market_structure(df_m1, is_substructure=True)
    
    # Detect liquidity zones
    df_m1 = detect_liquidity_zones(df_m1)
    
    # Train ML model for signal probability
    clf = train_signal_classifier(df_m1)
    df_m1['signal_probability'] = clf.predict_proba(df_m1[['atr', 'momentum', 'volume']].fillna(0))[:, 1]
    
    # Backtest strategy
    metrics = backtest_strategy(df_m1)
    print(f"Backtest Metrics: {metrics}")
    
    # Submit trade for precision entries (if trade_symbol provided)
    if trade_symbol and df_m1['is_precision_entry'].iloc[-1]:
        submit_order(trade_symbol, 
                     'buy' if df_m1['market_structure'].iloc[-1] == 'bullish' else 'sell',
                     qty=0.01,
                     entry_price=df_m1['close'].iloc[-1],
                     stop_loss=df_m1['close'].iloc[-1] - df_m1['atr'].iloc[-1] * 1.5,
                     take_profit=df_m1['close'].iloc[-1] + df_m1['atr'].iloc[-1] * 3.0)
    
    # Send alert for precision entries (if email_alert provided)
    if email_alert and df_m1['is_precision_entry'].iloc[-1]:
        message = (f"Precision Entry on {timeframe}: BOS={df_m1['bos'].iloc[-1]}, "
                   f"CHoCH={df_m1['choch'].iloc[-1]}, Signal Probability={df_m1['signal_probability'].iloc[-1]:.2f}")
        send_alert(message, email_alert)
    
    # Plot if requested
    if plot:
        plot_structure(df_m1, timeframe)
    
    return df_m1


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def vwap(df: pd.DataFrame) -> float:
    """Calculate Volume-Weighted Average Price (VWAP)."""
    if 'volume' in df.columns and df['volume'].sum() > 0:
        return (df['price'] * df['volume']).sum() / df['volume'].sum()
    return df['price'].mean()

def convert_ticks_to_ohlc(
    df: pd.DataFrame,
    timeframe: str = None,
    tz: Optional[str] = None,
    drop_na: bool = False,
    fill_method: Optional[str] = 'ffill',
    agg_funcs: Optional[Dict[str, Callable]] = None,
    extra_cols: Optional[Dict[str, Callable]] = None,
    use_dask: bool = False,
    session_hours: Optional[Tuple[str, str]] = None,
    include_vwap: bool = False,
    outlier_threshold: Optional[float] = None,
    include_metadata: bool = False,
    plot: bool = False,
    export_to: Optional[str] = None,
    volume_per_bar: Optional[int] = None
) -> pd.DataFrame:
    """
    Convert tick data to OHLC format with advanced options.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'timestamp' and 'price' columns, and optionally 'volume' or other columns.
    timeframe : str, optional
        Resampling timeframe (e.g., '1min', '1H', '1D'). Required unless volume_per_bar is specified.
    tz : str, optional
        Time zone for timestamps (e.g., 'UTC', 'US/Eastern').
    drop_na : bool, optional
        If True, drop rows with missing OHLC values.
    fill_method : str, optional
        Method to fill missing values ('ffill', 'bfill', None).
    agg_funcs : Dict[str, Callable], optional
        Custom aggregation functions for columns (e.g., {'price': 'ohlc', 'volume': 'sum'}).
    extra_cols : Dict[str, Callable], optional
        Additional columns and their aggregation functions (e.g., {'bid': 'mean'}).
    use_dask : bool, optional
        If True, use Dask for parallel processing of large datasets.
    session_hours : Tuple[str, str], optional
        Market session hours (e.g., ('09:30', '16:00') for NYSE).
    include_vwap : bool, optional
        If True, include VWAP column (requires 'volume' column).
    outlier_threshold : float, optional
        Z-score threshold for filtering price outliers (e.g., 3.0).
    include_metadata : bool, optional
        If True, attach metadata (e.g., source, timeframe) to the output DataFrame.
    plot : bool, optional
        If True, generate a candlestick chart using Plotly.
    export_to : str, optional
        File path to export the result (e.g., 'output.csv', 'output.parquet').
    volume_per_bar : int, optional
        Number of volume units per bar for event-based resampling.

    Returns:
    --------
    pd.DataFrame
        DataFrame with aggregated columns based on timeframe or volume bars.

    Raises:
    -------
    ValueError
        If required columns are missing, timeframe is invalid, or timestamps are invalid.
    """
    logger.info(f"Converting {len(df)} ticks to OHLC with timeframe {timeframe or 'volume-based'}")

    # Validate input
    if not {'timestamp', 'price'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'timestamp' and 'price' columns")
    
    if not timeframe and not volume_per_bar:
        raise ValueError("Either 'timeframe' or 'volume_per_bar' must be specified")

    # Create a copy to avoid modifying the original
    df = df.copy()

    # Clean data: handle outliers and duplicates
    if outlier_threshold:
        z_scores = (df['price'] - df['price'].mean()) / df['price'].std()
        outliers = abs(z_scores) >= outlier_threshold
        if outliers.any():
            logger.warning(f"Filtering {outliers.sum()} price outliers (z-score >= {outlier_threshold})")
            df = df[~outliers]
    
    if df['timestamp'].duplicated().any():
        logger.warning("Duplicate timestamps detected, dropping duplicates")
        df = df.drop_duplicates(subset='timestamp')

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isna().any():
        raise ValueError("Invalid timestamp values detected")
    
    # Apply time zone
    if tz:
        df['timestamp'] = df['timestamp'].dt.tz_localize(tz)

    df.set_index('timestamp', inplace=True)

    # Handle event-based resampling (volume bars)
    if volume_per_bar and 'volume' in df.columns:
        logger.info(f"Resampling by volume bars (volume_per_bar={volume_per_bar})")
        df['cum_volume'] = df['volume'].cumsum()
        df['bar_group'] = (df['cum_volume'] // volume_per_bar).astype(int)
        agg_dict = {
            'timestamp': 'first',
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        }
        if extra_cols:
            agg_dict.update(extra_cols)
        if include_vwap:
            agg_dict['vwap'] = vwap
        
        resampled = df.groupby('bar_group').agg(agg_dict)
        resampled.columns = ['timestamp', 'open', 'high', 'low', 'close'] + \
                            ([f"{col}_{func.__name__}" for col, func in (extra_cols or {}).items()] +
                             (['vwap'] if include_vwap else []))
        resampled = resampled.reset_index(drop=True)
    else:
        # Validate timeframe
        try:
            pd.Timedelta(timeframe)
        except ValueError:
            raise ValueError(f"Invalid timeframe: {timeframe}. Use formats like '1min', '1H', '1D'.")

        # Apply session hours
        if session_hours:
            logger.info(f"Restricting to session hours: {session_hours}")
            bhour = CustomBusinessHour(start=session_hours[0], end=session_hours[1])
            df = df[df.index.map(lambda x: bhour.is_on_offset(x))]

        # Default aggregations
        default_agg = {'price': 'ohlc'}
        if 'volume' in df.columns:
            default_agg['volume'] = 'sum'
        if extra_cols:
            default_agg.update(extra_cols)
        if include_vwap and 'volume' in df.columns:
            default_agg['vwap'] = vwap

        # Apply custom aggregations if provided
        agg_funcs = agg_funcs or default_agg

        # Resample using Dask or pandas
        if use_dask:
            logger.info("Using Dask for parallel processing")
            ddf = dd.from_pandas(df, npartitions=4)
            resampled = ddf.resample(timeframe).agg(agg_funcs)
            if 'price' in agg_funcs and agg_funcs['price'] == 'ohlc':
                resampled = resampled['price'].join(resampled.drop('price', axis=1)).compute()
            else:
                resampled = resampled.compute()
        else:
            resampled = df.resample(timeframe).agg(agg_funcs)
            if 'price' in agg_funcs and agg_funcs['price'] == 'ohlc':
                resampled = resampled['price'].join(resampled.drop('price', axis=1))

    # Handle missing values
    if 'open' in resampled.columns and fill_method:
        resampled[['open', 'high', 'low', 'close']] = resampled[['open', 'high', 'low', 'close']].fillna(method=fill_method)
    if drop_na:
        resampled = resampled.dropna()

    # Reset index
    resampled = resampled.reset_index()

    # Add metadata
    if include_metadata:
        resampled.attrs['source'] = 'tick_data'
        resampled.attrs['timeframe'] = timeframe or f"volume_per_bar_{volume_per_bar}"
        resampled.attrs['processed_at'] = pd.Timestamp.now()
        resampled.attrs['tz'] = tz
        resampled.attrs['session_hours'] = session_hours
        logger.info("Metadata attached to DataFrame")

    # Plot candlestick chart
    if plot:
        if 'open' in resampled.columns:
            logger.info("Generating candlestick chart")
            fig = go.Figure(data=[
                go.Candlestick(
                    x=resampled['timestamp'],
                    open=resampled['open'],
                    high=resampled['high'],
                    low=resampled['low'],
                    close=resampled['close']
                )
            ])
            fig.update_layout(
                title=f"OHLC Chart ({timeframe or 'volume-based'})",
                xaxis_title="Time",
                yaxis_title="Price"
            )
            fig.show()
        else:
            logger.warning("Cannot plot: OHLC columns not found")

    # Export to file
    if export_to:
        logger.info(f"Exporting to {export_to}")
        if export_to.endswith('.csv'):
            resampled.to_csv(export_to, index=False)
        elif export_to.endswith('.parquet'):
            resampled.to_parquet(export_to)
        else:
            logger.warning(f"Unsupported file format: {export_to}")

    logger.info("Conversion complete")
    return resampled

import numpy as np
from numba import jit

@jit(nopython=True)
def detect_swings(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Detect 3-bar fractal swing points in price data with closing price confirmation.

    Parameters:
    -----------
    highs : np.ndarray
        Array of high prices.
    lows : np.ndarray
        Array of low prices.
    closes : np.ndarray
        Array of close prices used to confirm swing points.

    Returns:
    --------
    np.ndarray
        Array of strings indicating swing points:
        - 'neutral': No swing point.
        - 'HH': Higher high (high > highs[i-2], close within 5% of high).
        - 'LH': Lower high (high <= highs[i-2], close within 5% of high).
        - 'HL': Higher low (low > lows[i-2], close within 5% of low).
        - 'LL': Lower low (low <= lows[i-2], close within 5% of low).
        - 'HH_HL', 'HH_LL', 'LH_HL', 'LH_LL': Combined swing high and low.

    Raises:
    -------
    ValueError
        If input arrays have different lengths or fewer than 5 elements.

    Notes:
    ------
    - The first two and last two bars are always 'neutral' due to the 5-bar window.
    - A swing high/low is only labeled if the closing price is within 5% of the high/low.
    - A bar can be both a swing high and low, resulting in a combined label.
    """
    if len(highs) != len(lows) or len(highs) != len(closes):
        raise ValueError("Input arrays 'highs', 'lows', and 'closes' must have the same length")
    if len(highs) < 5:
        raise ValueError("Input arrays must have at least 5 elements for 3-bar fractal detection")

    # Use numerical codes for Numba: 0=neutral, 1=HH, 2=LH, 3=HL, 4=LL, 5=HH_HL, 6=HH_LL, 7=LH_HL, 8=LH_LL
    swings = np.zeros(len(highs), dtype=np.int8)
    close_threshold = 0.95  # Close must be within 5% of high/low

    for i in range(2, len(highs) - 2):
        # Check for swing high with close confirmation
        if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                highs[i] > highs[i-2] and highs[i] > highs[i+2] and
                closes[i] >= highs[i] * close_threshold):
            swings[i] = 1 if highs[i] > highs[i-2] else 2  # 1=HH, 2=LH

        # Check for swing low with close confirmation
        if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
                lows[i] < lows[i-2] and lows[i] < lows[i+2] and
                closes[i] <= lows[i] / close_threshold):  # Equivalent to closes[i] <= lows[i] * 1.05
            if swings[i] != 0:  # Already a swing high
                swings[i] += 4  # 5=HH_HL, 6=HH_LL, 7=LH_HL, 8=LH_LL
            else:
                swings[i] = 3 if lows[i] > lows[i-2] else 4  # 3=HL, 4=LL

    # Map numerical codes to strings
    result = np.array(['neutral'] * len(highs), dtype='U7')
    for i in range(len(highs)):
        if swings[i] == 1:
            result[i] = 'HH'
        elif swings[i] == 2:
            result[i] = 'LH'
        elif swings[i] == 3:
            result[i] = 'HL'
        elif swings[i] == 4:
            result[i] = 'LL'
        elif swings[i] == 5:
            result[i] = 'HH_HL'
        elif swings[i] == 6:
            result[i] = 'HH_LL'
        elif swings[i] == 7:
            result[i] = 'LH_HL'
        elif swings[i] == 8:
            result[i] = 'LH_LL'

    return result

import pandas as pd
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_swings(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                 window: int = 5, min_strength: float = 0.0) -> np.ndarray:
    """Detect swing points using a fractal-based algorithm (Williams Fractal).
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        window: Lookback/lookforward periods for fractal detection (default: 5)
        min_strength: Minimum price difference for swing confirmation (default: 0.0)
    
    Returns:
        Array of swing labels ('HH', 'LH', 'HL', 'LL', 'neutral')
    """
    logger.info("Starting swing detection with window=%d, min_strength=%.4f", window, min_strength)
    swings = np.full(len(highs), 'neutral', dtype=object)
    half_window = window // 2
    
    for i in range(half_window, len(highs) - half_window):
        # Check for swing high (fractal pattern)
        if (highs[i] > np.max(highs[i-half_window:i]) and 
            highs[i] > np.max(highs[i+1:i+half_window+1])):
            prev_high = np.max(highs[max(0, i-window):i]) if i > window else highs[0]
            swings[i] = 'HH' if highs[i] > prev_high + min_strength else 'LH'
        
        # Check for swing low (fractal pattern)
        if (lows[i] < np.min(lows[i-half_window:i]) and 
            lows[i] < np.min(lows[i+1:i+half_window+1])):
            prev_low = np.min(lows[max(0, i-window):i]) if i > window else lows[0]
            swings[i] = 'HL' if lows[i] > prev_low - min_strength else 'LL'
    
    logger.info("Swing detection completed. Found %d swing points", np.sum(swings != 'neutral'))
    return swings

def detect_market_structure(df_ohlc: pd.DataFrame, 
                          is_substructure: bool = False,
                          atr_period: int = 14,
                          momentum_period: int = 20,
                          ith_window: int = 10,
                          ith_shift: int = 5,
                          displacement_threshold: float = 2.0,
                          swing_window: int = 5,
                          min_swing_strength: float = 0.0,
                          volume_threshold: float = 1.5) -> pd.DataFrame:
    """Detect complex market structure with fractal-based swings, BOS, CHoCH, MSS, displacement, and order flow bias.
    
    Args:
        df_ohlc: DataFrame with columns ['open', 'high', 'low', 'close', 'volume' (optional)]
        is_substructure: Enable precision entry detection for substructure (e.g., M1)
        atr_period: Period for ATR calculation
        momentum_period: Period for momentum calculation
        ith_window: Window for intermediate-term high/low detection
        ith_shift: Shift for intermediate-term high/low detection
        displacement_threshold: Base multiplier for displacement detection
        swing_window: Window for fractal-based swing detection
        min_swing_strength: Minimum price difference for swing confirmation
        volume_threshold: Multiplier for volume confirmation (if volume available)
    
    Returns:
        DataFrame with market structure indicators
    
    Raises:
        ValueError: If required columns are missing or DataFrame is empty
    """
    logger.info("Starting market structure detection for DataFrame with %d rows", len(df_ohlc))
    
    # Input validation
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in df_ohlc.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    if df_ohlc.empty:
        raise ValueError("Input DataFrame is empty")
    
    df = df_ohlc.copy()
    
    # Calculate ATR (standard formula)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(atr_period, min_periods=1).mean()
    
    # Adaptive displacement threshold based on ATR volatility
    atr_volatility = df['atr'].std()
    adaptive_threshold = displacement_threshold * (1 + atr_volatility / df['atr'].mean() if df['atr'].mean() != 0 else 1)
    logger.info("Adaptive displacement threshold: %.4f", adaptive_threshold)
    
    # Calculate momentum with slope confirmation
    df['momentum'] = df['close'].diff().rolling(momentum_period, min_periods=1).mean()
    df['momentum_slope'] = df['momentum'].diff().rolling(5, min_periods=1).mean()
    
    # Volume confirmation (if available)
    has_volume = 'volume' in df.columns
    if has_volume:
        df['volume_ma'] = df['volume'].rolling(momentum_period, min_periods=1).mean()
        df['volume_spike'] = df['volume'] > volume_threshold * df['volume_ma']
    
    # Detect swing points
    swings = detect_swings(df['high'].to_numpy(), df['low'].to_numpy(), 
                          df['close'].to_numpy(), swing_window, min_swing_strength)
    df['structure'] = swings
    df['swing_high'] = df['structure'].isin(['HH', 'LH'])
    df['swing_low'] = df['structure'].isin(['HL', 'LL'])
    
    # Intermediate-term highs/lows
    df['ith'] = df['swing_high'] & (df['high'] > df['high'].shift(ith_shift).rolling(ith_window, min_periods=1).max())
    df['itl'] = df['swing_low'] & (df['low'] < df['low'].shift(ith_shift).rolling(ith_window, min_periods=1).min())
    
    # Initialize output columns
    df['bos'] = 'neutral'
    df['true_break'] = False
    df['choch'] = 'neutral'
    df['mss'] = False
    df['displacement'] = False
    df['market_structure'] = 'neutral'
    
    # Vectorized BOS detection
    swing_highs = df[df['swing_high']][['high']].copy()
    swing_lows = df[df['swing_low']][['low']].copy()
    
    # Forward-fill swing highs/lows for vectorized comparison
    df['last_swing_high'] = swing_highs['high'].reindex(df.index, method='ffill').fillna(float('inf'))
    df['last_swing_low'] = swing_lows['low'].reindex(df.index, method='ffill').fillna(-float('inf'))
    
    # BOS: Bullish if close > last swing high (not HH), Bearish if close < last swing low (not LL)
    df.loc[(df['close'] > df['last_swing_high']) & (~df['structure'].isin(['HH'])), 'bos'] = 'bullish'
    df.loc[(df['close'] < df['last_swing_low']) & (~df['structure'].isin(['LL'])), 'bos'] = 'bearish'
    
    # True Break: Confirm BOS with open price
    df.loc[(df['bos'] == 'bullish') & (df['open'] > df['last_swing_high']), 'true_break'] = True
    df.loc[(df['bos'] == 'bearish') & (df['open'] < df['last_swing_low']), 'true_break'] = True
    
    # CHoCH and MSS (loop for sequential dependency)
    for i in range(1, len(df)):
        idx = df.index[i]
        if df['bos'].iloc[i-1] == 'bullish' and df['close'].iloc[i] < df['last_swing_low'].iloc[i]:
            df.loc[idx, 'choch'] = 'bearish'
            if (df['high'].iloc[i] - df['low'].iloc[i]) > adaptive_threshold * df['atr'].iloc[i]:
                df.loc[idx, 'mss'] = True
        elif df['bos'].iloc[i-1] == 'bearish' and df['close'].iloc[i] > df['last_swing_high'].iloc[i]:
            df.loc[idx, 'choch'] = 'bullish'
            if (df['high'].iloc[i] - df['low'].iloc[i]) > adaptive_threshold * df['atr'].iloc[i]:
                df.loc[idx, 'mss'] = True
    
    # Displacement with adaptive threshold
    df['displacement'] = (df['high'] - df['low']) > adaptive_threshold * df['atr']
    
    # Order Flow Bias with momentum and slope confirmation
    df.loc[(df['momentum'] > 0) & (df['momentum_slope'] > 0), 'market_structure'] = 'bullish'
    df.loc[(df['momentum'] < 0) & (df['momentum_slope'] < 0), 'market_structure'] = 'bearish'
    
    # Precision entries with volume confirmation (if available)
    if is_substructure:
        df['is_precision_entry'] = ((df['bos'] != 'neutral') | 
                                  (df['choch'] != 'neutral') | 
                                  df['displacement'])
        if has_volume:
            df['is_precision_entry'] &= df['volume_spike']
    
    # Clean up temporary columns
    df.drop(columns=['last_swing_high', 'last_swing_low'], inplace=True)
    if has_volume:
        df.drop(columns=['volume_ma', 'volume_spike'], inplace=True)
    df.drop(columns=['momentum_slope'], inplace=True)
    
    logger.info("Market structure detection completed. Columns added: %s", 
                [col for col in df.columns if col not in df_ohlc.columns])
    return df


def detect_liquidity_zones(
    df: pd.DataFrame,
    window: Union[int, str] = 20,
    gap_threshold: Optional[float] = None,
    displacement_threshold: Optional[float] = None,
    atr_period: int = 14,
    round_level_proximity: Optional[float] = None
) -> pd.DataFrame:
    """
    Detect liquidity zones above swing highs or below swing lows with robust handling.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'high', 'low', and optionally 'open', 'close', 'displacement' columns.
    - window (int or str): Rolling window size for swing high/low detection, or 'dynamic' for ATR-based window.
    - gap_threshold (float, optional): Threshold for detecting price gaps (e.g., as multiple of ATR).
    - displacement_threshold (float, optional): Threshold for calculating displacement (e.g., as multiple of ATR).
    - atr_period (int): Period for ATR calculation if used for displacement or dynamic window.
    - round_level_proximity (float, optional): Proximity to round price levels for filtering zones.

    Returns:
    - pd.DataFrame: DataFrame with 'liquidity_zone', 'liq_high', 'liq_low', and other derived columns.
    """
    # Input validation
    required_columns = ['high', 'low']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain {required_columns} columns.")
    
    df = df.copy()
    df['liquidity_zone'] = 'none'
    
    # Calculate ATR for dynamic window or displacement if needed
    if 'close' in df.columns and (displacement_threshold is not None or window == 'dynamic'):
        df['atr'] = df['close'].diff().abs().rolling(atr_period).mean()
    
    # Dynamic window based on ATR if specified
    if window == 'dynamic':
        if 'atr' not in df.columns:
            raise ValueError("ATR calculation requires 'close' column for dynamic window.")
        avg_atr = df['atr'].mean()
        window = max(10, min(50, int(avg_atr / df['close'].mean() * 100)))  # Scale window between 10-50
    elif not isinstance(window, int):
        raise ValueError("Window must be an integer or 'dynamic'.")
    
    # Calculate rolling swing highs and lows
    df['liq_high'] = df['high'].rolling(window).max().shift(1)
    df['liq_low'] = df['low'].rolling(window).min().shift(1)
    
    # Fill NaNs to avoid early data loss
    df['liq_high'] = df['liq_high'].fillna(df['high'])
    df['liq_low'] = df['liq_low'].fillna(df['low'])
    
    # Handle gaps if threshold provided
    if gap_threshold is not None and 'open' in df.columns and 'close' in df.columns:
        atr_factor = df['atr'] if 'atr' in df.columns else df['close'].diff().abs().mean()
        df['gap'] = abs(df['open'] - df['close'].shift(1)) > gap_threshold * atr_factor
        df.loc[df['gap'], ['liq_high', 'liq_low']] = pd.NA
        df['liq_high'] = df['liq_high'].fillna(df['high'])
        df['liq_low'] = df['liq_low'].fillna(df['low'])
    
    # Calculate displacement if not provided
    if 'displacement' not in df.columns and displacement_threshold is not None:
        if 'close' not in df.columns or 'open' not in df.columns:
            raise ValueError("Displacement calculation requires 'open' and 'close' columns.")
        df['displacement'] = abs(df['close'] - df['open']) > displacement_threshold * df['atr']
    elif 'displacement' not in df.columns:
        raise ValueError("DataFrame must contain 'displacement' column or provide displacement_threshold.")
    
    # Filter by proximity to round levels if specified
    round_filter = pd.Series(True, index=df.index)
    if round_level_proximity is not None and 'close' in df.columns:
        df['round_level'] = df['close'].round(-2)  # Nearest 100
        round_filter = abs(df['close'] - df['round_level']) < round_level_proximity
    
    # Identify liquidity zones
    above_condition = (df['high'] > df['liq_high']) & (df['displacement']) & round_filter
    below_condition = (df['low'] < df['liq_low']) & (df['displacement']) & round_filter
    
    # Handle overlapping zones
    df.loc[above_condition & below_condition, 'liquidity_zone'] = 'both'
    df.loc[above_condition & ~below_condition, 'liquidity_zone'] = 'above'
    df.loc[~above_condition & below_condition, 'liquidity_zone'] = 'below'
    
    # Track zone persistence
    df['zone_active'] = df['liquidity_zone'].replace('none', pd.NA).ffill()
    
    # Warn if no zones detected
    if not df['liquidity_zone'].isin(['above', 'below', 'both']).any():
        warnings.warn("No liquidity zones detected. Check 'displacement' or data conditions.")
    
    return df

import pandas as pd
from typing import List, Optional

def map_parent_structure(
    df_m1: pd.DataFrame,
    df_m15: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    structure_col: str = 'market_structure',
    suffix: str = '_m15',
    columns_to_map: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Map M15 structure to M1 for substructure analysis with enhanced validation and flexibility.

    Args:
        df_m1 (pd.DataFrame): M1 DataFrame with timestamp column.
        df_m15 (pd.DataFrame): M15 DataFrame with timestamp and structure columns.
        timestamp_col (str): Name of the timestamp column in both DataFrames. Defaults to 'timestamp'.
        structure_col (str): Name of the structure column in df_m15. Defaults to 'market_structure'.
        suffix (str): Suffix for mapped columns in the output. Defaults to '_m15'.
        columns_to_map (Optional[List[str]]): List of columns to map from df_m15. If None, uses structure_col.

    Returns:
        pd.DataFrame: M1 DataFrame with mapped M15 structure columns.

    Raises:
        ValueError: If required columns are missing, DataFrames are empty, or timestamps are invalid.
    """
    # Input validation
    if df_m1.empty or df_m15.empty:
        raise ValueError("Input DataFrames cannot be empty.")
    
    if timestamp_col not in df_m1.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in df_m1.")
    
    if timestamp_col not in df_m15.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in df_m15.")
    
    # Default to structure_col if columns_to_map is not provided
    columns_to_map = [structure_col] if columns_to_map is None else columns_to_map
    missing_cols = [col for col in columns_to_map if col not in df_m15.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in df_m15.")

    # Create copies to avoid modifying originals
    df_m1 = df_m1.copy()
    df_m15 = df_m15[[timestamp_col] + columns_to_map].copy()

    # Convert timestamps to datetime
    try:
        df_m1[timestamp_col] = pd.to_datetime(df_m1[timestamp_col])
        df_m15[timestamp_col] = pd.to_datetime(df_m15[timestamp_col])
    except Exception as e:
        '''
时代
System: It looks like your response was cut off at the end. Here's the completion of the improved function, continuing from where it left off, along with a summary of the improvements and how they address the identified issues:
'''
### Completion of the Improved Function

```python
        raise ValueError(f"Failed to convert '{timestamp_col}' to datetime: {str(e)}")

    # Ensure timestamps are sorted
    df_m1 = df_m1.sort_values(timestamp_col).reset_index(drop=True)
    df_m15 = df_m15.sort_values(timestamp_col).reset_index(drop=True)

    # Check for timezone consistency
    if df_m1[timestamp_col].dt.tz is not None and df_m15[timestamp_col].dt.tz is not None:
        if df_m1[timestamp_col].dt.tz != df_m15[timestamp_col].dt.tz:
            df_m15[timestamp_col] = df_m15[timestamp_col].dt.tz_convert(df_m1[timestamp_col].dt.tz)

    # Perform as-of merge
    try:
        result = pd.merge_asof(
            df_m1,
            df_m15,
            on=timestamp_col,
            direction='backward',
            suffixes=(None, suffix)
        )
    except Exception as e:
        raise ValueError(f"Merge failed: {str(e)}")

    # Warn if there are unmatched timestamps (NaN values in mapped columns)
    if result[columns_to_map[0] + suffix].isna().all():
        print(f"Warning: No matching timestamps found. All values in mapped columns are NaN.")

    return result

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union, Callable, Dict
import joblib
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_signal_classifier(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_columns: Optional[List[str]] = ['bos', 'choch'],
    target_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    target_mapping: Optional[Dict[str, Dict]] = None,
    n_estimators: int = 100,
    random_state: int = 42,
    cv_folds: int = 5,
    scale_numeric: bool = True,
    encode_categorical: bool = True,
    evaluate: bool = True,
    tune_params: bool = False,
    save_model_path: Optional[str] = None,
    balance_classes: Optional[str] = None,
    custom_preprocessor: Optional[Pipeline] = None,
    time_series: bool = False,
    bos_choch_mode: str = 'binary',
    verbose: bool = False
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train a Random Forest classifier to predict trading signals, handling diverse bos/choch formats.

    Args:
        df (pd.DataFrame): Input DataFrame containing feature and target columns.
        feature_columns (List[str], optional): Columns to use as features. If None, uses all numeric columns.
        target_columns (List[str], optional): Columns to derive the target (e.g., ['bos', 'choch']). Defaults to ['bos', 'choch'].
        target_fn (Callable, optional): Custom function to create target from df. Takes precedence over target_columns.
        target_mapping (Dict[str, Dict], optional): Mapping for bos/choch values to target classes (e.g., {'bos': {1: 'buy', -1: 'sell', 0: 'neutral'}}).
        n_estimators (int): Number of trees in the Random Forest. Defaults to 100.
        random_state (int): Random seed for reproducibility. Defaults to 42.
        cv_folds (int): Number of cross-validation folds. Defaults to 5, ignored if evaluate=False.
        scale_numeric (bool): Scale numeric features using StandardScaler. Defaults to True.
        encode_categorical (bool): Encode categorical features using OneHotEncoder. Defaults to True.
        evaluate (bool): Compute and return evaluation metrics. Defaults to True.
        tune_params (bool): Perform hyperparameter tuning with GridSearchCV. Defaults to False.
        save_model_path (str, optional): Path to save the trained model and preprocessor. Defaults to None.
        balance_classes (str, optional): Method to handle imbalanced classes ('balanced', 'smote', None). Defaults to None.
        custom_preprocessor (Pipeline, optional): Custom sklearn Pipeline for preprocessing. Defaults to None.
        time_series (bool): Use TimeSeriesSplit for cross-validation. Defaults to False.
        bos_choch_mode (str): Target creation mode ('binary', 'multi-class', 'numerical'). Defaults to 'binary'.
        verbose (bool): Enable detailed logging. Defaults to False.

    Returns:
        Tuple[RandomForestClassifier, dict]: Trained classifier and dictionary with evaluation metrics and metadata.

    Raises:
        ValueError: For invalid inputs, missing data, or constant targets.
        RuntimeError: For unexpected errors during training or evaluation.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame.")

    # Handle feature columns
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        logging.info(f"No feature_columns provided. Using numeric columns: {feature_columns}")
    if not feature_columns:
        raise ValueError("No valid feature columns specified or found.")

    # Validate feature columns
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in DataFrame: {missing_features}")

    # Prepare features
    X = df[feature_columns].copy()

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Handle missing values in features
    if X.isna().any().any():
        logging.warning("Missing values in features. Imputing with mean for numeric, mode for categorical.")
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

    if X.empty:
        raise ValueError("Feature data is empty after preprocessing.")

    # Prepare target
    if target_fn is not None:
        y = target_fn(df)
    elif target_columns is not None:
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_targets:
            raise ValueError(f"Target columns not found in DataFrame: {missing_targets}")

        # Validate bos and choch values and types
        for col in target_columns:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                col_type = df[col].dtype
                logging.debug(f"Column {col}: Type={col_type}, Unique values={unique_vals}")

        # Handle missing values in target columns
        df[target_columns] = df[target_columns].fillna(0 if bos_choch_mode == 'numerical' else 'neutral')

        # Apply target mapping if provided
        if target_mapping is not None:
            for col in target_columns:
                if col in target_mapping:
                    logging.info(f"Applying target mapping to {col}")
                    df[col] = df[col].map(target_mapping[col]).fillna('neutral' if bos_choch_mode != 'numerical' else 0)

        # Create target based on bos_choch_mode
        if bos_choch_mode == 'binary':
            # Binary: 1 for any non-neutral/non-zero signal, 0 otherwise
            if df[target_columns].select_dtypes(include=[np.number]).columns.any():
                y = (df[target_columns] != 0).any(axis=1).astype(int)
            else:
                y = (df[target_columns] != 'neutral').any(axis=1).astype(int)
        elif bos_choch_mode == 'multi-class':
            # Multi-class: Prioritize buy/sell, fallback to neutral
            if df[target_columns].select_dtypes(include=[np.number]).columns.any():
                # Numerical: Assume >0 = buy, <0 = sell, 0 = neutral
                y = df[target_columns].apply(
                    lambda row: 'buy' if (row > 0).any() else 'sell' if (row < 0).any() else 'neutral', axis=1)
            else:
                # Categorical: Prioritize buy/sell
                y = df[target_columns].apply(
                    lambda row: 'buy' if 'buy' in row.values else 'sell' if 'sell' in row.values else 'neutral', axis=1)
        elif bos_choch_mode == 'numerical':
            # Numerical: Use raw numerical values or mapped values
            if len(target_columns) == 1:
                y = df[target_columns[0]]
            else:
                raise ValueError("Numerical mode requires a single target column or custom target_fn.")
        else:
            raise ValueError("bos_choch_mode must be 'binary', 'multi-class', or 'numerical'.")
    else:
        raise ValueError("Either target_columns or target_fn must be provided.")

    if not isinstance(y, (pd.Series, np.ndarray)) or len(y) != len(X):
        raise ValueError("Target must be a Series/array with same length as features.")
    if y.nunique() < 2:
        raise ValueError(f"Target variable is constant (all {y.unique()[0]}). At least two classes required.")

    # Log target distribution
    logging.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

    # Preprocessing pipeline
    if custom_preprocessor is not None:
        preprocessor = custom_preprocessor
    else:
        transformers = []
        if numeric_cols and scale_numeric:
            transformers.append(('num', StandardScaler(), numeric_cols))
        if categorical_cols and encode_categorical:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols))
        if not transformers:
            raise ValueError("No valid preprocessing steps defined.")
        preprocessor = ColumnTransformer(transformers, remainder='passthrough')

    # Apply preprocessing
    try:
        X_transformed = preprocessor.fit_transform(X)
        feature_names = []
        if numeric_cols and scale_numeric:
            feature_names.extend(numeric_cols)
        if categorical_cols and encode_categorical:
            encoder = preprocessor.named_transformers_['cat']
            feature_names.extend(encoder.get_feature_names_out(categorical_cols))
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {str(e)}")

    # Handle class imbalance
    if balance_classes == 'balanced':
        clf_params = {'class_weight': 'balanced'}
    elif balance_classes == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_transformed, y = smote.fit_resample(X_transformed, y)
            logging.info(f"Applied SMOTE. New target distribution: {pd.Series(y).value_counts().to_dict()}")
        except Exception as e:
            logging.warning(f"SMOTE failed: {str(e)}. Proceeding without balancing.")
            balance_classes = None
            clf_params = {}
    else:
        clf_params = {}

    # Initialize classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **clf_params)

    # Hyperparameter tuning
    if tune_params:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        cv = TimeSeriesSplit(n_splits=cv_folds) if time_series else cv_folds
        grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        try:
            grid_search.fit(X_transformed, y)
            clf = grid_search.best_estimator_
            logging.info(f"Best parameters from GridSearchCV: {grid_search.best_params_}")
        except Exception as e:
            logging.warning(f"GridSearchCV failed: {str(e)}. Using default parameters.")

    # Train model
    try:
        clf.fit(X_transformed, y)
    except Exception as e:
        raise RuntimeError(f"Model training failed: {str(e)}")

    # Save model and preprocessor
    if save_model_path:
        try:
            joblib.dump({'model': clf, 'preprocessor': preprocessor}, save_model_path)
            logging.info(f"Model and preprocessor saved to {save_model_path}")
        except Exception as e:
            logging.warning(f"Failed to save model: {str(e)}")

    # Evaluation
    eval_metrics = {}
    if evaluate:
        try:
            # Cross-validation with time-series split if specified
            cv = TimeSeriesSplit(n_splits=cv_folds) if time_series else cv_folds
            cv_scores = cross_val_score(clf, X_transformed, y, cv=cv, scoring='accuracy', n_jobs=-1)
            eval_metrics['cv_accuracy_mean'] = np.mean(cv_scores)
            eval_metrics['cv_accuracy_std'] = np.std(cv_scores)

            # Feature importance
            eval_metrics['feature_importance'] = dict(zip(X_transformed.columns, clf.feature_importances_))

            # Classification report
            y_pred = clf.predict(X_transformed)
            eval_metrics['classification_report'] = classification_report(y, y_pred, output_dict=True)

            # Confusion matrix
            eval_metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()

            # bos and choch diagnostics
            for col in target_columns:
                if col in df.columns:
                    eval_metrics[f'{col}_distribution'] = df[col].value_counts().to_dict()
                    eval_metrics[f'{col}_type'] = str(df[col].dtype)
        except Exception as e:
            logging.warning(f"Evaluation failed: {str(e)}. Skipping evaluation metrics.")

    return clf, eval_metrics

def backtest_strategy(df: pd.DataFrame, stop_loss_atr: float = 1.5, take_profit_atr: float = 3.0) -> Dict:
    """Backtest structure-based trading signals."""
    trades = []
    for i in range(1, len(df)):
        if df['is_precision_entry'].iloc[i] and df['signal_probability'].iloc[i] > 0.7:
            entry_price = df['close'].iloc[i]
            stop_loss = entry_price - df['atr'].iloc[i] * stop_loss_atr
            take_profit = entry_price + df['atr'].iloc[i] * take_profit_atr
            for j in range(i+1, len(df)):
                if df['low'].iloc[j] <= stop_loss:
                    trades.append(-df['atr'].iloc[i] * stop_loss_atr)
                    break
                if df['high'].iloc[j] >= take_profit:
                    trades.append(df['atr'].iloc[i] * take_profit_atr)
                    break
    
    win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0
    profit_factor = sum(t for t in trades if t > 0) / abs(sum(t for t in trades if t < 0)) if any(t < 0 for t in trades) else float('inf')
    return {'win_rate': win_rate, 'total_trades': len(trades), 'profit_factor': profit_factor}

def submit_order(symbol: str, side: str, qty: float, entry_price: float, stop_loss: float, take_profit: float):
    """Submit order to Alpaca trading platform."""
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        client = TradingClient('API_KEY', 'SECRET_KEY', paper=True)  # Replace with your API keys
        order = LimitOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                                 time_in_force=TimeInForce.GTC, limit_price=entry_price,
                                 stop_loss={'stop_price': stop_loss}, take_profit={'limit_price': take_profit})
        client.submit_order(order)
        print(f"Order submitted: {side} {qty} {symbol} at {entry_price}")
    except ImportError:
        warnings.warn("alpaca-py not installed. Use `pip install alpaca-py`.")
    except Exception as e:
        warnings.warn(f"Order submission failed: {str(e)}")

def send_alert(message: str, email: str):
    """Send email alert for precision entries."""
    try:
        msg = MIMEText(message)
        msg['Subject'] = 'Trade Alert'
        msg['From'] = 'your_email@gmail.com'  # Replace with your email
        msg['To'] = email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your_email@gmail.com', 'your_app_password')  # Replace with your email and app password
            server.send_message(msg)
        print(f"Alert sent to {email}: {message}")
    except Exception as e:
        warnings.warn(f"Failed to send alert: {str(e)}")

def plot_structure(df_ohlc: pd.DataFrame, timeframe: str) -> None:
    """Plot OHLC data with market structure annotations."""
    fig = go.Figure(data=[
        go.Candlestick(x=df_ohlc['timestamp'],
                      open=df_ohlc['open'], high=df_ohlc['high'],
                      low=df_ohlc['low'], close=df_ohlc['close'],
                      name='OHLC')
    ])
    
    # Plot swing highs/lows
    for structure in ['HH', 'LH', 'HL', 'LL']:
        mask = df_ohlc['structure'] == structure
        y_values = df_ohlc[mask]['high'] if structure in ['HH', 'LH'] else df_ohlc[mask]['low']
        fig.add_scatter(x=df_ohlc[mask]['timestamp'], y=y_values,
                       mode='markers', name=structure,
                       marker=dict(size=10, symbol='circle',
                                  color='green' if structure in ['HH', 'HL'] else 'red'))
    
    # Plot BOS and CHoCH
    for event, color in [('bos', 'purple'), ('choch', 'orange')]:
        for direction in ['bullish', 'bearish']:
            mask = df_ohlc[event] == direction
            y_values = df_ohlc[mask]['high'] if direction == 'bullish' else df_ohlc[mask]['low']
            fig.add_scatter(x=df_ohlc[mask]['timestamp'], y=y_values,
                           mode='markers', name=f"{event.upper()} ({direction})",
                           marker=dict(size=8, symbol='diamond', color=color))
    
    # Plot displacement
    mask = df_ohlc['displacement']
    fig.add_scatter(x=df_ohlc[mask]['timestamp'], y=df_ohlc[mask]['close'],
                   mode='markers', name='Displacement',
                   marker=dict(size=8, symbol='star', color='blue'))
    
    # Plot liquidity zones
    for zone in ['above', 'below']:
        mask = df_ohlc['liquidity_zone'] == zone
        y_values = df_ohlc[mask]['high'] if zone == 'above' else df_ohlc[mask]['low']
        fig.add_scatter(x=df_ohlc[mask]['timestamp'], y=y_values,
                       mode='markers', name=f"Liquidity ({zone})",
                       marker=dict(size=8, symbol='square', color='cyan'))
    
    fig.update_layout(title=f'Market Structure Analysis ({timeframe})', xaxis_title='Time', yaxis_title='Price')
    fig.show()

def fetch_tick_data(symbol: str = 'BTC/USDT', limit: int = 1000) -> pd.DataFrame:
    """Fetch tick data from an exchange using ccxt."""
    try:
        import ccxt
        exchange = ccxt.binance()
        ticks = exchange.fetch_trades(symbol, limit=limit)
        df = pd.DataFrame(ticks, columns=['timestamp', 'price', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except ImportError:
        warnings.warn("ccxt not installed. Use `pip install ccxt`.")
        return pd.DataFrame()


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from collections import deque
from numba import jit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pip_factor(symbol):
    """
    Returns the pip factor for a given trading symbol.
    """
    pip_factors = {
        'EURUSD': 0.0001,
        'USDJPY': 0.01,
        'GBPUSD': 0.0001,
        'USDCHF': 0.0001,
        'SPX': 1.0,
    }
    return pip_factors.get(symbol, 0.0001)

def get_correlation_type(symbol1, symbol2):
    """
    Returns the correlation type between two symbols ('positive', 'negative', 'none').
    """
    positive_pairs = {('EURUSD', 'GBPUSD'), ('GBPUSD', 'EURUSD')}
    negative_pairs = {('EURUSD', 'USDCHF'), ('USDCHF', 'EURUSD'), ('GBPUSD', 'USDCHF'), ('USDCHF', 'GBPUSD')}
    
    if (symbol1, symbol2) in positive_pairs:
        return 'positive'
    elif (symbol1, symbol2) in negative_pairs:
        return 'negative'
    return 'none'

@jit(nopython=True)
def compute_po3(high, low, open, close, po3_lookback, min_swing_pips, pip_factor):
    """
    Numba-optimized function to compute PO3 pattern for a single symbol.
    """
    n = len(high)
    accumulation = np.zeros(n, dtype=np.bool_)
    manipulation = np.zeros(n, dtype=np.bool_)
    distribution = np.array(['none'] * n, dtype='U7')
    
    for i in range(po3_lookback, n):
        high_window = high[i - po3_lookback:i]
        low_window = low[i - po3_lookback:i]
        range_size = np.max(high_window) - np.min(low_window)
        
        if range_size < (10 * min_swing_pips * pip_factor):
            accumulation[i] = True
        
        if low[i] < np.min(low_window) or high[i] > np.max(high_window):
            manipulation[i] = True
            if low[i] < np.min(low_window) and close[i] < open[i]:
                distribution[i] = 'bearish'
            elif high[i] > np.max(high_window) and close[i] > open[i]:
                distribution[i] = 'bullish'
    
    return accumulation, manipulation, distribution

def detect_po3_multi_symbol(symbol_dfs, po3_lookback=20, min_swing_pips=10, pip_factors=None):
    """
    Detects PO3 patterns across multiple correlated symbols and generates combined signals.

    Parameters:
    -----------
    symbol_dfs : dict
        Dictionary of {symbol: DataFrame} with OHLC data for each symbol.
    po3_lookback : int, optional (default=20)
        Number of periods to look back for calculating range.
    min_swing_pips : int, optional (default=10)
        Minimum price movement in pips to define accumulation range.
    pip_factors : dict, optional (default=None)
        Dictionary of {symbol: pip_factor}. If None, determined automatically.

    Returns:
    --------
    dict
        Dictionary of {symbol: DataFrame} with PO3 columns and a 'combined_signal' column.
    """
    logger.info(f"Processing PO3 detection for {len(symbol_dfs)} symbols")
    
    # Input validation
    required_columns = ['open', 'high', 'low', 'close']
    for symbol, df in symbol_dfs.items():
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame for {symbol} must contain columns: {required_columns}")
        if len(df) < po3_lookback:
            raise ValueError(f"DataFrame for {symbol} must have at least {po3_lookback} rows")
    if not (isinstance(po3_lookback, int) and po3_lookback > 0):
        raise ValueError("po3_lookback must be a positive integer")
    if not (isinstance(min_swing_pips, int) and min_swing_pips > 0):
        raise ValueError("min_swing_pips must be a positive integer")
    
    if pip_factors is None:
        pip_factors = {symbol: get_pip_factor(symbol) for symbol in symbol_dfs}
    
    # Process PO3 for each symbol
    results = {}
    for symbol, df in symbol_dfs.items():
        df = df.copy().dropna(subset=required_columns)
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        open = df['open'].to_numpy()
        close = df['close'].to_numpy()
        
        accumulation, manipulation, distribution = compute_po3(
            high, low, open, close, po3_lookback, min_swing_pips, pip_factors[symbol]
        )
        
        df['po3_accumulation'] = accumulation
        df['po3_manipulation'] = manipulation
        df['po3_distribution'] = distribution
        df['po3_phase'] = np.where(
            distribution != 'none', distribution,
            np.where(manipulation, 'manipulation',
                     np.where(accumulation, 'accumulation', None))
        )
        df['po3_signal'] = np.where(distribution == 'bullish', 'buy',
                                   np.where(distribution == 'bearish', 'sell', None))
        results[symbol] = df
    
    # Generate combined signals based on correlations
    for symbol, df in results.items():
        df['combined_signal'] = df['po3_signal']
        for other_symbol, other_df in results.items():
            if symbol == other_symbol:
                continue
            correlation_type = get_correlation_type(symbol, other_symbol)
            if correlation_type == 'none':
                continue
            if correlation_type == 'positive':
                df['combined_signal'] = np.where(
                    (df['po3_signal'] == other_df['po3_signal']) & (df['po3_signal'].notnull()),
                    df['po3_signal'], None
                )
            elif correlation_type == 'negative':
                df['combined_signal'] = np.where(
                    (df['po3_signal'] == 'buy') & (other_df['po3_signal'] == 'sell'), 'buy',
                    np.where((df['po3_signal'] == 'sell') & (other_df['po3_signal'] == 'buy'), 'sell', None)
                )
    
    logger.info("PO3 detection with correlation completed")
    return results

def plot_po3_multi_symbol(symbol_dfs, title='PO3 Patterns'):
    """
    Plots candlestick charts with PO3 annotations for multiple symbols.
    
    Parameters:
    -----------
    symbol_dfs : dict
        Dictionary of {symbol: DataFrame} with OHLC and PO3 columns.
    title : str, optional
        Title for the plot.
    """
    logger.info(f"Generating PO3 visualization for {len(symbol_dfs)} symbols")
    
    fig = go.Figure()
    
    for symbol, df in symbol_dfs.items():
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=f'{symbol} OHLC',
            visible='legendonly' if symbol != list(symbol_dfs.keys())[0] else True
        ))
        
        for idx, row in df.iterrows():
            if row['po3_phase'] == 'accumulation':
                fig.add_vrect(
                    x0=idx, x1=idx + pd.Timedelta(minutes=1),
                    fillcolor='blue', opacity=0.1, layer='below',
                    annotation_text=f'{symbol} Accum', annotation_position='top left'
                )
            elif row['po3_phase'] == 'manipulation':
                fig.add_vrect(
                    x0=idx, x1=idx + pd.Timedelta(minutes=1),
                    fillcolor='yellow', opacity=0.2, layer='below',
                    annotation_text=f'{symbol} Manip', annotation_position='top left'
                )
            elif row['po3_phase'] in ['bullish', 'bearish']:
                fig.add_vrect(
                    x0=idx, x1=idx + pd.Timedelta(minutes=1),
                    fillcolor='green' if row['po3_phase'] == 'bullish' else 'red',
                    opacity=0.3, layer='below',
                    annotation_text=f'{symbol} {row["po3_phase"].capitalize()}',
                    annotation_position='top left'
                )
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    fig.show()

class PO3DetectorMultiSymbol:
    """
    Class for real-time PO3 detection across multiple correlated symbols.
    """
    def __init__(self, symbols, po3_lookback=20, min_swing_pips=10, pip_factors=None):
        self.symbols = symbols
        self.detectors = {
            symbol: PO3Detector(po3_lookback, min_swing_pips, pip_factors.get(symbol, get_pip_factor(symbol))
                                if pip_factors else get_pip_factor(symbol))
            for symbol in symbols
        }
    
    def update(self, symbol_data):
        """
        Update with new price data for each symbol and return combined results.
        
        Parameters:
        -----------
        symbol_data : dict
            Dictionary of {symbol: {'open': float, 'high': float, 'low': float, 'close': float}}
        
        Returns:
        --------
        dict
            Dictionary of {symbol: result_dict} with PO3 results and combined signal.
        """
        results = {}
        for symbol in self.symbols:
            if symbol not in symbol_data:
                continue
            data = symbol_data[symbol]
            results[symbol] = self.detectors[symbol].update(
                data['open'], data['high'], data['low'], data['close']
            )
        
        for symbol in results:
            result = results[symbol]
            result['combined_signal'] = result['po3_signal']
            for other_symbol in results:
                if symbol == other_symbol:
                    continue
                correlation_type = get_correlation_type(symbol, other_symbol)
                if correlation_type == 'none':
                    continue
                if correlation_type == 'positive':
                    if result['po3_signal'] == results[other_symbol]['po3_signal'] and result['po3_signal'] is not None:
                        result['combined_signal'] = result['po3_signal']
                    else:
                        result['combined_signal'] = None
                elif correlation_type == 'negative':
                    if (result['po3_signal'] == 'buy' and results[other_symbol]['po3_signal'] == 'sell') or \
                       (result['po3_signal'] == 'sell' and results[other_symbol]['po3_signal'] == 'buy'):
                        result['combined_signal'] = result['po3_signal']
                    else:
                        result['combined_signal'] = None
        
        return results

class PO3Detector:
    """
    Helper class for single-symbol real-time PO3 detection.
    """
    def __init__(self, po3_lookback=20, min_swing_pips=10, pip_factor=0.0001):
        self.po3_lookback = po3_lookback
        self.min_swing_pips = min_swing_pips
        self.pip_factor = pip_factor
        self.highs = deque(maxlen=po3_lookback)
        self.lows = deque(maxlen=po3_lookback)
        self.opens = deque(maxlen=po3_lookback)
        self.closes = deque(maxlen=po3_lookback)
        self.results = []

    def update(self, open, high, low, close):
        self.highs.append(high)
        self.lows.append(low)
        self.opens.append(open)
        self.closes.append(close)
        
        if len(self.highs) < self.po3_lookback:
            return {'po3_phase': None, 'po3_accumulation': False, 'po3_manipulation': False, 'po3_distribution': 'none', 'po3_signal': None}
        
        range_size = max(self.highs) - min(self.lows)
        accumulation = range_size < (10 * self.min_swing_pips * self.pip_factor)
        manipulation = low < min(self.lows) or high > max(self.highs)
        distribution = 'none'
        signal = None
        
        if manipulation:
            if low < min(self.lows) and close < open:
                distribution = 'bearish'
                signal = 'sell'
            elif high > max(self.highs) and close > open:
                distribution = 'bullish'
                signal = 'buy'
        
        phase = distribution if distribution != 'none' else ('manipulation' if manipulation else ('accumulation' if accumulation else None))
        
        result = {
            'po3_phase': phase,
            'po3_accumulation': accumulation,
            'po3_manipulation': manipulation,
            'po3_distribution': distribution,
            'po3_signal': signal
        }
        self.results.append(result)
        return result


from datetime import timedelta
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_order_blocks(
    df: pd.DataFrame,
    risk_reward_ratio: float = 2.0,
    mitigation_threshold: float = 0.0002,
    chain_window_hours: float = 4.0,
    internal_range: float = 0.001,
    sessions: list = ['London', 'Asia', 'New York']
) -> list:
    """
    Identify order blocks based on swing highs and lows (Compendium Section 3), using UTC timestamps.

    Args:
        df (pd.DataFrame): DataFrame with columns: high, low, close, time, swing_high, swing_low, bos, displacement
        risk_reward_ratio (float): Risk-reward ratio for target calculation
        mitigation_threshold (float): Price difference threshold for mitigation
        chain_window_hours (float): Time window (hours) for order block chaining
        internal_range (float): Price range for internal order block detection
        sessions (list): Trading sessions to flag ('London', 'Asia', 'New York')

    Returns:
        list: List of order block dictionaries with UTC timestamps and session flags

    Raises:
        ValueError: If required columns are missing or time is not datetime
    """
    # Input validation
    required_columns = ['high', 'low', 'close', 'time', 'swing_high', 'swing_low', 'bos', 'displacement']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame missing required columns: {required_columns}")
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        raise ValueError("Time column must be in datetime format")

    # Convert time to UTC
    df = df.copy()
    if not df['time'].dt.tz:
        df['time'] = df['time'].dt.tz_localize('Africa/Nairobi').dt.tz_convert('UTC')
    else:
        df['time'] = df['time'].dt.tz_convert('UTC')

    # Log edge cases
    swing_highs = df[df['swing_high']].reset_index()
    swing_lows = df[df['swing_low']].reset_index()
    if swing_highs.empty:
        logging.warning("No swing highs detected in the DataFrame")
    if swing_lows.empty:
        logging.warning("No swing lows detected in the DataFrame")

    # Vectorized stop calculations
    df['stop_high'] = df['high'].rolling(window=4, closed='right').max().shift(-3).fillna(df['high'])
    df['stop_low'] = df['low'].rolling(window=4, 'right').min().shift(-3).fillna(df['low'])

    obs = []

    def is_mitigated_or_breaker(entry: float, type_: str, idx: int) -> tuple[bool, bool]:
        """Check if an order block is mitigated or a breaker based on subsequent price action."""
        subsequent_data = df.loc[idx:df.index[-1], 'close']
        if type_ == 'sell' or type_ == 'breaker_buy':
            breaker = any(subsequent_data > entry)
            mitigated = any(abs(subsequent_data - entry) < mitigation_threshold)
        else:  # buy or breaker_sell
            breaker = any(subsequent_data < entry)
            mitigated = any(abs(subsequent_data - entry) < mitigation_threshold)
        return mitigated, breaker

    def is_in_session(time: pd.Timestamp, session: str) -> bool:
        """Check if timestamp is in specified session (UTC)."""
        hour = time.hour
        # Adjust for DST: July 6, 2025, is EDT (UTC-4) for New York
        if session == 'London':
            return 8 <= hour <= 17  # 08:00–17:00 UTC
        if session == 'Asia':
            return 0 <= hour <= 9   # 00:00–09:00 UTC (Tokyo)
        if session == 'New York':
            return 12 <= hour <= 21  # 12:00–21:00 UTC (EDT, July 2025)
        return False

    # Sell Order Blocks (Swing Highs)
    for i in range(len(swing_highs)):
        idx = swing_highs.loc[i, 'index']
        entry = swing_highs.loc[i, 'high']
        stop = df.loc[idx, 'stop_high']
        target = entry - (stop - entry) * risk_reward_ratio
        refined_price = (entry + swing_highs.loc[i, 'close']) / 2
        mitigated, breaker = is_mitigated_or_breaker(entry, 'sell', idx)
        time = swing_highs.loc[i, 'time']
        ob = {
            'type': 'sell' if not breaker else 'breaker_buy',
            'price': entry,
            'refined_price': refined_price,
            'stop': stop,
            'target': target,
            'time': time,
            'mitigation': mitigated and df.loc[idx, 'bos'],
            'breaker': breaker,
            'continuation': df.loc[idx, 'bos'],
            'internal': False,
            'institutional_candle': df.loc[idx, 'displacement'],
            'chain': False,
            'in_london_session': is_in_session(time, 'London') if 'London' in sessions else False,
            'in_asia_session': is_in_session(time, 'Asia') if 'Asia' in sessions else False,
            'in_new_york_session': is_in_session(time, 'New York') if 'New York' in sessions else False
        }
        obs.append(ob)

    # Buy Order Blocks (Swing Lows)
    for i in range(len(swing_lows)):
        idx = swing_lows.loc[i, 'index']
        entry = swing_lows.loc[i, 'low']
        stop = df.loc[idx, 'stop_low']
        target = entry + (entry - stop) * risk_reward_ratio
        refined_price = (entry + swing_lows.loc[i, 'close']) / 2
        mitigated, breaker = is_mitigated_or_breaker(entry, 'buy', idx)
        time = swing_lows.loc[i, 'time']
        ob = {
            'type': 'buy' if not breaker else 'breaker_sell',
            'price': entry,
            'refined_price': refined_price,
            'stop': stop,
            'target': target,
            'time': time,
            'mitigation': mitigated and df.loc[idx, 'bos'],
            'breaker': breaker,
            'continuation': df.loc[idx, 'bos'],
            'internal': False,
            'institutional_candle': df.loc[idx, 'displacement'],
            'chain': False,
            'in_london_session': is_in_session(time, 'London') if 'London' in sessions else False,
            'in_asia_session': is_in_session(time, 'Asia') if 'Asia' in sessions else False,
            'in_new_york_session': is_in_session(time, 'New York') if 'New York' in sessions else False
        }
        obs.append(ob)

    # Internal Order Block Detection
    if len(obs) >= 2:
        obs = sorted(obs, key=lambda x: x['time'])
        for i in range(1, len(obs)):
            current_ob = obs[i]
            for prev_ob in obs[:i]:
                if current_ob['type'] == prev_ob['type']:
                    price_diff = abs(current_ob['price'] - prev_ob['price'])
                    if price_diff < internal_range:
                        current_ob['internal'] = True
                        break

    # Order Block Chain
    if len(obs) >= 2:
        for i in range(1, len(obs)):
            if (
                obs[i]['type'] == obs[i-1]['type']
                and isinstance(obs[i]['time'], pd.Timestamp)
                and isinstance(obs[i-1]['time'], pd.Timestamp)
                and (obs[i]['time'] - obs[i-1]['time']) < timedelta(hours=chain_window_hours)
            ):
                obs[i]['chain'] = True

    return obs

import pandas as pd
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_fvg(df, min_fvg_size=0.0, min_imbalance_size=0.0, lookahead=1, volume_spike_factor=1.5, 
               displacement_factor=1.5, timeframe='M15', m1_fvgs=None):
    """
    Detect Fair Value Gaps (FVGs), BISI (Buy-Side Imbalance), SIBI (Sell-Side Imbalance),
    volume imbalances, breakaway gaps, premium/discount FVGs, gap refills, and equilibrium returns
    as per Compendium Section 4. Supports M15 Finnhub data for FVG detection and M1 Deriv data for real-time monitoring.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns 'high', 'low', 'close', 'time', and optionally 'volume'.
    - min_fvg_size (float): Minimum size of the FVG gap (default: 0.0).
    - min_imbalance_size (float): Minimum price movement beyond FVG for BISI/SIBI (default: 0.0).
    - lookahead (int): Number of candles to check for BISI/SIBI and equilibrium returns (default: 1).
    - volume_spike_factor (float): Factor to determine volume spike (default: 1.5x rolling mean).
    - displacement_factor (float): Factor to determine displacement candle (default: 1.5x avg range).
    - timeframe (str): Data timeframe ('M15' for Finnhub FVG detection, 'M1' for Deriv real-time monitoring).
    - m1_fvgs (list or None): List of previously detected FVGs for M1 monitoring (default: None).

    Returns:
    - tuple: (fvgs, df)
        - fvgs (list): List of dictionaries with FVG details (type, range, time, tested, volume_imbalance, breakaway).
        - df (pd.DataFrame): Modified DataFrame with new columns for FVG-related data.
    """
    # Validate inputs
    required_columns = ['high', 'low', 'close', 'time']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    if timeframe not in ['M15', 'M1']:
        raise ValueError("timeframe must be 'M15' or 'M1'")
    if lookahead < 0:
        raise ValueError("lookahead must be non-negative")

    # Create a copy of the DataFrame
    df = df.copy()

    # Ensure 'time' is datetime for resampling
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    # Handle M1 data: Aggregate to M15 if specified
    if timeframe == 'M1' and m1_fvgs is None:
        logger.info("Aggregating M1 data to M15 for FVG detection")
        df_m15 = df.resample('15min', on='time').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df.columns else lambda x: np.nan
        }).dropna().reset_index()
        df = df_m15
        timeframe = 'M15'  # Treat as M15 after aggregation

    # Initialize new columns
    df['bisi'] = False  # Buy-Side Imbalance: Close exceeds upper FVG bound (bullish)
    df['sibi'] = False  # Sell-Side Imbalance: Close below lower FVG bound (bearish)
    df['gap_refill'] = False  # True if close price is within FVG range
    df['premium_fvg'] = False  # True if FVG is above 1.618 equilibrium
    df['discount_fvg'] = False  # True if FVG is below 1.618 equilibrium
    df['volume_imbalance'] = False  # True if volume spike in FVG zone
    df['breakaway'] = False  # True if FVG is associated with a displacement candle
    df['equilibrium_return'] = False  # True if price returns to FVG range after BISI/SIBI

    # Real-time M1 monitoring mode
    if timeframe == 'M1' and m1_fvgs is not None:
        logger.info("Running M1 real-time monitoring against provided FVGs")
        fvgs = m1_fvgs  # Use provided FVGs
        latest_price = df['close'].iloc[-1]
        latest_time = df['time'].iloc[-1]
        for fvg in fvgs:
            if fvg['tested']:
                continue
            # Check for gap refill
            if fvg['range'][0] <= latest_price <= fvg['range'][1]:
                logger.info(f"Real-time: Price {latest_price} entered FVG range {fvg['range']} at {latest_time}")
                df.iloc[-1, df.columns.get_loc('gap_refill')] = True
                fvg['tested'] = True
            # Check for BISI
            elif fvg['type'] == 'bullish' and latest_price > (fvg['range'][1] + min_imbalance_size):
                logger.info(f"Real-time: BISI triggered for FVG range {fvg['range']} at {latest_time}")
                df.iloc[-1, df.columns.get_loc('bisi')] = True
            # Check for SIBI
            elif fvg['type'] == 'bearish' and latest_price < (fvg['range'][0] - min_imbalance_size):
                logger.info(f"Real-time: SIBI triggered for FVG range {fvg['range']} at {latest_time}")
                df.iloc[-1, df.columns.get_loc('sibi')] = True
            # Check for equilibrium return
            if (fvg['type'] == 'bullish' and df['bisi'].iloc[-1]) or (fvg['type'] == 'bearish' and df['sibi'].iloc[-1]):
                if fvg['range'][0] <= latest_price <= fvg['range'][1]:
                    logger.info(f"Real-time: Equilibrium return for FVG range {fvg['range']} at {latest_time}")
                    df.iloc[-1, df.columns.get_loc('equilibrium_return')] = True
        return fvgs, df

    # M15 FVG detection mode
    # Shift data for vectorized operations
    df['low_i2'] = df['low'].shift(2)
    df['high_i2'] = df['high'].shift(2)
    df['high_i1'] = df['high'].shift(1)
    df['low_i1'] = df['low'].shift(1)

    # Detect FVGs (Compendium: candle 1 low > candle 3 high for bullish, candle 1 high < candle 3 low for bearish)
    bullish_fvg = (df['low_i2'] < df['high']) & \
                  (df['high'] - df['low_i2'] >= min_fvg_size) & \
                  (df['high_i1'] < df['low_i2'])
    bearish_fvg = (df['high_i2'] > df['low']) & \
                  (df['high_i2'] - df['low'] >= min_fvg_size) & \
                  (df['low_i1'] > df['high_i2'])

    # Calculate equilibrium (Compendium: 1.618 Fibonacci line)
    df['equilibrium'] = (df['high'] + df['low']) / 1.618

    # Initialize temporary columns for FVG properties
    df['fvg_type'] = np.nan
    df['fvg_low'] = np.nan
    df['fvg_high'] = np.nan
    df['fvg_tested'] = False

    # Assign FVG properties
    df.loc[bullish_fvg, 'fvg_type'] = 'bullish'
    df.loc[bullish_fvg, 'fvg_low'] = df['low_i2']
    df.loc[bullish_fvg, 'fvg_high'] = df['high']
    df.loc[bearish_fvg, 'fvg_type'] = 'bearish'
    df.loc[bearish_fvg, 'fvg_low'] = df['low']
    df.loc[bearish_fvg, 'fvg_high'] = df['high_i2']

    # Detect premium/discount FVGs
    df.loc[df['fvg_type'].notna(), 'premium_fvg'] = df['fvg_low'] > df['equilibrium']
    df.loc[df['fvg_type'].notna(), 'discount_fvg'] = df['fvg_high'] <= df['equilibrium']

    # Detect BISI and SIBI with lookahead and threshold
    for offset in range(lookahead + 1):
        close_shift = df['close'].shift(-offset)
        df.loc[bullish_fvg, 'bisi'] |= (close_shift > (df['fvg_high'] + min_imbalance_size))
        df.loc[bearish_fvg, 'sibi'] |= (close_shift < (df['fvg_low'] - min_imbalance_size))

    # Detect gap refill
    df.loc[(df['fvg_type'].notna()) & (df['fvg_low'] <= df['close']) & (df['close'] <= df['fvg_high']), 'gap_refill'] = True
    df.loc[df['gap_refill'], 'fvg_tested'] = True

    # Detect volume imbalance (if volume column is available)
    if 'volume' in df.columns and not df['volume'].isna().all():
        df['avg_volume'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df.loc[df['fvg_type'].notna(), 'volume_imbalance'] = df['volume'] > (df['avg_volume'] * volume_spike_factor)
    else:
        logger.warning("Volume column not found or all NaN. Skipping volume imbalance detection.")

    # Detect breakaway gaps (displacement candles: candle range > avg range * displacement_factor)
    df['candle_range'] = df['high'] - df['low']
    df['avg_range'] = df['candle_range'].rolling(window=20, min_periods=1).mean()
    df.loc[df['fvg_type'].notna(), 'breakaway'] = df['candle_range'] > (df['avg_range'] * displacement_factor)

    # Detect equilibrium returns (price returns to FVG range after BISI/SIBI within lookahead)
    for offset in range(1, lookahead + 1):
        close_shift = df['close'].shift(-offset)
        df.loc[(df['bisi'] | df['sibi']) & (df['fvg_low'] <= close_shift) & (close_shift <= df['fvg_high']), 'equilibrium_return'] = True

    # Log details
    for idx in df[bullish_fvg & df['bisi']].index:
        logger.debug(f"BISI at index {idx}, close={df.loc[idx, 'close']}, FVG range=[{df.loc[idx, 'fvg_low']}, {df.loc[idx, 'fvg_high']}]")
    for idx in df[bearish_fvg & df['sibi']].index:
        logger.debug(f"SIBI at index {idx}, close={df.loc[idx, 'close']}, FVG range=[{df.loc[idx, 'fvg_low']}, {df.loc[idx, 'fvg_high']}]")
    for idx in df[df['volume_imbalance']].index:
        logger.debug(f"Volume Imbalance at index {idx}, volume={df.loc[idx, 'volume'] if 'volume' in df.columns else 'N/A'}, avg_volume={df.loc[idx, 'avg_volume'] if 'avg_volume' in df.columns else 'N/A'}")
    for idx in df[df['breakaway']].index:
        logger.debug(f"Breakaway Gap at index {idx}, candle_range={df.loc[idx, 'candle_range']}, avg_range={df.loc[idx, 'avg_range']}")
    for idx in df[df['equilibrium_return']].index:
        logger.debug(f"Equilibrium Return at index {idx}, close={df.loc[idx, 'close']}, FVG range=[{df.loc[idx, 'fvg_low']}, {df.loc[idx, 'fvg_high']}]")

    # Log summary
    num_bullish = bullish_fvg.sum()
    num_bearish = bearish_fvg.sum()
    num_bisi = df['bisi'].sum()
    num_sibi = df['sibi'].sum()
    num_volume_imbalance = df['volume_imbalance'].sum()
    num_breakaway = df['breakaway'].sum()
    num_equilibrium_return = df['equilibrium_return'].sum()
    logger.info(f"Detected {num_bullish} bullish FVGs, {num_bearish} bearish FVGs, "
                f"{num_bisi} BISI, {num_sibi} SIBI, {num_volume_imbalance} volume imbalances, "
                f"{num_breakaway} breakaway gaps, {num_equilibrium_return} equilibrium returns")

    # Create FVG list
    fvgs = []
    fvg_df = df[df['fvg_type'].notna()][['fvg_type', 'fvg_low', 'fvg_high', 'time', 'fvg_tested', 'volume_imbalance', 'breakaway']]
    for _, row in fvg_df.iterrows():
        fvg = {
            'type': row['fvg_type'],
            'range': [row['fvg_low'], row['fvg_high']],
            'time': row['time'],
            'tested': row['fvg_tested'],
            'volume_imbalance': row['volume_imbalance'],
            'breakaway': row['breakaway']
        }
        fvgs.append(fvg)

    # Drop temporary columns
    df = df.drop(['low_i2', 'high_i2', 'high_i1', 'low_i1', 'equilibrium', 'fvg_type', 'fvg_low', 'fvg_high', 
                  'fvg_tested', 'candle_range', 'avg_range', 'avg_volume'], axis=1, errors='ignore')

    return fvgs, df

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import logging

def detect_liquidity_zones(
    df: pd.DataFrame,
    equal_threshold: float = 0.0001,
    liquidity_threshold: float = 0.0005,
    max_zones: int = 10,
    atr_period: int = 14,
    use_atr: bool = False,
    volume_threshold: float = None,
    multi_candle_inducement: int = 2,
    timeframe: str = None,
    monitor_mode: bool = False,
    custom_patterns: Optional[callable] = None,
    window_size: int = 100,
    max_history: int = 1000,
    gap_tolerance: str = '30min',
    output_format: str = 'list',
    log_level: str = 'INFO'
) -> Union[List[Dict], pd.DataFrame]:
    """
    Identify liquidity pools, equal highs/lows, inducements, and resting/floating liquidity in price data.

    Parameters:
        df (pd.DataFrame): DataFrame with columns: high, low, close, time, [swing_high, swing_low, volume, atr]
        equal_threshold (float): Threshold for detecting equal highs/lows (default: 0.0001)
        liquidity_threshold (float): Threshold for classifying resting/floating liquidity (default: 0.0005)
        max_zones (int): Maximum number of zones to return, sorted by time (default: 10)
        atr_period (int): Period for ATR calculation if use_atr=True (default: 14)
        use_atr (bool): Use ATR-based thresholds instead of fixed thresholds (default: False)
        volume_threshold (float): Minimum volume for volume-based liquidity zones (default: None)
        multi_candle_inducement (int): Number of candles to check for equal highs/lows (default: 2)
        timeframe (str): Resample data to this timeframe (e.g., '15min', '1H') if provided (default: None)
        monitor_mode (bool): Monitor existing zones for price interactions in real-time (default: False)
        custom_patterns (callable): Function to detect custom liquidity patterns (default: None)
        window_size (int): Maximum candles for multi-candle inducements (default: 100)
        max_history (int): Maximum rows to process for performance (default: 1000)
        gap_tolerance (str): Maximum time gap between candles (e.g., '30min') (default: '30min')
        output_format (str): Output format, either 'list' or 'dataframe' (default: 'list')
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', etc.) (default: 'INFO')

    Returns:
        Union[List[Dict], pd.DataFrame]: Liquidity zones or monitoring results

    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    # Set up logging
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    # Input validation
    required_columns = ['high', 'low', 'close', 'time']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    df = df.copy()
    # Ensure time is datetime
    try:
        df['time'] = pd.to_datetime(df['time'])
    except ValueError:
        raise ValueError("Invalid 'time' column format; must be convertible to datetime")

    df = df.sort_values('time')
    # Limit to max_history rows
    df = df.tail(max_history)

    # Check for large time gaps
    time_diffs = df['time'].diff().dt.total_seconds().fillna(0) / 60
    if (time_diffs > pd.Timedelta(gap_tolerance).total_seconds() / 60).any():
        logger.warning(f"Time gaps larger than {gap_tolerance} detected; consider filtering data")

    # Handle missing or NaN values
    if df[required_columns].isna().any().any():
        logger.warning("NaN values detected; applying forward/backward fill")
        df[required_columns] = df[required_columns].ffill().bfill()
        if df[required_columns].isna().any().any():
            raise ValueError("Unresolvable NaN values in required columns")

    # Resample to specified timeframe
    if timeframe:
        logger.info(f"Resampling data to {timeframe}")
        agg_dict = {'high': 'max', 'low': 'min', 'close': 'last'}
        if 'volume' in df.columns:
            agg_dict['volume'] = 'sum'
        df = df.set_index('time').resample(timeframe).agg(agg_dict).dropna().reset_index()
        if 'volume' in df.columns and df['volume'].isna().all():
            df = df.drop(columns='volume')

    # Calculate ATR if requested
    if use_atr:
        logger.info("Computing ATR for dynamic thresholds")
        if 'atr' not in df.columns:
            high_low = df['high'] - df['low']
            high_close = (df[('high')] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(window=atr_period).mean()
        atr_mean = df['atr'].mean()
        if np.isnan(atr_mean):
            logger.warning("ATR calculation resulted in NaN; using default thresholds")
            atr_mean = 0
        equal_threshold = atr_mean * 0.1 or equal_threshold
        liquidity_threshold = atr_mean * 0.5 or liquidity_threshold
        logger.debug(f"Thresholds: equal={equal_threshold}, liquidity={liquidity_threshold}")

    # Fallback for swing high/low
    if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
        logger.info("Computing swing highs/lows")
        df['swing_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['swing_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        df['swing_high'] = df['swing_high'].fillna(False)
        df['swing_low'] = df['swing_low'].fillna(False)

    # Initialize columns
    df['equal_high'] = False
    df['equal_low'] = False
    df['inducement'] = False
    lz = []

    # Monitor mode: Check price interactions with existing zones
    if monitor_mode:
        logger.info("Running in monitor mode")
        if len(df) > 0:
            last_price = df['close'].iloc[-1]
            last_time = df['time'].iloc[-1]
            monitored_zones = []
            for zone in lz:
                if abs(last_price - zone['price']) <= liquidity_threshold:
                    monitored_zones.append({
                        'type': f"hit_{zone['type']}",
                        'price': zone['price'],
                        'time': last_time,
                        'status': 'hit'
                    })
            if output_format == 'dataframe':
                return pd.DataFrame(monitored_zones)
            return monitored_zones

    # Identify swing highs/lows
    recent_highs = df[df['swing_high']].nlargest(3, 'high').sort_values('time')
    recent_lows = df[df['swing_low']].nsmallest(3, 'low').sort_values('time')

    # Add swing highs/lows as liquidity zones
    for _, row in recent_highs.iterrows():
        zone = {'type': 'sell_stops', 'price': row['high'], 'time': row['time']}
        if volume_threshold is None or ('volume' in df.columns and row.get('volume', 0) >= volume_threshold):
            lz.append(zone)
    for _, row in recent_lows.iterrows():
        zone = {'type': 'buy_stops', 'price': row['low'], 'time': row['time']}
        if volume_threshold is None or ('volume' in df.columns and row.get('volume', 0) >= volume_threshold):
            lz.append(zone)

    # Detect equal highs/lows and inducements
    window_size = min(window_size, len(df))
    for i in range(multi_candle_inducement, window_size):
        recent_highs_window = df['high'].iloc[max(0, i-multi_candle_inducement):i]
        recent_lows_window = df['low'].iloc[max(0, i-multi_candle_inducement):i]
        if any(abs(df['high'].iloc[i] - h) < equal_threshold for h in recent_highs_window):
            df.at[df.index[i], 'equal_high'] = True
            if df['close'].iloc[i] < df['low'].iloc[max(0, i-multi_candle_inducement):i].min():
                df.at[df.index[i], 'inducement'] = 'bearish'
                lz.append({
                    'type': 'inducement_sell',
                    'price': df['high'].iloc[i],
                    'time': df['time'].iloc[i]
                })
        if any(abs(df['low'].iloc[i] - l) < equal_threshold for l in recent_lows_window):
            df.at[df.index[i], 'equal_low'] = True
            if df['close'].iloc[i] > df['high'].iloc[max(0, i-multi_candle_inducement):i].max():
                df.at[df.index[i], 'inducement'] = 'bullish'
                lz.append({
                    'type': 'inducement_buy',
                    'price': df['low'].iloc[i],
                    'time': df['time'].iloc[i]
                })

    # Default custom pattern: Liquidity sweep detection
    if custom_patterns is None:
        def default_custom_patterns(df):
            zones = []
            # Example: Detect liquidity sweeps (price revisits zone then breaks)
            for i in range(2, len(df)):
                if df['high'].iloc[i-2] > df['high'].iloc[i-1] and df['close'].iloc[i] < df['low'].iloc[i-2]:
                    zones.append({
                        'type': 'liquidity_sweep_sell',
                        'price': df['high'].iloc[i-2],
                        'time': df['time'].iloc[i]
                    })
                if df['low'].iloc[i-2] < df['low'].iloc[i-1] and df['close'].iloc[i] > df['high'].iloc[i-2]:
                    zones.append({
                        'type': 'liquidity_sweep_buy',
                        'price': df['low'].iloc[i-2],
                        'time': df['time'].iloc[i]
                    })
            return zones
        custom_patterns = default_custom_patterns

    # Apply custom patterns
    logger.info("Applying custom liquidity patterns")
    custom_zones = custom_patterns(df)
    lz.extend(custom_zones)

    # Classify resting/floating liquidity
    last_close = df['close'].iloc[-1]
    resting_liquidity = [
        {'type': 'resting_' + z['type'], 'price': z['price'], 'time': z['time']}
        for z in lz if abs(last_close - z['price']) > liquidity_threshold
    ]
    floating_liquidity = [
        {'type': 'floating_' + z['type'], 'price': z['price'], 'time': z['time']}
        for z in lz if abs(last_close - z['price']) <= liquidity_threshold
    ]

    # Combine zones and remove duplicates
    all_zones = lz + resting_liquidity + floating_liquidity
    unique_zones = {}
    for zone in all_zones:
        key = (zone['price'], zone['time'])
        if key not in unique_zones or 'resting_' in zone['type'] or 'floating_' in zone['type']:
            unique_zones[key] = zone
    all_zones = list(unique_zones.values())

    # Sort and limit
    all_zones = sorted(all_zones, key=lambda x: x['time'], reverse=True)[:max_zones]
    logger.info(f"Detected {len(all_zones)} liquidity zones")

    # Output format
    if output_format == 'dataframe':
        return pd.DataFrame(all_zones)
    return all_zones

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_swings_scipy(df, distance=5):
    """
    Detect swing highs and lows using scipy.signal.find_peaks.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'high' and 'low' columns.
    distance : int
        Minimum distance between peaks/troughs (in candles).
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'swing_high' and 'swing_low' columns.
    """
    df = df.copy()
    df['swing_high'] = False
    df['swing_low'] = False
    
    # Detect peaks (swing highs) and troughs (swing lows)
    peaks, _ = find_peaks(df['high'], distance=distance)
    troughs, _ = find_peaks(-df['low'], distance=distance)
    
    df.iloc[peaks, df.columns.get_loc('swing_high')] = True
    df.iloc[troughs, df.columns.get_loc('swing_low')] = True
    return df

def validate_data(df):
    """
    Validate DataFrame for chronological index and missing data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate.
    
    Raises:
    -------
    ValueError
        If index is not monotonic increasing or significant data gaps are detected.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonic increasing (chronological)")
    
    if df.index.inferred_type == "datetime64":
        # Check for significant gaps in datetime index
        time_diff = df.index.to_series().diff().dropna()
        if time_diff.max() > time_diff.median() * 5:  # Arbitrary threshold for gaps
            logging.warning("Significant gaps detected in datetime index; consider interpolating")

def calculate_ote(df, min_retracement=0.618, max_retracement=0.786, min_swing_points=3, 
                  lookback_period=None, price_col='low', swing_distance=5, plot=False):
    """
    Calculate Optimal Trade Entry (OTE) zones using Fibonacci retracement levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns 'high', 'low', and optionally 'swing_high', 'swing_low'.
    min_retracement : float, optional
        Minimum Fibonacci retracement level (default: 0.618).
    max_retracement : float, optional
        Maximum Fibonacci retracement level (default: 0.786).
    min_swing_points : int, optional
        Minimum number of swing points required (default: 3).
    lookback_period : int, optional
        Number of rows to consider for swing points (default: None, use all data).
    price_col : str, optional
        Column to use for retracement price ('low' for bullish, 'high' for bearish; default: 'low').
    swing_distance : int, optional
        Minimum distance between swing points for detection (default: 5).
    plot : bool, optional
        If True, plot price data with OTE zones and Fibonacci levels (default: False).
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns: 'ote_zone', 'ote_price', 'ote_type'.
    
    Raises:
    -------
    ValueError
        If required columns are missing, invalid parameters, or data validation fails.
    """
    # Input validation
    required_columns = ['high', 'low']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    if not (0 <= min_retracement <= max_retracement <= 1):
        raise ValueError("Retracement levels must satisfy 0 <= min_retracement <= max_retracement <= 1")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")
    
    # Validate index and data gaps
    validate_data(df)
    
    df = df.copy()
    
    # Detect swing points if not provided
    if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
        logging.info("Swing points not provided; computing with scipy.signal.find_peaks, distance=%d", swing_distance)
        df = detect_swings_scipy(df, distance=swing_distance)
    
    # Limit to lookback period if specified
    if lookback_period is not None:
        if lookback_period < min_swing_points:
            raise ValueError("lookback_period must be >= min_swing_points")
        df = df.tail(lookback_period)
    
    # Initialize output columns
    df['ote_zone'] = False
    df['ote_price'] = np.nan
    df['ote_type'] = np.nan
    
    # Get swing highs and lows
    swing_highs = df[df['swing_high']].index
    swing_lows = df[df['swing_low']].index
    
    if len(swing_highs) < 2 or len(swing_lows) < 1 or (len(swing_highs) + len(swing_lows)) < min_swing_points:
        logging.warning("Insufficient swing points for OTE calculation")
        return df
    
    # Get the last two swing highs and last swing low
    last_high_idx = swing_highs[-1]
    prev_high_idx = swing_highs[-2]
    last_low_idx = swing_lows[-1]
    
    fib_levels = {}  # Store Fibonacci levels for plotting
    swing_range = None
    
    # Bullish OTE
    if (df['high'].loc[last_high_idx] > df['low'].loc[last_low_idx] > df['high'].loc[prev_high_idx]):
        logging.info("Detected bullish OTE structure")
        swing_high = df['high'].loc[last_low_idx:last_high_idx].max()
        swing_low = df['low'].loc[last_low_idx:last_high_idx].min()
        swing_range = swing_high - swing_low
        
        if swing_range == 0:
            logging.warning("Zero swing range in bullish OTE; skipping")
            return df
        
        mask = df.index > last_high_idx
        retracement = (swing_high - df.loc[mask, price_col]) / swing_range
        ote_mask = mask & (retracement >= min_retracement) & (retracement <= max_retracement)
        
        df.loc[ote_mask, 'ote_zone'] = True
        df.loc[ote_mask, 'ote_price'] = df.loc[ote_mask, price_col]
        df.loc[ote_mask, 'ote_type'] = 'buy'
        
        # Store Fibonacci levels
        fib_levels['min'] = swing_high - swing_range * max_retracement
        fib_levels['max'] = swing_high - swing_range * min_retracement
        fib_levels['swing_high'] = swing_high
        fib_levels['swing_low'] = swing_low
    
    # Bearish OTE
    elif (df['low'].loc[last_low_idx] > df['high'].loc[last_high_idx] > df['high'].loc[prev_high_idx]):
        logging.info("Detected bearish OTE structure")
        swing_high = df['high'].loc[last_high_idx:last_low_idx].max()
        swing_low = df['low'].loc[last_high_idx:last_low_idx].min()
        swing_range = swing_high - swing_low
        
        if swing_range == 0:
            logging.warning("Zero swing range in bearish OTE; skipping")
            return df
        
        mask = df.index > last_low_idx
        retracement = (df.loc[mask, 'high'] - swing_low) / swing_range
        ote_mask = mask & (retracement >= min_retracement) & (retracement <= max_retracement)
        
        df.loc[ote_mask, 'ote_zone'] = True
        df.loc[ote_mask, 'ote_price'] = df.loc[ote_mask, 'high']
        df.loc[ote_mask, 'ote_type'] = 'sell'
        
        # Store Fibonacci levels
        fib_levels['min'] = swing_low + swing_range * min_retracement
        fib_levels['max'] = swing_low + swing_range * max_retracement
        fib_levels['swing_high'] = swing_high
        fib_levels['swing_low'] = swing_low
    
    # Enhanced plotting
    if plot:
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df['high'], label='High', color='green', alpha=0.5)
        plt.plot(df.index, df['low'], label='Low', color='red', alpha=0.5)
        
        # Plot swing points
        swing_highs_df = df[df['swing_high']]
        swing_lows_df = df[df['swing_low']]
        plt.scatter(swing_highs_df.index, swing_highs_df['high'], marker='^', color='blue', label='Swing High')
        plt.scatter(swing_lows_df.index, swing_lows_df['low'], marker='v', color='purple', label='Swing Low')
        
        # Annotate swing points
        for idx in swing_highs_df.index:
            plt.annotate(f'SH {df.loc[idx, "high"]:.2f}', (idx, df.loc[idx, 'high']), xytext=(0, 5), 
                         textcoords='offset points', color='blue')
        for idx in swing_lows_df.index:
            plt.annotate(f'SL {df.loc[idx, "low"]:.2f}', (idx, df.loc[idx, 'low']), xytext=(0, -10), 
                         textcoords='offset points', color='purple')
        
        # Plot OTE zones
        ote_zones = df[df['ote_zone']]
        plt.scatter(ote_zones.index, ote_zones['ote_price'], marker='o', color='orange', label='OTE Zone')
        for idx in ote_zones.index:
            plt.annotate(f'OTE {ote_zones.loc[idx, "ote_price"]:.2f}', (idx, ote_zones.loc[idx, 'ote_price']), 
                         xytext=(0, 5), textcoords='offset points', color='orange')
        
        # Plot Fibonacci levels
        if fib_levels:
            plt.axhline(fib_levels['min'], linestyle='--', color='cyan', alpha=0.7, 
                        label=f'Fib {min_retracement*100:.1f}%')
            plt.axhline(fib_levels['max'], linestyle='--', color='magenta', alpha=0.7, 
                        label=f'Fib {max_retracement*100:.1f}%')
            plt.axhline(fib_levels['swing_high'], linestyle=':', color='gray', alpha=0.5, label='Swing High')
            plt.axhline(fib_levels['swing_low'], linestyle=':', color='gray', alpha=0.5, label='Swing Low')
        
        plt.title('OTE Zones with Fibonacci Levels')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()
    
    return df 

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple

def analyze_market(
    symbols_data: Dict[str, pd.DataFrame],
    ote_levels: Optional[Dict[str, Tuple[float, float]]] = None,
    ote_lookback: int = 20,
    ote_tolerance: float = 0.005,
    include_rsi: bool = False
) -> pd.DataFrame:
    """
    Generate detailed Holy Grail signals by combining Type 2 (Stochastic) divergences, SMT signals, and OTE.
    Uses M15 Finnhub data, per Compendium Section 7.

    Parameters:
    -----------
    symbols_data : Dict[str, pd.DataFrame]
        Dictionary of M15 DataFrames for 'frxEURUSD', 'frxGBPUSD', 'frxUSDJPY' with 'high', 'low', 'close'.
    ote_levels : Optional[Dict[str, Tuple[float, float]]], optional (default=None)
        Dictionary of OTE levels (buy_level, sell_level) per pair. If None, calculates dynamically.
    ote_lookback : int, optional (default=20)
        Lookback period for dynamic OTE calculation (~5 hours on M15).
    ote_tolerance : float, optional (default=0.005)
        Price tolerance for OTE proximity (0.5%).
    include_rsi : bool, optional (default=False)
        Include RSI values in signal output if True.

    Returns:
    --------
    pd.DataFrame
        DataFrame with detailed Holy Grail signals, columns:
        - 'timestamp': M15 timestamp of the signal.
        - 'pair': Forex pair triggering the signal.
        - 'holy_grail_signal': 'buy', 'sell', or ''.
        - 'smt_signal': 'bullish', 'bearish', or ''.
        - 'stoch_divergence': 'type2_bullish', 'type2_bearish', or ''.
        - 'close_price': Current close price.
        - 'ote_level': OTE level used (buy or sell level).
        - 'ote_distance': Percentage distance from close price to OTE level.
        - 'macd_line': Current MACD line value.
        - 'stoch_k': Current Stochastic %K value.
        - 'rsi': Current RSI value (if include_rsi=True).
        - 'recent_high': Recent swing high (over ote_lookback).
        - 'recent_low': Recent swing low (over ote_lookback).

    Raises:
    -------
    ValueError
        If required pairs or OTE levels are missing, or data is insufficient.
    """
    required_pairs = ['frxEURUSD', 'frxGBPUSD', 'frxUSDJPY']
    if not all(pair in symbols_data for pair in required_pairs):
        raise ValueError(f"Data must include: {required_pairs}")

    # Calculate OTE levels if not provided
    if ote_levels is None:
        ote_levels = {pair: calculate_ote(symbols_data[pair], lookback=ote_lookback)
                      for pair in required_pairs}
    else:
        if not all(pair in ote_levels for pair in required_pairs):
            raise ValueError(f"OTE levels must include: {required_pairs}")

    # Run divergence detection (Type 2 for Stochastic only)
    for pair in required_pairs:
        symbols_data[pair] = detect_divergence(
            symbols_data[pair],
            include_rsi=include_rsi,
            include_macd_type2=False  # Per Compendium Section 7
        )

    # Get SMT signals
    smt_signals = detect_smt(symbols_data)

    # Initialize result DataFrame for detailed signals
    signals = []

    # Combine signals
    for t in smt_signals.index:
        smt = smt_signals.loc[t, 'smt_signal']
        if smt == '':
            continue

        for pair in required_pairs:
            if t not in symbols_data[pair].index:
                continue
            df = symbols_data[pair]
            row = df.loc[t]
            ote_buy, ote_sell = ote_levels[pair]

            # Calculate recent swing high/low
            lookback_data = df.loc[df.index <= t].tail(ote_lookback)
            recent_high = lookback_data['high'].max() if not lookback_data.empty else np.nan
            recent_low = lookback_data['low'].min() if not lookback_data.empty else np.nan

            # Holy Grail Buy
            if (smt == 'bullish' and row['stoch_divergence'] == 'type2_bullish' and
                abs(row['close'] - ote_buy) / ote_buy <= ote_tolerance):
                signals.append({
                    'timestamp': t,
                    'pair': pair,
                    'holy_grail_signal': 'buy',
                    'smt_signal': smt,
                    'stoch_divergence': row['stoch_divergence'],
                    'close_price': row['close'],
                    'ote_level': ote_buy,
                    'ote_distance': abs(row['close'] - ote_buy) / ote_buy * 100,
                    'macd_line': row['macd_line'],
                    'stoch_k': row['stoch_k'],
                    'rsi': row['rsi'] if include_rsi else np.nan,
                    'recent_high': recent_high,
                    'recent_low': recent_low
                })
            # Holy Grail Sell
            elif (smt == 'bearish' and row['stoch_divergence'] == 'type2_bearish' and
                  abs(row['close'] - ote_sell) / ote_sell <= ote_tolerance):
                signals.append({
                    'timestamp': t,
                    'pair': pair,
                    'holy_grail_signal': 'sell',
                    'smt_signal': smt,
                    'stoch_divergence': row['stoch_divergence'],
                    'close_price': row['close'],
                    'ote_level': ote_sell,
                    'ote_distance': abs(row['close'] - ote_sell) / ote_sell * 100,
                    'macd_line': row['macd_line'],
                    'stoch_k': row['stoch_k'],
                    'rsi': row['rsi'] if include_rsi else np.nan,
                    'recent_high': recent_high,
                    'recent_low': recent_low
                })

    # Convert signals to DataFrame
    result = pd.DataFrame(signals)
    if result.empty:
        result = pd.DataFrame(columns=[
            'timestamp', 'pair', 'holy_grail_signal', 'smt_signal', 'stoch_divergence',
            'close_price', 'ote_level', 'ote_distance', 'macd_line', 'stoch_k', 'rsi',
            'recent_high', 'recent_low'
        ])

    return result

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations
import logging

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def check_correlation(symbols_data, min_corr=0.7):
    """Check if all symbol pairs have correlation above min_corr based on returns."""
    try:
        for s1, s2 in combinations(symbols_data.keys(), 2):
            returns1 = symbols_data[s1]['close'].pct_change().dropna()
            returns2 = symbols_data[s2]['close'].pct_change().dropna()
            if len(returns1) < 10 or len(returns2) < 10:
                logger.warning(f"Insufficient data for correlation: {s1}, {s2}")
                return False
            corr, _ = pearsonr(returns1, returns2)
            if corr < min_corr:
                logger.warning(f"Correlation {corr:.2f} below {min_corr} for {s1}, {s2}")
                return False
        return True
    except Exception as e:
        logger.error(f"Correlation error: {e}")
        return False

def detect_market_structure(df, lookback=2, min_change=0.001):
    """Vectorized swing high/low detection. Expects 'high', 'low', 'close' columns."""
    try:
        if not {'high', 'low', 'close'}.issubset(df.columns):
            raise ValueError("Missing 'high', 'low', or 'close' columns")
        highs = df['high']
        lows = df['low']
        df['swing_high'] = (highs > highs.shift(lookback)) & (highs > highs.shift(-lookback)) & (highs > highs.shift(1) * (1 + min_change))
        df['swing_low'] = (lows < lows.shift(lookback)) & (lows < lows.shift(-lookback)) & (lows < lows.shift(1) * (1 - min_change))
        return df.fillna(False)
    except Exception as e:
        logger.error(f"Market structure error: {e}")
        return df.assign(swing_high=False, swing_low=False)

def detect_smt(symbols_data, lookback=2, min_change=0.001, min_corr=0.7):
    """
    Smart Money Technique for multiple pairs with correlation and strength.
    Args:
        symbols_data (dict): {symbol: DataFrame} with 'high', 'low', 'close', datetime index.
        lookback (int): Periods for swing detection.
        min_change (float): Min price change for swing significance.
        min_corr (float): Min correlation between pairs.
    Returns:
        list: [{'symbol': str, 'type': str, 'timestamp': datetime, 'price': float, 'strength': float}]
    """
    if not symbols_data:
        logger.warning("Empty symbols_data")
        return []

    # Check correlation
    if len(symbols_data) > 1 and not check_correlation(symbols_data, min_corr):
        logger.warning("Correlation check failed")
        return []

    # Cache swing states
    swing_states = {}
    for symbol, df in symbols_data.items():
        if not isinstance(df, pd.DataFrame) or df.empty or len(df) < (lookback * 2 + 1):
            logger.warning(f"Invalid or short DataFrame: {symbol}")
            continue
        if not df.index.is_datetime64_any_dtype():
            logger.warning(f"Non-datetime index: {symbol}")
            continue

        df = detect_market_structure(df.copy(), lookback, min_change)
        if 'swing_high' not in df.columns or 'swing_low' not in df.columns:
            logger.warning(f"No swing columns: {symbol}")
            continue

        swing_states[symbol] = {
            'high': df['swing_high'].iloc[-1],
            'low': df['swing_low'].iloc[-1],
            'time': df.index[-1],
            'price': df['close'].iloc[-1],
            'high_price': df['high'].iloc[-1],
            'low_price': df['low'].iloc[-1]
        }

    # Check time alignment
    times = {s['time'] for s in swing_states.values()}
    if len(times) > 1:
        logger.warning("Timestamp misalignment")
        return []

    # Detect SMT signals
    signals = []
    for symbol, state in swing_states.items():
        others_high = all(swing_states[s]['high'] for s in swing_states if s != symbol)
        others_low = all(swing_states[s]['low'] for s in swing_states if s != symbol)
        if state['high'] and not others_high:
            strength = abs(state['price'] - state['low_price']) / state['price']
            signals.append({
                'symbol': symbol,
                'type': 'smt_bearish',
                'timestamp': state['time'],
                'price': state['price'],
                'strength': strength
            })
        elif state['low'] and not others_low:
            strength = abs(state['high_price'] - state['price']) / state['price']
            signals.append({
                'symbol': symbol,
                'type': 'smt_bullish',
                'timestamp': state['time'],
                'price': state['price'],
                'strength': strength
            })

    return signals

import pandas as pd
import numpy as np

def calculate_tpo_profile(df, timeframe='30min', bin_size=None, num_bins=20, custom_bins=None, value_area_percent=0.7, bin_method='linear'):
    """
    Calculate TPO and Volume Profile with signal generation.
    
    Parameters:
    - df: DataFrame with 'time' (Unix seconds), 'close' (prices), 'volume' (trade volumes).
    - timeframe: Resampling timeframe (e.g., '30min', '1H').
    - bin_size: Price increment per bin (e.g., 0.5 for $0.50 bins). If None, uses num_bins.
    - num_bins: Number of bins if bin_size is None.
    - custom_bins: List of custom bin edges (overrides bin_size and num_bins).
    - value_area_percent: Percentage of volume for value area (default: 0.7).
    - bin_method: 'linear' (evenly spaced bins) or 'quantile' (equal data bins).
    
    Returns:
    - dict: Contains 'tpo' (time-based price counts), 'poc' (price bin midpoint with max volume),
            'value_area' (price bin midpoints for value area), 'vah' (value area high),
            'val' (value area low), 'signal' (buy, sell, or neutral).
    """
    # Input validation
    required_columns = ['time', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    if not np.issubdtype(df['close'].dtype, np.number) or not np.issubdtype(df['volume'].dtype, np.number):
        raise ValueError("'close' and 'volume' must be numeric")
    
    if df['volume'].lt(0).any():
        raise ValueError("Volume contains negative values")
    
    # Copy DataFrame to avoid modifying the original
    df = df.copy()
    
    # Convert time to datetime and set as index
    try:
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
    except Exception as e:
        raise ValueError(f"Failed to convert 'time' to datetime: {e}")
    
    # Validate timeframe
    try:
        pd.Timedelta(timeframe)
    except ValueError:
        raise ValueError(f"Invalid timeframe: {timeframe}. Use pandas offset strings (e.g., '30min', '1H').")
    
    # Calculate bin edges
    price_min, price_max = df['close'].min(), df['close'].max()
    if price_min == price_max:
        raise ValueError("Price range is zero; cannot create bins")
    
    if custom_bins is not None:
        bins = np.array(custom_bins)
        if not (np.all(np.diff(bins) > 0) and bins[0] <= price_min and bins[-1] >= price_max):
            raise ValueError("Custom bins must be monotonically increasing and cover the price range")
    elif bin_size is not None:
        bins = np.arange(price_min, price_max + bin_size, bin_size)
    elif bin_method == 'quantile':
        bins = pd.qcut(df['close'], q=num_bins, duplicates='drop').cat.categories.left.to_list()
        bins = np.append(bins, pd.qcut(df['close'], q=num_bins, duplicates='drop').cat.categories.right.max())
    else:
        bins = np.linspace(price_min, price_max, num_bins + 1)
    
    # Bin midpoints for labeling
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    
    # Calculate TPO
    tpo = {}
    for period, group in df.resample(timeframe):
        if not group.empty:
            prices = pd.cut(group['close'], bins=bins, include_lowest=True, labels=bin_midpoints)
            tpo[period] = prices.value_counts().to_dict()
    
    # Calculate Volume Profile
    volume_profile = df['volume'].groupby(pd.cut(df['close'], bins=bins, include_lowest=True, labels=bin_midpoints)).sum()
    
    # Point of Control (POC)
    poc = float(volume_profile.idxmax()) if not volume_profile.empty else None
    
    # Value Area
    if not volume_profile.empty:
        sorted_vol = volume_profile.sort_values(ascending=False)
        cumsum_vol = sorted_vol.cumsum()
        total_vol = sorted_vol.sum()
        value_area = sorted_vol[cumsum_vol <= total_vol * value_area_percent].index
        vah = float(max(value_area, default=None)) if not value_area.empty else None
        val = float(min(value_area, default=None)) if not value_area.empty else None
    else:
        value_area, vah, val = pd.Index([]), None, None
    
    # Generate Signal
    signal = 'neutral'
    if poc is not None and vah is not None and val is not None:
        current_price = df['close'].iloc[-1]  # Latest price
        # Simple breakout/breakdown logic
        if current_price > vah:
            signal = 'buy'
        elif current_price < val:
            signal = 'sell'
        else:
            # Check volume trend near POC
            recent_data = df.tail(5)  # Last 5 data points for volume trend
            volume_trend = recent_data['volume'].diff().mean()
            if abs(current_price - poc) < (vah - val) * 0.1:  # Price near POC
                if volume_trend > 0:
                    signal = 'buy'  # Increasing volume near POC
                elif volume_trend < 0:
                    signal = 'sell'  # Decreasing volume near POC
    
    return {
        'tpo': tpo,
        'poc': poc,
        'value_area': value_area,
        'vah': vah,
        'val': val,
        'signal': signal
    }


import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pandas.api.types import is_datetime64_any_dtype
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class SignalType(Enum):
    BUY = "turtle_soup_buy"
    SELL = "turtle_soup_sell"

@dataclass
class TradeSignal:
    type: SignalType
    price: float
    stop_loss: float
    take_profit: float
    timestamp: Union[datetime, int]
    index: int

def detect_turtle_soup(
    df: pd.DataFrame,
    rr_ratio: float = 2.0,
    pip_buffer: float = 10.0,
    pip_value: float = 0.0001,
    lookback: int = 20,
    atr_period: int = 14,
    min_breakout_factor: float = 0.5,
    instrument: str = "EURUSD"
) -> List[TradeSignal]:
    """
    Advanced Turtle Soup model for detecting fake breakouts with robust error handling and dynamic parameters.

    Args:
        df: DataFrame with OHLC, swing_high, swing_low, and market_structure columns.
        rr_ratio: Risk-reward ratio for take-profit calculation.
        pip_buffer: Buffer in pips for stop loss placement.
        pip_value: Pip value for the instrument (e.g., 0.0001 for EURUSD, 0.01 for USDJPY).
        lookback: Lookback period for swing point detection.
        atr_period: Period for ATR calculation for volatility-based filtering.
        min_breakout_factor: Minimum breakout size as a fraction of ATR.
        instrument: Instrument name for pip value configuration.

    Returns:
        List of TradeSignal objects containing trade details.

    Raises:
        ValueError: If DataFrame is invalid or missing required columns.
        TypeError: If input types are incorrect.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")
    if len(df) < max(5, lookback, atr_period):
        raise ValueError(f"DataFrame must have at least {max(5, lookback, atr_period)} rows")
    
    required_columns = {'open', 'high', 'low', 'close', 'swing_high', 'swing_low', 'market_structure'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    if not (isinstance(pip_buffer, (int, float)) and pip_buffer >= 0):
        raise ValueError("pip_buffer must be a non-negative number")
    if not (isinstance(rr_ratio, (int, float)) and rr_ratio > 0):
        raise ValueError("rr_ratio must be a positive number")
    if not (isinstance(lookback, int) and lookback > 0):
        raise ValueError("lookback must be a positive integer")
    if not (isinstance(atr_period, int) and atr_period > 0):
        raise ValueError("atr_period must be a positive integer")

    # Ensure numeric columns
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains non-numeric or NaN values")

    # Ensure boolean swing columns
    for col in ['swing_high', 'swing_low']:
        if not df[col].dtype == bool:
            raise ValueError(f"Column '{col}' must be boolean")

    # Ensure valid market structure values
    valid_structures = {'bullish', 'bearish'}
    if not df['market_structure'].isin(valid_structures).all():
        raise ValueError("market_structure must contain only 'bullish' or 'bearish'")

    # Calculate ATR for volatility-based filtering
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=atr_period, min_periods=atr_period).mean()

    signals: List[TradeSignal] = []
    
    # Vectorized swing point detection
    swing_highs = df[df['swing_high']]['high']
    swing_lows = df[df['swing_low']]['low']
    
    for i in range(max(lookback, atr_period), len(df)):
        # Get recent swing points
        recent_highs = swing_highs[swing_highs.index <= df.index[i]].tail(2)
        recent_lows = swing_lows[swing_lows.index <= df.index[i]].tail(2)
        
        prev_swing_high = recent_highs.iloc[-2] if len(recent_highs) >= 2 else df['high'].iloc[i-lookback:i].max()
        prev_swing_low = recent_lows.iloc[-2] if len(recent_lows) >= 2 else df['low'].iloc[i-lookback:i].min()
        
        # Skip if swing points are NaN
        if pd.isna(prev_swing_high) or pd.isna(prev_swing_low):
            continue

        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_open = df['open'].iloc[i]
        current_close = df['close'].iloc[i]
        prev_market_structure = df['market_structure'].iloc[i-1]
        atr = df['atr'].iloc[i]

        # Turtle Soup Sell Signal
        if (
            current_high > prev_swing_high
            and current_close < current_open
            and prev_market_structure == 'bullish'
            and (current_high - prev_swing_high) >= (min_breakout_factor * atr)
        ):
            stop_loss = current_high + (pip_buffer * pip_value)
            risk = stop_loss - current_close
            take_profit = current_close - (rr_ratio * risk)
            signals.append(TradeSignal(
                type=SignalType.SELL,
                price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=df.index[i],
                index=i
            ))

        # Turtle Soup Buy Signal
        elif (
            current_low < prev_swing_low
            and current_close > current_open
            and prev_market_structure == 'bearish'
            and (prev_swing_low - current_low) >= (min_breakout_factor * atr)
        ):
            stop_loss = current_low - (pip_buffer * pip_value)
            risk = current_close - stop_loss
            take_profit = current_close + (rr_ratio * risk)
            signals.append(TradeSignal(
                type=SignalType.BUY,
                price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timestamp=df.index[i],
                index=i
            ))

    return signals

import pandas as pd
from datetime import time, timedelta
import logging
from typing import Callable, Optional, Dict, List

def is_in_kill_zone(session: str, timestamp: pd.Timestamp, session_times: Dict[str, tuple]) -> bool:
    """Check if timestamp falls within the specified session."""
    sessions = {
        'Asian': (time(0, 0), time(4, 0)),  # UTC, adjustable
    }
    start, end = session_times.get(session, sessions.get(session, (time(0, 0), time(0, 0))))
    t = timestamp.time()
    return start <= t <= end

def calculate_displacement(df: pd.DataFrame, atr_period: int = 14, atr_threshold: float = 1.5) -> pd.Series:
    """Calculate displacement based on ATR."""
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()
    return (df['high'] - df['low']) > (atr * atr_threshold)

def calculate_trend(df: pd.DataFrame, trend_period: int = 20) -> pd.Series:
    """Determine trend direction using SMA."""
    sma = df['close'].rolling(window=trend_period).mean()
    return df['close'] > sma  # True for uptrend, False for downtrend

def get_pip_value(instrument: str, price: float) -> float:
    """Return pip value based on instrument type or price level."""
    pip_values = {
        'EURUSD': 0.0001,
        'USDJPY': 0.01,
        'SP500': 0.25
    }
    return pip_values.get(instrument, 0.0001 if price < 10 else 0.01)

def detect_judas_swing(
    df: pd.DataFrame,
    instrument: Optional[str] = None,
    risk_reward_ratio: float = 2.0,
    session_times: Optional[Dict[str, tuple]] = None,
    min_volume: Optional[float] = None,
    cooldown_candles: int = 5,
    log_level: str = 'INFO',
    calculate_displacement_if_missing: bool = True,
    atr_period: int = 14,
    atr_stop_multiplier: float = 1.5,
    trend_period: int = 20,
    use_trend_filter: bool = True,
    timeframe_minutes: Optional[int] = None,
    signal_callback: Optional[Callable[[dict], None]] = None,
    batch_size: int = 10000
) -> List[dict]:
    """
    Judas Swing model for pre-session manipulation (Compendium Section 8).
    
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'open', 'close', 'timestamp', and optionally 'displacement', 'volume'.
        instrument (str): Financial instrument (e.g., 'EURUSD', 'USDJPY').
        risk_reward_ratio (float): Reward-to-risk ratio for take-profit.
        session_times (dict): Custom session times (e.g., {'Asian': (time(0,0), time(4,0))}).
        min_volume (float): Minimum volume to filter low-liquidity candles.
        cooldown_candles (int): Minimum candles between signals.
        log_level (str): Logging level ('DEBUG', 'INFO', etc.).
        calculate_displacement_if_missing (bool): Calculate displacement if not in DataFrame.
        atr_period (int): Period for ATR calculation (displacement and stop-loss).
        atr_stop_multiplier (float): Multiplier for ATR-based stop-loss.
        trend_period (int): Period for trend calculation (SMA).
        use_trend_filter (bool): Filter signals by trend direction.
        timeframe_minutes (int): Expected candle timeframe in minutes for validation.
        signal_callback (Callable): Optional function to handle signals in real-time.
        batch_size (int): Number of rows to process per batch for large datasets.
    
    Returns:
        List[dict]: List of trade signals with type, price, stop_loss, take_profit, timestamp, index, and strength.
    """
    logging.basicConfig(level=log_level)
    logger = logging.getLogger(__name__)

    # Data validation
    required_cols = ['high', 'low', 'open', 'close', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {set(required_cols) - set(df.columns)}")
    
    df = df.dropna(subset=required_cols).copy()
    if len(df) < max(6, atr_period, trend_period):
        logger.warning("Insufficient data for analysis")
        return []

    # Timeframe validation
    if timeframe_minutes:
        time_diffs = df['timestamp'].diff().dt.total_seconds().iloc[1:] / 60
        if not (time_diffs - timeframe_minutes).abs().mean() < 1:
            logger.warning(f"Inconsistent timeframe detected; expected {timeframe_minutes} minutes")

    # Calculate displacement if missing
    if 'displacement' not in df.columns and calculate_displacement_if_missing:
        df['displacement'] = calculate_displacement(df, atr_period=atr_period)

    # Volume validation
    if min_volume is not None and 'volume' not in df.columns:
        raise ValueError("Volume column required when min_volume is specified")
    
    # Calculate ATR for stop-loss
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    # Trend filter
    if use_trend_filter:
        df['uptrend'] = calculate_trend(df, trend_period=trend_period)

    pip_value = get_pip_value(instrument, df['close'].iloc[-1]) if instrument else 0.0001
    session_times = session_times or {}
    signals = []
    last_signal_idx = -cooldown_candles - 1
    min_distance = 5 * pip_value

    # Batch processing
    for start_idx in range(0, len(df), batch_size):
        batch_df = df.iloc[start_idx:start_idx + batch_size].copy()
        
        # Vectorized conditions
        batch_df['in_kill_zone'] = batch_df['timestamp'].apply(lambda x: is_in_kill_zone('Asian', x, session_times))
        batch_df['higher_high'] = batch_df['high'] > batch_df['high'].shift(1)
        batch_df['lower_low'] = batch_df['low'] < batch_df['low'].shift(1)
        batch_df['bearish_candle'] = batch_df['close'] < batch_df['open']
        batch_df['bullish_candle'] = batch_df['close'] > batch_df['open']
        batch_df['valid_volume'] = (batch_df['volume'] >= min_volume) if min_volume is not None else True

        # Sell signals
        sell_mask = (batch_df['in_kill_zone'] & batch_df['displacement'] & batch_df['higher_high'] & 
                     batch_df['bearish_candle'] & batch_df['valid_volume'])
        if use_trend_filter:
            sell_mask &= ~batch_df['uptrend']  # Only sell in downtrend or no trend
        for idx in batch_df[sell_mask].index:
            if idx - last_signal_idx <= cooldown_candles:
                continue
            price = batch_df['close'].iloc[idx]
            atr_value = atr.iloc[idx]
            stop_loss = price + (atr_value * atr_stop_multiplier)
            risk = stop_loss - price
            if risk < min_distance or price <= 0 or stop_loss <= 0:
                continue
            take_profit = price - (risk * risk_reward_ratio)
            if take_profit <= 0:
                continue
            signal = {
                'type': 'judas_swing_sell',
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': batch_df['timestamp'].iloc[idx],
                'index': idx,
                'strength': (batch_df['high'].iloc[idx] - batch_df['low'].iloc[idx]) / atr_value
            }
            signals.append(signal)
            if signal_callback:
                signal_callback(signal)
            last_signal_idx = idx

        # Buy signals
        buy_mask = (batch_df['in_kill_zone'] & batch_df['displacement'] & batch_df['lower_low'] & 
                    batch_df['bullish_candle'] & batch_df['valid_volume'])
        if use_trend_filter:
            buy_mask &= batch_df['uptrend']  # Only buy in uptrend
        for idx in batch_df[buy_mask].index:
            if idx - last_signal_idx <= cooldown_candles:
                continue
            price = batch_df['close'].iloc[idx]
            atr_value = atr.iloc[idx]
            stop_loss = price - (atr_value * atr_stop_multiplier)
            risk = price - stop_loss
            if risk < min_distance or price <= 0 or stop_loss <= 0:
                continue
            take_profit = price + (risk * risk_reward_ratio)
            if take_profit <= 0:
                continue
            signal = {
                'type': 'judas_swing_buy',
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': batch_df['timestamp'].iloc[idx],
                'index': idx,
                'strength': (batch_df['high'].iloc[idx] - batch_df['low'].iloc[idx]) / atr_value
            }
            signals.append(signal)
            if signal_callback:
                signal_callback(signal)
            last_signal_idx = idx

    logger.info(f"Generated {len(signals)} signals")
    return sorted(signals, key=lambda x: x['timestamp'])

import pandas as pd
from datetime import datetime
import pytz
from typing import Optional, Dict, Union

def detect_daily_high_low_engine(
    df_d1: pd.DataFrame,
    offset_pips: float = 20.0,
    pip_size: float = 0.0001,
    lookback: int = 5,
    time_threshold_hour: int = 13,
    timezone: str = "UTC"
) -> Optional[Dict[str, Union[str, float]]]:
    """
    Predict daily high/low price before a specified UTC hour (default: 13:00 UTC).
    Optimized for robustness, flexibility, and financial trading applications.

    Args:
        df_d1 (pd.DataFrame): Daily DataFrame with 'high', 'low', 'po3_distribution' columns.
        offset_pips (float): Offset in pips to adjust predicted price (default: 20.0).
        pip_size (float): Pip size for the instrument (default: 0.0001 for forex).
        lookback (int): Number of days to analyze for high/low (default: 5).
        time_threshold_hour (int): UTC hour before which predictions are made (default: 13).
        timezone (str): Timezone for time check (default: 'UTC').

    Returns:
        dict or None: {'type': 'daily_high' or 'daily_low', 'price': float} or None if no prediction.

    Raises:
        ValueError: If DataFrame is invalid or missing required columns.
        TypeError: If 'high' or 'low' columns are non-numeric.
    """
    # Input validation
    if not isinstance(df_d1, pd.DataFrame) or df_d1.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame")
    
    required_columns = {'high', 'low', 'po3_distribution'}
    if not required_columns.issubset(df_d1.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    if not (pd.api.types.is_numeric_dtype(df_d1['high']) and pd.api.types.is_numeric_dtype(df_d1['low'])):
        raise TypeError("'high' and 'low' columns must be numeric")
    
    if lookback < 1:
        raise ValueError("lookback must be at least 1")
    
    # Check for NaN in critical columns
    if df_d1[['high', 'low']].isna().any().any() or pd.isna(df_d1['po3_distribution'].iloc[-1]):
        return None
    
    # Time check
    try:
        tz = pytz.timezone(timezone)
    except pytz.exceptions.UnknownTimeZoneError:
        raise ValueError(f"Invalid timezone: {timezone}")
    
    now = datetime.now(tz)
    if now.hour >= time_threshold_hour:
        return None
    
    # Get recent data (safely handle small DataFrames)
    lookback = min(lookback, len(df_d1))
    recent = df_d1.iloc[-lookback:]
    
    # Calculate offset
    offset = offset_pips * pip_size
    
    # Get and normalize po3_distribution
    po3 = str(df_d1['po3_distribution'].iloc[-1]).lower()
    
    # Prediction logic
    if po3 == 'bullish':
        return {'type': 'daily_high', 'price': round(recent['high'].max() + offset, 5)}
    elif po3 == 'bearish':
        return {'type': 'daily_low', 'price': round(recent['low'].min() - offset, 5)}
    
    return None

from enum import Enum
from datetime import datetime, time
import pytz
from functools import lru_cache
from typing import Optional, Dict, Tuple, Union

class TradingSession(Enum):
    LONDON = "London"
    NEW_YORK = "NewYork"

KILL_ZONES_DEFAULT = {
    TradingSession.LONDON: (7, 9),
    TradingSession.NEW_YORK: (13, 15),
}

@lru_cache(maxsize=32)  # Cache recent checks
def is_in_kill_zone(
    session: Optional[TradingSession] = None,
    kill_zones: Optional[Dict[TradingSession, Tuple[int, int]]] = None,
    verbose: bool = False,
) -> bool:
    """Check if current UTC time is within a kill zone.

    Args:
        session: TradingSession enum (e.g., TradingSession.LONDON).
        kill_zones: Custom kill zones (defaults to KILL_ZONES_DEFAULT).
        verbose: If True, logs the check result.

    Returns:
        bool: True if in a kill zone.
    """
    kill_zones = kill_zones or KILL_ZONES_DEFAULT
    now = datetime.now(pytz.UTC)
    current_hour = now.hour

    if session:
        if session not in kill_zones:
            raise ValueError(f"Invalid session. Must be one of {list(kill_zones.keys())}")
        start, end = kill_zones[session]
        in_zone = (start <= current_hour < end) if start < end else (current_hour >= start or current_hour < end)
    else:
        in_zone = any(
            (start <= current_hour < end) if start < end else (current_hour >= start or current_hour < end)
            for start, end in kill_zones.values()
        )

    if verbose:
        print(f"Kill Zone Check: Session={session}, InZone={in_zone} (Current UTC Hour: {current_hour})")
    return in_zone

from typing import Tuple, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans

# Predefined constants (customize for your strategy)
DEFAULT_LEVELS = [0.0000, 0.2500, 0.5000, 0.7500, 1.0000]
LEVEL_WEIGHTS = {
    0.0000: 1.0,   # Whole numbers = strongest
    0.5000: 0.9,
    0.2500: 0.7,
    0.7500: 0.7,
    0.6180: 0.6,   # Fibonacci
    0.3820: 0.6,
}
TIMEFRAME_THRESHOLDS = {
    "M1": 0.0010,
    "M5": 0.0008,
    "H1": 0.0005,
    "D1": 0.0003,
}

def check_institutional_level(
    price: float,
    symbol: Optional[str] = None,
    timeframe: str = "H1",
    volume_check: bool = False,
    dynamic_levels: bool = False,
    price_history: Optional[list] = None,
) -> Dict[str, float or bool]:
    """
    Advanced institutional level detection with:
    - Dynamic thresholds (volatility/timeframe-aware)
    - Level significance weighting
    - Volume/orderbook confirmation
    - Adaptive level generation (K-means clustering)
    
    Args:
        price: Price to check against levels.
        symbol: Instrument symbol (e.g., "EURUSD"). Required for volume checks.
        timeframe: Chart timeframe (e.g., "H1"). Affects threshold.
        volume_check: If True, requires volume spike confirmation.
        dynamic_levels: If True, uses K-means to auto-detect levels.
        price_history: Historical prices for dynamic level generation.
    
    Returns:
        Dict with keys: 
            is_near (bool), 
            nearest_level (float), 
            distance (float), 
            confidence (float), 
            is_confirmed (bool),
            threshold (float)
    """
    # --- 1. Determine Levels ---
    if dynamic_levels and price_history:
        levels = detect_cluster_levels(price_history)
    else:
        levels = DEFAULT_LEVELS.copy()
        if symbol and symbol.endswith(".JPY"):
            levels.extend([100.0, 200.0])  # JPY-specific levels

    # --- 2. Calculate Dynamic Threshold ---
    threshold = TIMEFRAME_THRESHOLDS.get(timeframe, 0.0005)
    if symbol:
        threshold *= get_volatility_factor(symbol)  # Scale by recent volatility

    # --- 3. Find Nearest Level ---
    nearest_level = min(levels, key=lambda x: abs(x - price))
    distance = abs(price - nearest_level)
    is_near = distance <= threshold

    # --- 4. Calculate Confidence Score ---
    weight = LEVEL_WEIGHTS.get(nearest_level, 0.5)
    confidence = weight * (1 - (distance / threshold))

    # --- 5. Volume/Orderbook Confirmation ---
    is_confirmed = True
    if volume_check and symbol:
        is_confirmed = is_level_confirmed(symbol, nearest_level)

    return {
        "is_near": is_near and is_confirmed,
        "nearest_level": nearest_level,
        "distance": distance,
        "confidence": round(confidence, 4),
        "is_confirmed": is_confirmed,
        "threshold": threshold,
    }

# ===== Helper Functions =====
def detect_cluster_levels(prices: list, num_clusters: int = 5) -> list:
    """Auto-detect significant price levels using K-means clustering."""
    if len(prices) < num_clusters:
        return DEFAULT_LEVELS
    kmeans = KMeans(n_clusters=num_clusters).fit(np.array(prices).reshape(-1, 1))
    return sorted([x[0] for x in kmeans.cluster_centers_])

def get_volatility_factor(symbol: str, lookback: int = 14) -> float:
    """Scale threshold based on recent volatility (e.g., 1.5x for high volatility)."""
    atr = get_atr(symbol, lookback)  # Replace with your ATR calculation
    return min(max(atr / 0.0010, 0.5), 2.0)  # Clamp between 0.5x and 2.0x

def is_level_confirmed(symbol: str, level: float) -> bool:
    """Check if there's elevated volume/liquidity at the level."""
    # Replace with your data source (e.g., orderbook/volume API)
    volume = get_volume_at_level(symbol, level)
    return volume > 1.2 * get_average_volume(symbol)

# === RISK MANAGEMENT ===
LOT_SIZE = 100  # Max lot size cap
STANDARD_LOT_UNITS = 100000

def calculate_position_size(entry, stop, 
                            account_balance=10000, 
                            risk_pct=None, 
                            fixed_risk=None, 
                            target_price=None,
                            symbol='EURUSD',
                            return_units=False,
                            return_both=False,
                            verbose=False):
    """
    Enhanced position size calculator with RR ratio, flexible risk input, and symbol auto-detection.
    
    Parameters:
    - entry: float - Trade entry price
    - stop: float - Stop-loss price
    - account_balance: float - Account size in base currency
    - risk_pct: float - Risk per trade as percentage of account
    - fixed_risk: float - Absolute risk per trade (overrides risk_pct)
    - target_price: float - Optional for reward-risk ratio estimation
    - symbol: str - Currency pair (e.g., 'USDJPY', 'GBPUSD') for pip size inference
    - return_units: bool - Return position size in units instead of lots
    - return_both: bool - Return both lots and units
    - verbose: bool - Print detailed output

    Returns:
    - float or dict: Position size in lots, units, or both
    """
    
    # 1. Validate input
    if entry == stop:
        raise ValueError("Entry and stop prices cannot be the same.")

    # 2. Determine pip size by instrument
    pip_size = 0.01 if symbol.endswith('JPY') else 0.0001
    pip_value = 10  # Assume $10 per pip per standard lot (adjust if needed)

    # 3. Determine risk amount
    if fixed_risk is not None:
        risk_amount = fixed_risk
    elif risk_pct is not None:
        risk_amount = account_balance * (risk_pct / 100)
    else:
        raise ValueError("Provide either risk_pct or fixed_risk.")

    # 4. Calculate pip risk
    risk_pips = abs(entry - stop) / pip_size
    if risk_pips == 0:
        raise ValueError("Risk in pips is zero. Check stop/entry.")

    # 5. Calculate position size in lots
    position_size_lots = risk_amount / (risk_pips * pip_value)
    position_size_lots = min(position_size_lots, LOT_SIZE)
    position_size_lots = round(position_size_lots, 2)

    # 6. Convert to units if needed
    position_size_units = int(position_size_lots * STANDARD_LOT_UNITS)

    # 7. Optional R:R ratio estimate
    rr_ratio = None
    if target_price is not None:
        reward_pips = abs(target_price - entry) / pip_size
        reward_amount = reward_pips * pip_value * position_size_lots
        rr_ratio = round(reward_amount / risk_amount, 2)

    # 8. Verbose breakdown
    if verbose:
        print(f"Symbol: {symbol}")
        print(f"Account Balance: ${account_balance}")
        print(f"Risk Amount: ${risk_amount}")
        print(f"Pip Size: {pip_size}")
        print(f"Risk in Pips: {risk_pips}")
        print(f"Lot Size: {position_size_lots} lots")
        print(f"Units: {position_size_units}")
        if rr_ratio is not None:
            print(f"Reward-Risk Ratio: {rr_ratio}:1")

    # 9. Return value(s)
    if return_both:
        return {
            "lots": position_size_lots,
            "units": position_size_units,
            "rr_ratio": rr_ratio
        }
    elif return_units:
        return position_size_units
    else:
        return position_size_lots

from datetime import datetime, timedelta
import pytz
import logging

# === CONFIGURATION ===
TIMEZONE = pytz.UTC  # or use pytz.timezone("Africa/Nairobi") for local time
MAX_TRADES_PER_DAY = 5
SESSION_FILTER = True

# Kill zone session definitions (in UTC time)
KILL_ZONES = {
    "london": {"start": "07:00", "end": "10:00"},
    "new_york": {"start": "13:00", "end": "16:00"},
    "asian": {"start": "00:00", "end": "03:00"}
}

# Enable or disable specific sessions
ACTIVE_SESSIONS = ["london", "new_york", "asian"]

# === STATE ===
daily_stats = {
    'last_trade_time': None,
    'trades_today': 0,
    'wins': 0,
    'losses': 0
}

# === LOGGER ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# === SESSION FILTER ===

def is_in_kill_zone():
    now_utc = datetime.now(TIMEZONE).time()

    for session_name in ACTIVE_SESSIONS:
        zone = KILL_ZONES.get(session_name)
        if not zone:
            continue

        start_time = datetime.strptime(zone["start"], "%H:%M").time()
        end_time = datetime.strptime(zone["end"], "%H:%M").time()

        if start_time <= now_utc <= end_time:
            logger.info(f"In {session_name.title()} Kill Zone ({zone['start']} - {zone['end']})")
            return True

    logger.info("Not in any active Kill Zone")
    return False


# === RISK CONTROL ===

import json
import os
from datetime import datetime, timedelta
import pytz
import logging

# === CONFIGURATION ===
TIMEZONE = pytz.UTC
MAX_DRAWDOWN_PCT = 3.0       # Daily drawdown % cap
MAX_WEEKLY_DRAWDOWN_PCT = 6.0
MAX_MONTHLY_DRAWDOWN_PCT = 12.0
STATS_FILE = "daily_stats.json"

# === LOGGER SETUP ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# === STATE ===
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            return json.load(f)
    return {
        'last_trade_time': None,
        'trades_today': 0,
        'wins': 0,
        'losses': 0,
        'drawdown_locked': False,
        'weekly_loss': 0,
        'monthly_loss': 0,
        'week_start': None,
        'month_start': None
    }

def save_stats():
    with open(STATS_FILE, "w") as f:
        json.dump(daily_stats, f, indent=2, default=str)

daily_stats = load_stats()

# === RESET TIMEFRAMES ===
def reset_timeframes(now):
    # Daily reset
    last = daily_stats.get('last_trade_time')
    if last:
        last_dt = datetime.fromisoformat(last)
        if (now - last_dt).days >= 1:
            logger.info("✅ Daily stats reset.")
            daily_stats['trades_today'] = 0
            daily_stats['losses'] = 0
            daily_stats['wins'] = 0
            daily_stats['drawdown_locked'] = False

    # Weekly reset
    if not daily_stats.get("week_start"):
        daily_stats["week_start"] = now.date().isoformat()
    else:
        start = datetime.fromisoformat(daily_stats["week_start"])
        if (now - start).days >= 7:
            logger.info("✅ Weekly stats reset.")
            daily_stats["weekly_loss"] = 0
            daily_stats["week_start"] = now.date().isoformat()

    # Monthly reset
    if not daily_stats.get("month_start"):
        daily_stats["month_start"] = now.date().isoformat()
    else:
        start = datetime.fromisoformat(daily_stats["month_start"])
        if now.month != start.month:
            logger.info("✅ Monthly stats reset.")
            daily_stats["monthly_loss"] = 0
            daily_stats["month_start"] = now.date().isoformat()

    save_stats()

# === CHECK DRAWDOWN ===
def check_max_drawdown(account_balance):
    daily_loss = daily_stats.get("losses", 0)
    weekly_loss = daily_stats.get("weekly_loss", 0)
    monthly_loss = daily_stats.get("monthly_loss", 0)

    max_daily = account_balance * (MAX_DRAWDOWN_PCT / 100)
    max_weekly = account_balance * (MAX_WEEKLY_DRAWDOWN_PCT / 100)
    max_monthly = account_balance * (MAX_MONTHLY_DRAWDOWN_PCT / 100)

    if daily_loss >= max_daily:
        logger.warning(f"🚫 DAILY drawdown breached: ${daily_loss:.2f} ≥ ${max_daily:.2f}")
        daily_stats['drawdown_locked'] = True
        save_stats()
        return False

    if weekly_loss >= max_weekly:
        logger.warning(f"🚫 WEEKLY drawdown breached: ${weekly_loss:.2f} ≥ ${max_weekly:.2f}")
        daily_stats['drawdown_locked'] = True
        save_stats()
        return False

    if monthly_loss >= max_monthly:
        logger.warning(f"🚫 MONTHLY drawdown breached: ${monthly_loss:.2f} ≥ ${max_monthly:.2f}")
        daily_stats['drawdown_locked'] = True
        save_stats()
        return False

    return True

# === CAN TRADE ===
def can_trade(account_balance):
    now = datetime.now(TIMEZONE)
    reset_timeframes(now)

    if daily_stats.get("drawdown_locked"):
        logger.warning("🚫 Trading locked due to drawdown.")
        return {
            "can_trade": False,
            "reason": "Drawdown lock"
        }

    if not check_max_drawdown(account_balance):
        return {
            "can_trade": False,
            "reason": "Drawdown breached"
        }

    return {
        "can_trade": True,
        "reason": "OK"
    }

# === RECORD TRADE OUTCOME ===
def record_trade_result(pnl):
    """
    Call after each trade:
    pnl > 0: profit
    pnl < 0: loss
    """
    now = datetime.now(TIMEZONE)

    if pnl < 0:
        loss = abs(pnl)
        daily_stats["losses"] += loss
        daily_stats["weekly_loss"] += loss
        daily_stats["monthly_loss"] += loss
    else:
        daily_stats["wins"] += pnl

    daily_stats["last_trade_time"] = now.isoformat()
    daily_stats["trades_today"] += 1
    save_stats()


# === LOGIC HELPERS ===

def reset_daily_stats_if_needed(now):
    last_time = daily_stats.get('last_trade_time')
    if last_time and (now - last_time) >= timedelta(days=1):
        logger.info("New trading day detected. Resetting daily stats.")
        daily_stats['trades_today'] = 0
        daily_stats['wins'] = 0
        daily_stats['losses'] = 0

def has_reached_max_trades():
    if daily_stats.get('trades_today', 0) >= MAX_TRADES_PER_DAY:
        logger.info(f"Trade rejected: Max trades ({MAX_TRADES_PER_DAY}) reached today.")
        return True
    return False

def is_session_allowed():
    if SESSION_FILTER and not is_in_kill_zone():
        logger.info("Trade rejected: Outside of kill zone hours.")
        return False
    return True


# === MAIN FUNCTION ===

def can_trade(account_balance):
    """
    Check if trading is allowed:
    - Resets daily stats if needed
    - Checks if within allowed kill zone sessions
    - Limits max trades/day
    - Applies drawdown protection
    """
    now = datetime.now(TIMEZONE)

    reset_daily_stats_if_needed(now)

    if has_reached_max_trades():
        return {
            "can_trade": False,
            "reason": "Max trades reached"
        }

    if not is_session_allowed():
        return {
            "can_trade": False,
            "reason": "Outside kill zone"
        }

    if not check_max_drawdown(account_balance):
        logger.info("Trade rejected: Max drawdown reached.")
        return {
            "can_trade": False,
            "reason": "Max drawdown"
        }

    return {
        "can_trade": True,
        "reason": "OK"
    }

def check_max_drawdown(account_balance, max_drawdown_pct=10):
    """Check if max drawdown exceeded (Compendium Section 11)."""
    current_drawdown = (daily_stats['losses'] - daily_stats['wins']) / account_balance * 100
    if current_drawdown > max_drawdown_pct:
        logger.error("Max drawdown exceeded. Stopping bot.")
        return False
    return True

# === DERIV INTERFACE ===
import json
import pytz
import logging
from datetime import datetime
import websockets
import asyncio
import requests  # Only for Telegram alerts

# External/global dependencies assumed:
# SYMBOL, DERIV_TOKEN, TRADE_EXECUTION
# redis_client, calculate_position_size
# TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)
MAX_RETRIES = 5


async def run_deriv_session():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with websockets.connect("wss://ws.deriv.com/websockets/v3") as ws:
                logger.info("WebSocket connected")

                authorized = await deriv_authorize(ws)
                if not authorized:
                    logger.error("Authorization failed.")
                    return

                await deriv_subscribe_ticks(ws)

                # Start trade monitoring in background
                asyncio.create_task(monitor_open_trades(ws))

                # === PLACE YOUR STRATEGY LOGIC HERE ===
                # await your_strategy_loop(ws)

                while True:
                    await asyncio.sleep(1)

        except Exception as e:
            retries += 1
            wait = 2 ** retries
            logger.warning(f"WebSocket error: {str(e)} — retrying in {wait}s ({retries}/{MAX_RETRIES})")
            await asyncio.sleep(wait)

    if retries == MAX_RETRIES:
        logger.critical("Max WebSocket retries reached. Exiting.")


async def deriv_authorize(ws):
    try:
        await ws.send(json.dumps({"authorize": DERIV_TOKEN}))
        response = await ws.recv()
        logger.info(f"Authorization response: {response}")
        return True
    except Exception as e:
        logger.error(f"Authorization failed: {str(e)}")
        return False


async def deriv_subscribe_ticks(ws):
    try:
        await ws.send(json.dumps({
            "ticks": SYMBOL,
            "subscribe": 1
        }))
        logger.info(f"Subscribed to {SYMBOL} ticks")
        return True
    except Exception as e:
        logger.error(f"Subscription failed: {str(e)}")
        return False


async def deriv_place_trade(ws, contract_type, entry_price, stop_loss, take_profit):
    if not TRADE_EXECUTION:
        logger.info(f"SIMULATION: {contract_type} at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
        return True

    try:
        position_size = calculate_position_size(entry_price, stop_loss)
        if position_size <= 0:
            logger.warning("Position size is zero or negative. Skipping trade.")
            return False

        # Step 1: Proposal
        proposal_req = {
            "proposal": 1,
            "amount": str(position_size),
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": 5,
            "duration_unit": "t",
            "symbol": SYMBOL
        }
        await ws.send(json.dumps(proposal_req))
        proposal_res = await ws.recv()
        proposal_data = json.loads(proposal_res)

        if 'error' in proposal_data:
            logger.error(f"Proposal error: {proposal_data['error']['message']}")
            return False

        proposal_id = proposal_data['proposal']['id']
        ask_price = proposal_data['proposal']['ask_price']

        # Step 2: Buy
        buy_req = {
            "buy": 1,
            "price": str(ask_price),
            "proposal_id": proposal_id
        }
        await ws.send(json.dumps(buy_req))
        buy_res = await ws.recv()
        buy_data = json.loads(buy_res)

        if 'error' in buy_data:
            logger.error(f"Buy error: {buy_data['error']['message']}")
            return False

        contract_id = buy_data.get('buy', {}).get('contract_id')
        trade = {
            'id': contract_id,
            'type': contract_type,
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'time': datetime.now(pytz.UTC).isoformat(),
            'status': 'open'
        }
        redis_client.rpush('trade_history', json.dumps(trade))
        send_telegram_message(f"✅ Trade opened: {contract_type}\nEntry: {entry_price}\nSL: {stop_loss}\nTP: {take_profit}")

        return True

    except Exception as e:
        logger.error(f"Trade execution failed: {str(e)}")
        return False


async def monitor_open_trades(ws):
    """Track trades and close them based on SL/TP or expiration."""
    logger.info("Lifecycle monitor started.")
    while True:
        try:
            trades = redis_client.lrange('trade_history', 0, -1)
            for raw in trades:
                trade = json.loads(raw)
                if trade.get('status') != 'open':
                    continue

                contract_id = trade['id']
                await ws.send(json.dumps({
                    "proposal_open_contract": 1,
                    "contract_id": contract_id
                }))
                response = json.loads(await ws.recv())

                if 'error' in response:
                    logger.error(f"Error tracking contract {contract_id}: {response['error']['message']}")
                    continue

                contract = response['proposal_open_contract']
                current_price = contract.get('bid_price', contract.get('sell_price'))

                # Check SL/TP
                if contract['contract_type'] in ['CALL', 'PUT']:
                    sl_hit = contract['contract_type'] == 'CALL' and current_price <= trade['stop_loss']
                    tp_hit = contract['contract_type'] == 'CALL' and current_price >= trade['take_profit']
                    sl_hit |= contract['contract_type'] == 'PUT' and current_price >= trade['stop_loss']
                    tp_hit |= contract['contract_type'] == 'PUT' and current_price <= trade['take_profit']

                    if sl_hit or tp_hit:
                        await ws.send(json.dumps({
                            "sell": contract_id,
                            "price": current_price
                        }))
                        result = json.loads(await ws.recv())

                        if 'error' in result:
                            logger.warning(f"Sell failed for {contract_id}: {result['error']['message']}")
                        else:
                            trade['status'] = 'closed'
                            trade['exit_price'] = current_price
                            trade['exit_time'] = datetime.now(pytz.UTC).isoformat()
                            trade['result'] = 'TP' if tp_hit else 'SL'
                            redis_client.rpush('trade_history', json.dumps(trade))
                            send_telegram_message(f"💹 Trade closed: {trade['result']} hit\nExit: {current_price}")

                elif contract.get('is_expired') or contract.get('is_sold'):
                    trade['status'] = 'closed'
                    trade['exit_price'] = contract.get('sell_price')
                    trade['exit_time'] = datetime.now(pytz.UTC).isoformat()
                    trade['result'] = 'expired'
                    redis_client.rpush('trade_history', json.dumps(trade))
                    send_telegram_message(f"⌛ Trade expired: Exit = {trade['exit_price']}")

            await asyncio.sleep(3)

        except Exception as e:
            logger.error(f"Lifecycle monitor error: {str(e)}")
            await asyncio.sleep(5)


def send_telegram_message(msg):
    """Send trade status to Telegram bot (optional)."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
            "parse_mode": "HTML"
        }
        requests.post(url, data=payload)
    except Exception as e:
        logger.warning(f"Telegram message failed: {str(e)}")

# === TRADING LOGIC ===
import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pytz
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======== Configuration ========
class Config:
    RISK_REWARD_RATIO = 1.5
    RISK_PER_TRADE = {
        'very_high': 0.02,  # 2% account risk
        'high': 0.01,
        'medium': 0.005,
        'low': 0.002
    }
    KILL_ZONES = {
        'london': (7, 12),
        'new_york': (13, 17),
        'asian': (0, 4)
    }
    PIP_VALUE = 0.0001
    MIN_CONFLUENCE_TIMEFRAMES = 2
    MAX_SLIPPAGE = 2  # In pips
    MIN_DATA_LENGTHS = {
        'm1': 50,
        'm5': 50,
        'm15': 50,
        'h4': 20,
        'd1': 5
    }

# ======== Data Structures ========
@dataclass
class Signal:
    type: str  # e.g. 'po3_buy', 'liquidity_sell'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: str  # 'very_high', 'high', 'medium', 'low'
    session: str
    risk_pct: float
    size: Optional[float] = None  # Position size in lots
    expiry: Optional[datetime] = None  # For time-based signals
    notes: Optional[str] = None

@dataclass
class MarketStructure:
    trend: str  # 'bullish', 'bearish', 'ranging'
    higher_highs: bool
    higher_lows: bool
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None

# ======== Core Utilities ========
def validate_dataframes(*dfs: pd.DataFrame) -> bool:
    """Ensure all DataFrames meet minimum length requirements."""
    for df, (name, min_len) in zip(dfs, Config.MIN_DATA_LENGTHS.items()):
        if len(df) < min_len:
            logger.warning(f"Insufficient data for {name} timeframe: {len(df)} < {min_len}")
            return False
    return True

def get_current_session() -> str:
    """Determine current trading session based on time."""
    current_hour = datetime.now(pytz.UTC).hour
    for session, (start, end) in Config.KILL_ZONES.items():
        if start <= current_hour < end:
            return session
    return 'none'

def calculate_position_size(entry: float, 
                          stop_loss: float, 
                          confidence: str, 
                          account_balance: float = 10000) -> Tuple[float, float]:
    """
    Calculate position size and targets based on volatility and account risk.
    Returns: (take_profit, size_in_lots)
    """
    risk_amount = account_balance * Config.RISK_PER_TRADE.get(confidence, 0.01)
    risk_distance = abs(entry - stop_loss)
    
    if risk_distance <= 0:
        logger.error("Zero/negative risk distance in position calculation")
        return entry, 0
    
    take_profit = entry + (risk_distance * Config.RISK_REWARD_RATIO) if entry > stop_loss \
                  else entry - (risk_distance * Config.RISK_REWARD_RATIO)
    
    # Standard forex lot size calculation (1 lot = 100,000 units)
    size = round((risk_amount / (risk_distance / Config.PIP_VALUE)) / 100000, 2)
    
    return take_profit, size

# ======== Market Analysis Components ========
def detect_market_structure(df: pd.DataFrame) -> MarketStructure:
    """Identify higher timeframe market structure."""
    highs = df['high'].rolling(5).max()
    lows = df['low'].rolling(5).min()
    
    higher_highs = (highs.diff() > 0).iloc[-3:].all()
    higher_lows = (lows.diff() > 0).iloc[-3:].all()
    
    if higher_highs and higher_lows:
        trend = 'bullish'
    elif not higher_highs and not higher_lows:
        trend = 'bearish'
    else:
        trend = 'ranging'
    
    return MarketStructure(
        trend=trend,
        higher_highs=higher_highs,
        higher_lows=higher_lows,
        swing_high=highs.iloc[-1],
        swing_low=lows.iloc[-1]
    )

def detect_po3(df: pd.DataFrame) -> pd.DataFrame:
    """Detect Power of Three price distribution patterns."""
    df = df.copy()
    conditions = {
        'bullish': (
            (df['close'].shift(3) < df['open'].shift(3)) &
            (df['close'].shift(2) < df['open'].shift(2)) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] > df['open'])
        ),
        'bearish': (
            (df['close'].shift(3) > df['open'].shift(3)) &
            (df['close'].shift(2) > df['open'].shift(2)) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] < df['open'])
        )
    }
    
    df['po3_distribution'] = np.select(
        [conditions['bullish'], conditions['bearish']],
        ['bullish', 'bearish'],
        default=None
    )
    return df

def calculate_ote(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Optimal Trade Entry zones."""
    df = df.copy()
    swing_high = df['high'].rolling(5).max().shift(1)
    swing_low = df['low'].rolling(5).min().shift(1)
    
    # Fibonacci-based OTE zones
    df['ote_buy_zone'] = swing_high - (0.618 * (swing_high - swing_low))
    df['ote_sell_zone'] = swing_low + (0.618 * (swing_high - swing_low))
    
    df['ote_type'] = np.where(
        df['close'] <= df['ote_buy_zone'],
        'buy',
        np.where(
            df['close'] >= df['ote_sell_zone'],
            'sell',
            None
        )
    )
    return df

# ======== Signal Generators ========
async def generate_po3_signals(df_m1: pd.DataFrame, 
                              df_m15: pd.DataFrame, 
                              htf_structure: MarketStructure,
                              account_balance: float) -> List[Signal]:
    """Generate PO3 signals with OTE confirmation."""
    signals = []
    current_session = get_current_session()
    
    # Bullish PO3 Signal
    if df_m15['po3_distribution'].iloc[-1] == 'bullish':
        if (df_m15['ote_type'].iloc[-1] == 'buy' and
            htf_structure.trend in ['bullish', 'ranging'] and
            df_m1['close'].iloc[-1] > df_m1['open'].iloc[-1]):  # Current M1 candle bullish
            
            entry = df_m15['ote_buy_zone'].iloc[-1]
            sl = min(df_m15['low'].iloc[-1] - (10 * Config.PIP_VALUE), 
                    df_m15['low'].iloc[-3:].min())  # Most recent swing low
            tp, size = calculate_position_size(entry, sl, 'high', account_balance)
            
            signals.append(Signal(
                type='po3_buy',
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence='high',
                session=current_session,
                risk_pct=Config.RISK_PER_TRADE['high'],
                size=size,
                notes=f"Bullish PO3 confirmed at OTE zone | HTF: {htf_structure.trend}"
            ))
    
    # Bearish PO3 Signal
    elif df_m15['po3_distribution'].iloc[-1] == 'bearish':
        if (df_m15['ote_type'].iloc[-1] == 'sell' and
            htf_structure.trend in ['bearish', 'ranging'] and
            df_m1['close'].iloc[-1] < df_m1['open'].iloc[-1]):  # Current M1 candle bearish
            
            entry = df_m15['ote_sell_zone'].iloc[-1]
            sl = max(df_m15['high'].iloc[-1] + (10 * Config.PIP_VALUE),
                    df_m15['high'].iloc[-3:].max())  # Most recent swing high
            tp, size = calculate_position_size(entry, sl, 'high', account_balance)
            
            signals.append(Signal(
                type='po3_sell',
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence='high',
                session=current_session,
                risk_pct=Config.RISK_PER_TRADE['high'],
                size=size,
                notes=f"Bearish PO3 confirmed at OTE zone | HTF: {htf_structure.trend}"
            ))
    
    return signals

async def generate_liquidity_signals(df_m1: pd.DataFrame,
                                   df_m15: pd.DataFrame,
                                   liquidity_zones: List[Dict],
                                   account_balance: float) -> List[Signal]:
    """Generate liquidity-based signals with sweep confirmation."""
    signals = []
    current_price = (df_m1['bid'].iloc[-1] + df_m1['ask'].iloc[-1]) / 2
    current_session = get_current_session()
    
    for zone in liquidity_zones[-3:]:  # Only consider 3 most recent zones
        if zone.get('tested', False):
            continue
            
        # ===== Sell-Side Liquidity =====
        if (zone['type'].startswith('sell') and 
            current_price > zone['price'] and
            df_m1['close'].iloc[-1] < df_m1['open'].iloc[-1] and  # Bearish M1 candle
            df_m15['close'].iloc[-1] < df_m15['open'].iloc[-1]):  # Bearish M15 candle
            
            # Confirm liquidity sweep
            recent_high = df_m1['high'].iloc[-5:-1].max()
            if df_m1['high'].iloc[-1] > recent_high:  # Sweep occurred
                sl = max(zone['price'] + (15 * Config.PIP_VALUE),
                        df_m1['high'].iloc[-1] + (5 * Config.PIP_VALUE))
                entry = current_price + (Config.MAX_SLIPPAGE * Config.PIP_VALUE)
                tp, size = calculate_position_size(entry, sl, 'very_high', account_balance)
                
                signals.append(Signal(
                    type='liquidity_sell',
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    confidence='very_high',
                    session=current_session,
                    risk_pct=Config.RISK_PER_TRADE['very_high'],
                    size=size,
                    notes=f"Untested {zone['type']} liquidity zone | Sweep: {recent_high:.5f}"
                ))
                zone['tested'] = True  # Mark as tested
        
        # ===== Buy-Side Liquidity =====
        elif (zone['type'].startswith('buy') and 
              current_price < zone['price'] and
              df_m1['close'].iloc[-1] > df_m1['open'].iloc[-1] and  # Bullish M1 candle
              df_m15['close'].iloc[-1] > df_m15['open'].iloc[-1]):  # Bullish M15 candle
              
              # Confirm liquidity sweep
              recent_low = df_m1['low'].iloc[-5:-1].min()
              if df_m1['low'].iloc[-1] < recent_low:  # Sweep occurred
                  sl = min(zone['price'] - (15 * Config.PIP_VALUE),
                          df_m1['low'].iloc[-1] - (5 * Config.PIP_VALUE))
                  entry = current_price - (Config.MAX_SLIPPAGE * Config.PIP_VALUE)
                  tp, size = calculate_position_size(entry, sl, 'very_high', account_balance)
                  
                  signals.append(Signal(
                      type='liquidity_buy',
                      entry_price=entry,
                      stop_loss=sl,
                      take_profit=tp,
                      confidence='very_high',
                      session=current_session,
                      risk_pct=Config.RISK_PER_TRADE['very_high'],
                      size=size,
                      notes=f"Untested {zone['type']} liquidity zone | Sweep: {recent_low:.5f}"
                  ))
                  zone['tested'] = True
    
    return signals

# ======== Main Analysis Engine ========
async def analyze_market(
    df_m1: pd.DataFrame,
    df_m5: pd.DataFrame,
    df_m15: pd.DataFrame,
    df_h4: pd.DataFrame,
    df_d1: pd.DataFrame,
    symbols_data: Dict,
    account_balance: float = 10000,
    *,
    min_confidence: str = "high",
    session_boost: bool = True,
    hft_alignment_check: bool = True,
    risk_per_trade: float = 0.01,
    timeframes_to_analyze: List[str] = ["m15", "h4", "d1"],
    signal_types: List[str] = ["po3", "liquidity", "order_block"],
    economic_calendar: Optional[EconomicCalendar] = None,
    correlated_assets: Optional[List[str]] = None
) -> List[Signal]:
    """
    Enhanced ICT Market Analysis Engine with:
    - Multi-timeframe confluence
    - Dynamic risk management
    - Institutional-grade signal generation
    - Machine learning enhancements
    - Market context awareness

    New Parameters:
        min_confidence: Minimum confidence level to return signals (low/medium/high/very_high)
        session_boost: Whether to increase confidence during active sessions
        hft_alignment_check: Verify alignment with higher timeframes
        risk_per_trade: Percentage of account to risk per trade (0.01 = 1%)
        timeframes_to_analyze: Which timeframes to consider for analysis
        signal_types: Which signal types to generate
        economic_calendar: Economic calendar integration
        correlated_assets: List of correlated assets for confirmation
    """
    
    # Data Validation
    if not validate_dataframes(df_m1, df_m5, df_m15, df_h4, df_d1):
        return []
    
    try:
        # Market Context Analysis
        market_context = determine_market_context(df_d1, df_h4, symbols_data.get('vix'))
        market_volatility = calculate_volatility(df_m15)
        
        # Parallel Market Analysis
        analysis_tasks = []
        if "d1" in timeframes_to_analyze:
            analysis_tasks.append(detect_market_structure(df_d1))
        if "h4" in timeframes_to_analyze:
            analysis_tasks.append(detect_market_structure(df_h4))
        if "m15" in timeframes_to_analyze:
            analysis_tasks.extend([
                detect_po3(df_m15),
                calculate_ote(df_m15),
                detect_liquidity_zones(df_m15),
                detect_order_blocks(df_m15),
                analyze_order_flow(df_m15),
                detect_volume_profile_nodes(df_m15)
            ])
        
        analysis_tasks.append(detect_smt(symbols_data))
        
        results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Unpack results with error handling
        result_index = 0
        d1_structure = MarketStructure('ranging', False, False)
        h4_structure = MarketStructure('ranging', False, False)
        
        if "d1" in timeframes_to_analyze:
            d1_structure = results[result_index] if not isinstance(results[result_index], Exception) else d1_structure
            result_index += 1
        if "h4" in timeframes_to_analyze:
            h4_structure = results[result_index] if not isinstance(results[result_index], Exception) else h4_structure
            result_index += 1
            
        m15_results = {}
        if "m15" in timeframes_to_analyze:
            m15_results = {
                'po3': results[result_index] if not isinstance(results[result_index], Exception) else df_m15.copy(),
                'ote': results[result_index+1] if not isinstance(results[result_index+1], Exception) else df_m15.copy(),
                'liquidity_zones': results[result_index+2] if not isinstance(results[result_index+2], Exception) else [],
                'order_blocks': results[result_index+3] if not isinstance(results[result_index+3], Exception) else [],
                'order_flow': results[result_index+4] if not isinstance(results[result_index+4], Exception) else None,
                'volume_profile': results[result_index+5] if not isinstance(results[result_index+5], Exception) else []
            }
            result_index += 6
            
        smt_signals = results[result_index] if not isinstance(results[result_index], Exception) else []

        # Generate Signals Concurrently
        signal_tasks = []
        if "po3" in signal_types:
            signal_tasks.append(
                generate_po3_signals(
                    df_m1, 
                    m15_results['po3'], 
                    h4_structure, 
                    account_balance,
                    market_context
                )
            )
        if "liquidity" in signal_types:
            signal_tasks.append(
                generate_liquidity_signals(
                    df_m1,
                    m15_results['ote'],
                    m15_results['liquidity_zones'],
                    account_balance,
                    m15_results['volume_profile']
                )
            )
        if "order_block" in signal_types:
            signal_tasks.append(
                generate_order_block_signals(
                    df_m1,
                    m15_results['order_blocks'],
                    h4_structure,
                    account_balance
                )
            )

        all_signals = await asyncio.gather(*signal_tasks)
        signals = [signal for sublist in all_signals for signal in sublist]

        # Apply Risk Management
        for signal in signals:
            # Position sizing
            signal.position_size = calculate_position_size(
                account_balance,
                risk_per_trade,
                signal.stop_loss_distance,
                symbols_data['symbol']
            )
            
            # Adjust for volatility
            if market_volatility > 0.5:
                signal.take_profit *= 1.5
                signal.stop_loss *= 1.2
                signal.position_size *= 0.8

            # Economic calendar impact
            if economic_calendar and economic_calendar.is_high_impact_event_soon():
                signal.position_size *= 0.5
                signal.add_note("High impact event pending")

        # Apply Advanced Filters
        final_signals = []
        for signal in signals:
            # Calculate composite score
            signal.score = calculate_signal_score(
                signal,
                d1_structure,
                h4_structure,
                m15_results['order_flow'],
                market_context
            )

            # Session-Specific Confidence Adjustment
            if session_boost and signal.session in ['london', 'new_york']:
                signal.score *= 1.2

            # HTF Alignment Check
            if hft_alignment_check:
                alignment_score = calculate_timeframe_alignment(
                    signal,
                    d1_structure,
                    h4_structure,
                    timeframe_weights={'d1': 0.4, 'h4': 0.3, 'm15': 0.2, 'm5': 0.1}
                )
                signal.score *= alignment_score

            # Correlation check
            if correlated_assets:
                correlation_strength = check_correlation(
                    symbols_data['symbol'],
                    correlated_assets
                )
                if correlation_strength > 0.7:
                    signal.score *= 1.1

            # ML confidence boost (if available)
            if hasattr(signal, 'ml_confidence'):
                signal.score *= signal.ml_confidence

            # Final threshold check
            confidence_thresholds = {
                'low': 0.5,
                'medium': 0.7,
                'high': 0.85,
                'very_high': 0.95
            }
            if signal.score >= confidence_thresholds[min_confidence]:
                final_signals.append(signal)

        logger.info(
            f"Market Analysis Complete\n"
            f"Context: {market_context}\n"
            f"Volatility: {market_volatility:.2f}\n"
            f"Initial Signals: {len(signals)}\n"
            f"Final Signals: {len(final_signals)}\n"
            f"Account Risk: {risk_per_trade*100}%"
        )

        return final_signals
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return []

# === REST API ===
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from pydantic import BaseModel, Field
import websockets
import logging
from contextlib import asynccontextmanager
from pydantic_settings import BaseSettings

# Configuration
class Settings(BaseSettings):
    deriv_app_id: str
    deriv_ws_url: str = "wss://ws.deriv.com/websockets/v3"
    api_key: str

settings = Settings()
logger = logging.getLogger(__name__)

# Models
class TradeSignal(BaseModel):
    type: str = Field(..., example="buy")
    symbol: str = Field(..., example="BTCUSD")
    price: float = Field(..., gt=0)
    stop_loss: float = Field(None, gt=0)
    take_profit: float = Field(None, gt=0)

class TradeResponse(BaseModel):
    success: bool
    trade_id: str

# Auth
async def verify_api_key(api_key: str = Header(..., alias="X-API-KEY")):
    if api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

# WebSocket
@asynccontextmanager
async def get_deriv_connection():
    try:
        async with websockets.connect(
            f"{settings.deriv_ws_url}?app_id={settings.deriv_app_id}"
        ) as ws:
            yield ws
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        raise HTTPException(status_code=502, detail="Deriv connection failed")

# FastAPI
app = FastAPI()

@app.post("/trade", dependencies=[Depends(verify_api_key)])
async def execute_trade(signal: TradeSignal, request: Request):
    logger.info(f"Trade request from {request.client.host}")
    
    try:
        async with get_deriv_connection() as ws:
            # Implement your actual trade logic here
            trade_result = await deriv_place_trade(
                ws, 
                signal.type,
                signal.price,
                signal.stop_loss,
                signal.take_profit
            )
            return TradeResponse(success=True, trade_id=trade_result['id'])
            
    except Exception as e:
        logger.error(f"Trade failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# === BACKTESTING ===
class ICTStrategy(bt.Strategy):
    def next(self):
        df_m1 = pd.DataFrame(self.datas[0].get(size=100))
        df_m5 = get_finnhub_data(SYMBOL, '5')
        df_m15 = get_finnhub_data(SYMBOL, '15')
        df_h4 = get_alpha_vantage_data(SYMBOL, '60min')
        df_d1 = get_alpha_vantage_data(SYMBOL, 'daily')
        symbols_data = {
            SYMBOL: df_m15,
            'frxGBPUSD': get_finnhub_data('frxGBPUSD', '15'),
            'frxUSDJPY': get_finnhub_data('frxUSDJPY', '15')
        }
        signals = asyncio.run(analyze_market(df_m1, df_m5, df_m15, df_h4, df_d1, symbols_data))
        if signals:
            for signal in signals:
                if signal['type'].startswith('buy'):
                    self.buy(size=calculate_position_size(signal['price'], signal['stop_loss']))
                else:
                    self.sell(size=calculate_position_size(signal['price'], signal['stop_loss']))

# === MAIN LOOP ===
import asyncio
import json
import time
import uuid
from collections import deque
from typing import Optional, Dict, Deque
from dataclasses import dataclass
import websockets
from websockets.client import WebSocketClientProtocol
import psutil
import structlog
import yaml
from functools import wraps
import signal
import os

# ======================
# 1. Configuration Setup
# ======================
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Constants from config
SYMBOL = config['symbol']
DERIV_APP_ID = config['deriv_app_id']
MAX_RETRIES = config['connection']['max_retries']
INITIAL_RETRY_DELAY = config['connection']['initial_delay']
CRITICAL_ALERTS_ENABLED = config['alerts']['enabled']
ADMIN_EMAIL = config['alerts']['email']

# =================
# 2. Logging Setup
# =================
logger = structlog.get_logger()

def alert_admin(message: str) -> None:
    if CRITICAL_ALERTS_ENABLED:
        logger.critical(message)
        # Implementation for send_email would go here

# ======================
# 3. Data Structures
# ======================
@dataclass
class TickData:
    time: float
    bid: float
    ask: float
    volume: float

class CircularBuffer(deque):
    def __init__(self, maxlen: int = 1000):
        super().__init__(maxlen=maxlen)
    
    def to_list(self) -> list:
        return list(self)

ticks_data: Deque[TickData] = CircularBuffer()

# ======================
# 4. Error Handling
# ======================
class TradingBotError(Exception):
    """Base class for all trading bot exceptions"""

class ConnectionError(TradingBotError):
    """Raised when connection fails after max retries"""

class TradeRejectedError(TradingBotError):
    """Raised when trade doesn't meet requirements"""

# ======================
# 5. Circuit Breaker
# ======================
class CircuitBreaker:
    def __init__(self, max_failures: int = 3, reset_timeout: int = 60):
        self.failures = 0
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
    
    async def call(self, func, *args, **kwargs):
        if self.failures >= self.max_failures:
            if time.time() - self.last_failure_time < self.reset_timeout:
                raise TradingBotError("Circuit tripped")
            self.failures = 0
        
        try:
            result = await func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            raise

# ======================
# 6. Connection Manager
# ======================
class ConnectionManager:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.circuit_breaker = CircuitBreaker()
    
    async def connect(self) -> Optional[WebSocketClientProtocol]:
        try:
            self.ws = await self.circuit_breaker.call(
                self._connect_with_retry,
                max_retries=MAX_RETRIES,
                initial_delay=INITIAL_RETRY_DELAY
            )
            self.connected = bool(self.ws)
            return self.ws
        except Exception as e:
            alert_admin(f"Critical connection failure: {str(e)}")
            return None
    
    async def _connect_with_retry(self, max_retries: int = 5, initial_delay: int = 5) -> WebSocketClientProtocol:
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                async with websockets.connect(
                    f"wss://ws.deriv.com/websockets/v3?app_id={DERIV_APP_ID}",
                    ping_interval=30,
                    ping_timeout=10
                ) as ws:
                    if await deriv_authorize(ws):
                        await deriv_subscribe_ticks(ws)
                        logger.info("Connection established")
                        return ws
            except Exception as e:
                logger.error(f"Connection failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
        
        raise ConnectionError("Failed to connect after max retries")

# ======================
# 7. Performance Tools
# ======================
def timed_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = await func(*args, **kwargs)
        duration = time.monotonic() - start
        logger.debug(f"{func.__name__} executed in {duration:.2f}s")
        return result
    return wrapper

async def monitor_resources():
    while True:
        await asyncio.sleep(300)
        mem = psutil.virtual_memory()
        logger.info(f"Memory usage: {mem.percent}%")

# ======================
# 8. Trade Execution
# ======================
class TradeQueue:
    def __init__(self, maxsize: int = 5):
        self.queue = asyncio.Queue(maxsize=maxsize)
    
    async def process_trades(self, ws: WebSocketClientProtocol):
        while True:
            signal = await self.queue.get()
            try:
                await self.execute_trade(ws, signal)
            except Exception as e:
                logger.error(f"Trade failed: {str(e)}")
            finally:
                self.queue.task_done()
    
    @timed_async
    async def execute_trade(self, ws: WebSocketClientProtocol, signal: Dict):
        if not validate_trade_signal(signal):
            raise TradeRejectedError("Invalid trade signal")
        
        current_price = get_current_price()
        if abs(signal['price'] - current_price) > signal.get('max_slippage', 0.0005):
            raise TradeRejectedError("Slippage too high")
        
        await deriv_place_trade(
            ws,
            signal['type'].split('_')[1].upper(),
            signal['price'],
            signal['stop_loss'],
            signal['take_profit']
        )

trade_queue = TradeQueue()

# ======================
# 9. Core Bot Logic
# ======================
async def monitor_trades(ws: WebSocketClientProtocol):
    for trade in trade_history:
        if trade['status'] == 'open':
            try:
                await ws.send(json.dumps({
                    "proposal_open_contract": 1,
                    "contract_id": trade['id']
                }))
                response = await ws.recv()
                res_data = json.loads(response)
                
                if 'error' not in res_data and res_data.get('proposal_open_contract', {}).get('is_sold'):
                    trade['status'] = 'closed'
                    profit = res_data['proposal_open_contract']['profit']
                    if profit > 0:
                        daily_stats['wins'] += 1
                    else:
                        daily_stats['losses'] += 1
                    logger.info(f"Trade {trade['id']} closed with profit: {profit}")
                    redis_client.set('trade_history', json.dumps(trade_history))
            except Exception as e:
                logger.error(f"Trade monitoring failed: {str(e)}")

async def main():
    # Setup graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda: asyncio.create_task(shutdown(sig, loop))
        )
    
    # Start monitoring tasks
    asyncio.create_task(monitor_resources())
    
    # Initialize connection
    conn_manager = ConnectionManager()
    
    while True:  # Outer loop for permanent operation
        try:
            ws = await conn_manager.connect()
            if not ws:
                await asyncio.sleep(60)
                continue
            
            # Start trade processor
            trade_processor = asyncio.create_task(trade_queue.process_trades(ws))
            
            # Start heartbeat
            heartbeat = asyncio.create_task(send_heartbeat(ws))
            
            # Main processing loop
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=60)
                    data = json.loads(message)
                    
                    if 'tick' in data and validate_tick(data['tick']):
                        tick = data['tick']
                        ticks_data.append(TickData(
                            time=tick['epoch'],
                            bid=tick['bid'],
                            ask=tick['ask'],
                            volume=tick.get('volume', 0)
                        ))
                        
                        redis_client.set('ticks_data', json.dumps(ticks_data.to_list()))
                        
                        # Convert and analyze data
                        df_m1 = convert_ticks_to_ohlc(ticks_data.to_list(), '1min')
                        signals = await analyze_market(df_m1)
                        
                        if signals and can_trade():
                            for signal in signals:
                                try:
                                    await trade_queue.queue.put(signal)
                                except asyncio.QueueFull:
                                    logger.warning("Trade queue full, skipping signal")
                    
                    await monitor_trades(ws)
                    
                except asyncio.TimeoutError:
                    logger.debug("Connection timeout, checking connection...")
                    try:
                        await ws.ping()
                    except:
                        logger.warning("Connection lost, reconnecting...")
                        break
                        
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    break
            
            # Cleanup before reconnection
            trade_processor.cancel()
            heartbeat.cancel()
            try:
                await ws.close()
            except:
                pass
            
        except Exception as e:
            logger.error(f"Main loop error: {str(e)}")
            await asyncio.sleep(60)

async def shutdown(sig, loop):
    """Handle graceful shutdown"""
    logger.info(f"Received {sig.name}, shutting down")
    
    # Cancel all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def run_bot():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        alert_admin(f"Bot crashed: {str(e)}")

if __name__ == "__main__":
    run_bot()