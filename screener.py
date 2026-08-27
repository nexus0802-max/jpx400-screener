"""
サテライト銘柄スクリーナー v2 - Phase 2
Phase 1（ファンダ複合スコア）に、RSランキング（対TOPIX超過リターン）を追加。

RS計算方法：
  各銘柄の「3ヶ月/6ヶ月/9ヶ月/12ヶ月リターン」を 0.4:0.2:0.2:0.2 で加重平均（IBD式）。
  同じ計算をTOPIXにも行い、銘柄の加重リターン - TOPIXの加重リターン = 超過リターン。
  超過リターンをJPX400内でパーセンタイル化したものを rs_percentile（0〜100）とする。

株価データはPhase1で取得済みの10年分の終値をそのまま使うので、追加のAPI呼び出しは
TOPIXベンチマーク1本のみ。
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
import math
import argparse
import sys
import time
import random
from datetime import datetime, timezone, timedelta
import os

# 2026-08-26以降、GitHub Actions実行環境からのYahoo Finance取得が
# ほぼ全銘柄で失敗し（yfinanceのバージョンアップに伴うcookie/crumb認証の
# 強制化＋クラウド/CI系IPに対するYahoo側のレート制限強化が原因と推定）、
# 取得件数0件でmain()内のpandas処理がKeyErrorで異常終了 → docs/data.jsonが
# 再生成されないままGitHub Pagesが直前の成功データを配信し続ける、という
# 障害が発生した。以下の対策を追加する：
#   1) yf.download / Ticker.infoにリトライ＋指数バックオフを追加
#   2) 銘柄間に待機を挟みレート制限に当たりにくくする
#   3) 取得成功率が閾値を下回ったら docs/data.json を書き換えずに
#      明示的に異常終了させる（CIが赤くなり気付ける状態にする。
#      サイレントに古いデータのまま固まるのを防ぐ）
MIN_FETCH_SUCCESS_RATE = 0.5  # これを下回ったら異常終了して前回データを保持する
YF_MAX_RETRIES = 4
YF_RETRY_BASE_DELAY_SEC = 5
INTER_TICKER_SLEEP_SEC = 1.2  # 連続リクエストによるレート制限回避用


def _with_retry(fn, *, what, max_retries=YF_MAX_RETRIES, base_delay=YF_RETRY_BASE_DELAY_SEC):
    """yfinance呼び出し用の指数バックオフ＋ジッター付きリトライ。
    YFRateLimitError等の具体的な例外クラスはyfinanceのバージョンによって
    名前・場所が変わるため、あえてException全般を対象にする。"""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay)
            print(f"  [retry] {what}: {type(e).__name__}: {e} "
                  f"-> {attempt}/{max_retries}回目失敗、{delay:.1f}秒待って再試行")
            time.sleep(delay)
    print(f"  [failed] {what}: {max_retries}回リトライしても失敗: {last_err}")
    return None

# TOPIX連動ETF（"^TOPX"の指数シンボルはyfinanceでの取得安定性に懸念があるため、
# 流動性が高く通常の株価と同じ経路で確実に取得できるETFをベンチマークの代理として使う）
BENCHMARK_TICKER = "1306.T"

# EMA9/21 BUY SIGNAL（detect_daily_buy_signal）で使うEMA期間のデフォルト値。
# --daily-ema-fast / --daily-ema-slow のコマンドライン引数で変更できる。
DEFAULT_DAILY_EMA_FAST = 20
DEFAULT_DAILY_EMA_SLOW = 30

JPX400_TICKERS = [
    "1332","1414","1518","1605","1662","1719","1721","1801","1802","1803",
    "1808","1812","1878","1885","1893","1911","1925","1928","1942","1944",
    "1951","1959","1961","1969","1980","2124","2127","2154","2181","2201",
    "2212","2222","2229","2264","2267","2269","2317","2327","2331","2371",
    "2379","2384","2413","2503","2587","2670","2678","2685","2702","2726",
    "2767","2768","2801","2802","2811","2871","2875","2897","2914","3003",
    "3038","3064","3086","3088","3092","3099","3107","3110","3116","3132",
    "3148","3186","3231","3288","3289","3291","3349","3382","3391","3402",
    "3465","3549","3563","3626","3635","3659","3697","3769","3774","3923",
    "4004","4021","4042","4062","4063","4091","4151","417A","4183","4186",
    "4194","4202","4203","4204","4206","4307","4373","4385","4401","4403",
    "4452","4503","4507","4516","4519","4523","4527","4528","4536","4543",
    "4568","4578","4612","4613","4626","4661","4666","4680","4681","4684",
    "4686","4689","4704","4716","4722","4732","4733","4751","4768","4812",
    "4816","4901","4966","4980","5019","5020","5021","5032","5076","5101",
    "5105","5108","5332","5333","5334","5344","5393","5401","5406","5411",
    "5471","547A","5706","5713","5714","5741","5801","5802","5803","5805",
    "5838","5857","5929","5947","5991","6005","6098","6101","6113","6141",
    "6146","6254","6269","6273","6301","6305","6323","6326","6361","6367",
    "6368","6383","6417","6432","6436","6448","6457","6460","6465","6479",
    "6501","6503","6504","6506","6507","6508","6526","6532","6544","6586",
    "6590","6632","6674","6701","6702","6703","6723","6724","6728","6752",
    "6758","6762","6787","6806","6841","6845","6856","6857","6861","6869",
    "6871","6902","6920","6951","6954","6960","6965","6981","6988","7003",
    "7011","7012","7013","7014","7148","7164","7167","7173","7186","7202",
    "7203","7211","7259","7261","7267","7269","7270","7272","7282","7309",
    "7419","7453","7532","7550","7564","7599","7649","7701","7729","7733",
    "7735","7740","7741","7747","7751","7762","7832","7867","7906","7912",
    "7936","7974","7988","7994","8001","8002","8015","8020","8031","8035",
    "8050","8053","8056","8058","8060","8078","8088","8098","8111","8113",
    "8130","8133","8136","8154","8174","8194","8227","8242","8252","8253",
    "8306","8308","8309","8316","8331","8334","8354","8411","8424","8425",
    "8439","8473","8572","8591","8593","8601","8604","8628","8630","8697",
    "8725","8750","8766","8795","8801","8802","8804","8830","8848","8850",
    "8876","8919","8923","8934","9001","9003","9005","9006","9007","9008",
    "9009","9021","9022","9024","9031","9041","9045","9065","9069","9101",
    "9104","9107","9110","9142","9143","9201","9202","9336","9418","9432",
    "9434","9435","9449","9502","9503","9504","9505","9506","9507","9508",
    "9509","9513","9531","9532","9602","9616","9682","9684","9697","9706",
    "9735","9744","9759","9766","9843","9934","9962","9983","9984","9989",
]

# ---------- 既存ロジック（テクニカル）：そのまま流用 ----------

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0).astype(float)
    losses = np.where(deltas < 0, -deltas, 0).astype(float)
    ag = np.mean(gains[:period])
    al = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        ag = (ag * (period-1) + gains[i]) / period
        al = (al * (period-1) + losses[i]) / period
    return 100 if al == 0 else float(100 - 100 / (1 + ag/al))


def calc_adx_ci(highs, lows, closes, period=14, lookback=60):
    n = len(closes)
    if n < period * 2 + 1:
        return None, None
    tr  = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        h, l, pc = highs[i], lows[i], closes[i-1]
        ph, pl = highs[i-1], lows[i-1]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        up, down = h - ph, pl - l
        pdm[i] = up if (up > down and up > 0) else 0
        mdm[i] = down if (down > up and down > 0) else 0
    sm_tr = np.full(n, np.nan); sm_pdm = np.full(n, np.nan); sm_mdm = np.full(n, np.nan)
    sm_tr[period] = tr[1:period+1].sum(); sm_pdm[period] = pdm[1:period+1].sum(); sm_mdm[period] = mdm[1:period+1].sum()
    for i in range(period + 1, n):
        sm_tr[i]  = sm_tr[i-1]  - sm_tr[i-1]/period  + tr[i]
        sm_pdm[i] = sm_pdm[i-1] - sm_pdm[i-1]/period + pdm[i]
        sm_mdm[i] = sm_mdm[i-1] - sm_mdm[i-1]/period + mdm[i]
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if sm_tr[i] == 0:
            pdi = mdi = 0.0
        else:
            pdi = 100 * sm_pdm[i] / sm_tr[i]; mdi = 100 * sm_mdm[i] / sm_tr[i]
        s = pdi + mdi
        dx[i] = 0 if s == 0 else 100 * abs(pdi - mdi) / s
    first_adx_idx = period * 2
    if first_adx_idx >= n:
        return None, None
    adx = np.full(n, np.nan)
    adx[first_adx_idx - 1] = np.mean(dx[period:first_adx_idx])
    for i in range(first_adx_idx, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    ci = np.full(n, np.nan)
    for i in range(period, n):
        window_tr = tr[i-period+1:i+1].sum()
        hh = highs[i-period+1:i+1].max(); ll = lows[i-period+1:i+1].min()
        rng = hh - ll
        ci[i] = 100 * np.log10(window_tr / rng) / np.log10(period) if rng > 0 else 0
    start = max(0, n - lookback)
    adx_vals = adx[start:][~np.isnan(adx[start:])]
    ci_vals  = ci[start:][~np.isnan(ci[start:])]
    if len(adx_vals) == 0 or len(ci_vals) == 0:
        return None, None
    return float(np.mean(adx_vals)), float(np.mean(ci_vals))


def classify_trend(adx, ci):
    if adx is None or ci is None:
        return None
    if adx >= 25 and ci <= 50:
        return "trend"
    if adx < 20 and ci >= 61.8:
        return "range"
    return "mid"


def backtest_sma5_20(opens, highs, lows, closes, sma_fast_len=5, sma_slow_len=20,
                      body_pct=50, max_bars_after_cross=10, stop_loss_pct=8,
                      min_body_to_range_pct=60):
    n = len(closes)
    if n < sma_slow_len + max_bars_after_cross + 5:
        return None
    sma5  = pd.Series(closes).rolling(sma_fast_len).mean().values
    sma20 = pd.Series(closes).rolling(sma_slow_len).mean().values
    waiting = False; cross_i = -1; position = False
    entry_price = 0.0; entry_index = -1; stop_price = 0.0
    trades = []
    start = sma_slow_len
    for i in range(start, n):
        if np.isnan(sma5[i]) or np.isnan(sma20[i]) or np.isnan(sma5[i-1]) or np.isnan(sma20[i-1]):
            continue
        golden = sma5[i-1] <= sma20[i-1] and sma5[i] > sma20[i]
        dead   = sma5[i-1] >= sma20[i-1] and sma5[i] < sma20[i]
        if golden: waiting = True; cross_i = i
        if dead: waiting = False
        if position:
            if lows[i] <= stop_price:
                exit_price = min(stop_price, opens[i])  # ギャップダウン対応済み
                bars_held = i - entry_index
                return_pct = (exit_price / entry_price - 1) * 100
                trades.append((bars_held, return_pct, return_pct >= 0))
                position = False
                continue
            bearish = closes[i] < opens[i]
            if closes[i] < sma5[i] and bearish:
                exit_price = closes[i]
                bars_held = i - entry_index
                return_pct = (exit_price / entry_price - 1) * 100
                trades.append((bars_held, return_pct, return_pct >= 0))
                position = False
                continue
        else:
            if waiting and (i - cross_i) <= max_bars_after_cross:
                bullish = closes[i] > opens[i]
                body_high = max(opens[i], closes[i]); body_low = min(opens[i], closes[i])
                body_size = body_high - body_low
                if body_size > 0:
                    if sma5[i] <= body_low: above_amt = body_size
                    elif sma5[i] >= body_high: above_amt = 0.0
                    else: above_amt = body_high - sma5[i]
                    above_pct = above_amt / body_size * 100
                else:
                    above_pct = 0.0
                total_range = highs[i] - lows[i]
                body_to_range_pct = (body_size / total_range * 100) if total_range > 0 else 0.0
                if (bullish and above_pct >= body_pct
                        and body_to_range_pct >= min_body_to_range_pct and i + 1 < n):
                    entry_price = opens[i + 1]; entry_index = i + 1
                    stop_price  = entry_price * (1 - stop_loss_pct / 100)
                    position = True; waiting = False
    if len(trades) == 0:
        return None
    wins_sum = sum(r for _, r, w in trades if w)
    losses_sum = sum(-r for _, r, w in trades if not w)
    trade_count = len(trades)
    win_trades = [t for t in trades if t[2]]
    loss_trades = [t for t in trades if not t[2]]
    pf = None if losses_sum == 0 and wins_sum == 0 else (999.0 if losses_sum == 0 else wins_sum / losses_sum)

    def _avg(vals):
        return float(np.mean(vals)) if len(vals) > 0 else None

    return {
        "pf": pf, "trade_count": trade_count,
        "win_rate": round(len(win_trades) / trade_count * 100, 1),
        "avg_bars_held": _avg([t[0] for t in trades]),
        "avg_return_pct": _avg([t[1] for t in trades]),
        "avg_bars_win": _avg([t[0] for t in win_trades]),
        "avg_bars_loss": _avg([t[0] for t in loss_trades]),
        "avg_return_win": _avg([t[1] for t in win_trades]),
        "avg_return_loss": _avg([t[1] for t in loss_trades]),
    }


# ---------- 新規：VCP収縮パターン検出（旧vcp.htmlのJSロジックをPython移植） ----------

def sma(arr, n):
    if len(arr) < n:
        return None
    return float(np.mean(arr[-n:]))


def sma_at(arr, n, end_idx_exclusive):
    start = end_idx_exclusive - n
    if start < 0:
        return None
    return float(np.mean(arr[start:end_idx_exclusive]))


def find_swings(prices, order):
    """左右order日間で最も高い/低い点をスイング高値・安値として検出し、
    同種の極値が連続する場合はより極端な方だけを残す（元JSロジックと同一）。"""
    n = len(prices)
    raw = []
    for i in range(order, n - order):
        is_high = True
        is_low = True
        for k in range(i - order, i + order + 1):
            if k == i:
                continue
            if prices[k] > prices[i]:
                is_high = False
            if prices[k] < prices[i]:
                is_low = False
        if is_high:
            raw.append({"idx": i, "type": "high", "price": prices[i]})
        elif is_low:
            raw.append({"idx": i, "type": "low", "price": prices[i]})

    cleaned = []
    for p in raw:
        if not cleaned:
            cleaned.append(p)
            continue
        last = cleaned[-1]
        if last["type"] == p["type"]:
            if p["type"] == "high" and p["price"] >= last["price"]:
                cleaned[-1] = p
            if p["type"] == "low" and p["price"] <= last["price"]:
                cleaned[-1] = p
        else:
            cleaned.append(p)
    return cleaned


def extract_contractions(swings):
    """高値→安値のペアごとに収縮幅(%)を算出。"""
    contractions = []
    for i in range(len(swings) - 1):
        if swings[i]["type"] == "high" and swings[i + 1]["type"] == "low":
            high, low = swings[i], swings[i + 1]
            pct = (high["price"] - low["price"]) / high["price"] * 100
            contractions.append({
                "highIdx": high["idx"], "lowIdx": low["idx"],
                "highPrice": high["price"], "lowPrice": low["price"], "pct": pct,
            })
    return contractions


def analyze_vcp(closes, volumes,
                 above_52w_low=25, near_52w_high=25, ma200_slope_days=20,
                 lookback_days=130, swing_order=4,
                 min_contractions=2, max_contractions=5,
                 final_pct=12, tolerance=0.20, pivot_pct=7):
    """VCP（ボラティリティ収縮パターン）分析。旧vcp.htmlのデフォルトパラメータをそのまま踏襲。
    filterで銘柄を弾かず、常にvcp_scoreを返す（統合スコア側で連続値として使うため）。"""
    N = len(closes)
    if N < 260:
        return None
    closes = np.asarray(closes, dtype=float)

    latest = float(closes[-1])
    ma50, ma150, ma200 = sma(closes, 50), sma(closes, 150), sma(closes, 200)
    if ma50 is None or ma150 is None or ma200 is None:
        return None

    is_trend = latest > ma150 and ma150 > ma200
    is_ma50_above = ma50 > ma150
    ma200_past = sma_at(closes, 200, N - ma200_slope_days)
    is_ma200_up = (ma200 > ma200_past) if ma200_past is not None else False

    win52 = closes[-252:]
    low52, high52 = float(np.min(win52)), float(np.max(win52))
    above_low_pct = (latest - low52) / low52 * 100
    below_high_pct = (high52 - latest) / high52 * 100

    stage2_pass = (is_trend and is_ma50_above and is_ma200_up
                   and above_low_pct >= above_52w_low and below_high_pct <= near_52w_high)

    start = max(0, N - lookback_days)
    window = closes[start:]
    swings = find_swings(window.tolist(), swing_order)
    contractions = extract_contractions(swings)
    if len(contractions) > max_contractions:
        contractions = contractions[-max_contractions:]
    has_enough_contractions = len(contractions) >= min_contractions

    is_monotonic = True
    for i in range(1, len(contractions)):
        if contractions[i]["pct"] > contractions[i - 1]["pct"] * (1 + tolerance):
            is_monotonic = False
            break

    final_contraction = contractions[-1] if contractions else None
    is_final_tight = (final_contraction["pct"] <= final_pct) if final_contraction else None

    is_vol_dry_up = None
    if final_contraction and volumes is not None and len(volumes) >= N:
        volumes = np.asarray(volumes, dtype=float)
        first = contractions[0]

        def vol_slice(c):
            a, b = start + c["highIdx"], start + c["lowIdx"]
            lo, hi = min(a, b), max(a, b) + 1
            seg = volumes[lo:hi]
            return float(np.mean(seg)) if len(seg) else None

        vol_first, vol_final = vol_slice(first), vol_slice(final_contraction)
        if vol_first and vol_final:
            is_vol_dry_up = vol_final < vol_first

    dist_to_pivot_pct = None
    is_near_pivot = None
    if final_contraction:
        pivot = final_contraction["highPrice"]
        dist_to_pivot_pct = (pivot - latest) / pivot * 100
        is_near_pivot = abs(dist_to_pivot_pct) <= pivot_pct

    score = 0
    if is_trend: score += 20
    if is_ma50_above: score += 10
    if is_ma200_up: score += 10
    score += min(15, len(contractions) * 5)
    if contractions:
        if is_monotonic:
            score += 20
        else:
            score -= 15  # 収縮が乱れている場合は明確に減点する（単に加点なしにするだけでは不十分なため）
    if is_final_tight: score += 15
    if final_contraction and final_contraction["pct"] <= final_pct / 2: score += 5
    if is_vol_dry_up: score += 10
    if is_near_pivot: score += 15

    # 満点120点を0-100スケールに正規化（統合スコアでファンダ/RS/地合いスコアと同じスケールで扱うため）
    SCORE_MAX = 120
    score_normalized = max(0, min(100, round(score / SCORE_MAX * 100, 1)))

    return {
        "vcp_stage2_pass": bool(stage2_pass),
        "vcp_above_52w_low_pct": round(above_low_pct, 2),
        "vcp_below_52w_high_pct": round(below_high_pct, 2),
        "vcp_contraction_count": len(contractions),
        "vcp_contractions_pct": [round(c["pct"], 2) for c in contractions],
        "vcp_has_enough_contractions": bool(has_enough_contractions),
        "vcp_is_monotonic": is_monotonic if contractions else None,
        "vcp_is_final_tight": is_final_tight,
        "vcp_is_vol_dry_up": is_vol_dry_up,
        "vcp_is_near_pivot": is_near_pivot,
        "vcp_dist_to_pivot_pct": round(dist_to_pivot_pct, 2) if dist_to_pivot_pct is not None else None,
        "vcp_score_raw": score,
        "vcp_score": score_normalized,
    }




FUND_KEYS = ["trailing_pe", "price_to_book", "roe", "profit_margin",
             "revenue_growth", "earnings_growth"]

def fetch_fundamentals(ticker):
    t = f"{ticker}.T"
    info = _with_retry(lambda: yf.Ticker(t).info, what=f"ファンダ取得 {ticker}")
    if not info:
        return {"company_name": None, **{k: None for k in FUND_KEYS}}
    pe = info.get("trailingPE")
    if pe is not None and pe <= 0:
        pe = None
    return {
        "company_name": info.get("shortName") or info.get("longName"),
        "trailing_pe": pe,
        "price_to_book": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
    }


def compute_fund_scores(records, weights=(0.30, 0.40, 0.30)):
    df = pd.DataFrame(records)

    def pct_rank(col, higher_is_better):
        return df[col].rank(pct=True, ascending=higher_is_better)

    per_score = pct_rank("trailing_pe", higher_is_better=False).fillna(0.5)
    pbr_score = pct_rank("price_to_book", higher_is_better=False).fillna(0.5)
    value_score = (per_score + pbr_score) / 2

    roe_score = pct_rank("roe", higher_is_better=True).fillna(0.5)
    margin_score = pct_rank("profit_margin", higher_is_better=True).fillna(0.5)
    quality_score = (roe_score + margin_score) / 2

    rev_score = pct_rank("revenue_growth", higher_is_better=True).fillna(0.5)
    earn_score = pct_rank("earnings_growth", higher_is_better=True).fillna(0.5)
    growth_score = (rev_score + earn_score) / 2

    w_value, w_quality, w_growth = weights
    fund_score = (value_score * w_value + quality_score * w_quality + growth_score * w_growth) * 100

    df["fund_value_score"] = (value_score * 100).round(1)
    df["fund_quality_score"] = (quality_score * 100).round(1)
    df["fund_growth_score"] = (growth_score * 100).round(1)
    df["fund_score"] = fund_score.round(1)
    df["fund_percentile"] = (fund_score.rank(pct=True) * 100).round(1)
    return df.to_dict(orient="records")


# ---------- 新規：RSランキング（Phase2） ----------

def calc_weighted_return(closes, w=(0.4, 0.2, 0.2, 0.2)):
    """IBD式：3ヶ月(63営業日)/6ヶ月(126)/9ヶ月(189)/12ヶ月(252)リターンの加重平均。
    データが足りない場合はNoneを返す。"""
    n = len(closes)

    def ret(days):
        if n <= days:
            return None
        base = closes[-1 - days]
        if base == 0:
            return None
        return closes[-1] / base - 1

    r3, r6, r9, r12 = ret(63), ret(126), ret(189), ret(252)
    if any(v is None for v in (r3, r6, r9, r12)):
        return None
    return w[0]*r3 + w[1]*r6 + w[2]*r9 + w[3]*r12


def fetch_benchmark_data():
    """TOPIX連動ETFのOHLCVを取得。加重リターン計算だけでなく地合い判定にも使うため、
    Closeだけでなく高値・安値も含めて2年分取得する。"""
    df = _with_retry(
        lambda: yf.download(BENCHMARK_TICKER, period="2y", progress=False, auto_adjust=True),
        what=f"ベンチマーク({BENCHMARK_TICKER})取得",
    )
    if df is None or len(df) < 260:
        print(f"  ベンチマーク({BENCHMARK_TICKER})のデータが不足しています")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    if len(df) < 260:
        return None
    return df


def regime_score(trend_class, above_ma200):
    """トレンド判定とMA200との位置関係を組み合わせて地合いスコア(0-100)を出す。
    ADXが高い＝トレンドが強い、というだけでは方向（上昇か下降か）が分からないため、
    MA200より上にいるかどうかを必ず組み合わせる。"""
    if trend_class is None or above_ma200 is None:
        return 50  # データ不足時は中立扱い
    table = {
        ("trend", True): 100,
        ("mid",   True): 70,
        ("range", True): 55,
        ("range", False): 45,
        ("mid",   False): 40,
        ("trend", False): 20,
    }
    return table.get((trend_class, above_ma200), 50)


def compute_market_regime(bench_df):
    """ベンチマーク自身にADX/CIトレンド判定を適用し、地合いスコアを算出する。"""
    if bench_df is None:
        return {
            "trend_class": None, "adx": None, "ci": None,
            "above_ma200": None, "ma200": None, "latest_close": None,
            "regime_score": 50, "note": "ベンチマーク取得失敗のため中立(50)にフォールバック",
        }
    highs = bench_df["High"].astype(float).values
    lows = bench_df["Low"].astype(float).values
    closes = bench_df["Close"].astype(float).values
    N = len(closes)
    adx, ci = calc_adx_ci(highs, lows, closes)
    trend_class = classify_trend(adx, ci)
    ma200 = float(np.mean(closes[-200:])) if N >= 200 else float(np.mean(closes))
    latest_close = float(closes[-1])
    above_ma200 = bool(latest_close > ma200)
    score = regime_score(trend_class, above_ma200)
    return {
        "trend_class": trend_class,
        "adx": round(adx, 1) if adx is not None else None,
        "ci": round(ci, 1) if ci is not None else None,
        "above_ma200": above_ma200,
        "ma200": round(ma200, 2),
        "latest_close": round(latest_close, 2),
        "regime_score": score,
    }


def compute_rs_scores(records, bench_return):
    df = pd.DataFrame(records)
    if bench_return is not None:
        df["excess_return"] = df["weighted_return"] - bench_return
        rs_basis = "excess_vs_topix"
    else:
        df["excess_return"] = df["weighted_return"]
        rs_basis = "raw_return_fallback"
    df["rs_percentile"] = (df["excess_return"].rank(pct=True) * 100).fillna(50.0).round(1)
    df["excess_return"] = df["excess_return"].round(4)
    df["weighted_return"] = df["weighted_return"].round(4) if "weighted_return" in df else None
    df.attrs["rs_basis"] = rs_basis
    return df.to_dict(orient="records"), rs_basis


# ---------- 価格取得＋テクニカル＋ファンダ統合 ----------

# ---------- 新規：EMA9/21 BUY SIGNALS ----------

def calc_ema(series, span):
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


def detect_daily_buy_signal(opens, highs, lows, closes, ema_fast=9, ema_slow=21):
    """日足BUY（Entry C、EMA戦略v2 Pineスクリプトのrawブザインアルと完全一致させる版）：
    1) 前日終値が前日EMA(fast)より上（＝押し目がまだ入っていない状態）
    2) 当日の安値がEMA(fast)以下、かつ当日の高値がEMA(slow)以上
       （当日の値幅がfast/slowのゾーンにきちんと交差＝押し目形成）
    3) 当日終値がEMA(slow)より上（ゾーンから上に戻ってきている）
    4) 当日が陽線（close > open）かつ終値が前日高値を上抜け（反発確認）
    """
    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if n < max(ema_fast, ema_slow) + 5:
        return False
    ema_f = calc_ema(closes, ema_fast)
    ema_s = calc_ema(closes, ema_slow)
    i = n - 1
    pullback_into_zone = (
        closes[i - 1] > ema_f[i - 1]
        and lows[i] <= ema_f[i]
        and highs[i] >= ema_s[i]
        and closes[i] > ema_s[i]
    )
    bullish_reversal = closes[i] > opens[i] and closes[i] > highs[i - 1]
    return bool(pullback_into_zone and bullish_reversal)


def detect_weekly_status(date_index, closes):
    """週足OK/NG：日足終値を週足(金曜終値)にリサンプルし、週足EMA9がEMA21より上ならOK。
    売買判定には使わない、参考表示専用。"""
    try:
        s = pd.Series(np.asarray(closes, dtype=float), index=pd.DatetimeIndex(date_index))
        weekly = s.resample("W-FRI").last().dropna()
        if len(weekly) < 22:
            return None
        w9 = calc_ema(weekly.values, 9)
        w21 = calc_ema(weekly.values, 21)
        return "OK" if w9[-1] > w21[-1] else "NG"
    except Exception:
        return None


def calc_momentum_63d(closes):
    closes = np.asarray(closes, dtype=float)
    if len(closes) < 64:
        return None
    base = closes[-64]
    if base == 0:
        return None
    return float(closes[-1] / base - 1) * 100


def fetch_and_analyze(ticker, ema_fast=DEFAULT_DAILY_EMA_FAST, ema_slow=DEFAULT_DAILY_EMA_SLOW):
    try:
        t = f"{ticker}.T"
        df = _with_retry(
            lambda: yf.download(t, period="10y", progress=False, auto_adjust=True),
            what=f"株価取得 {ticker}",
        )
        if df is None or len(df) < 100:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) < 100:
            return None

        opens = df["Open"].astype(float).values
        highs = df["High"].astype(float).values
        lows = df["Low"].astype(float).values
        closes = df["Close"].astype(float).values
        volumes = df["Volume"].astype(float).values
        N = len(closes)

        rsi = calc_rsi(closes)
        ma200 = float(np.mean(closes[-200:])) if N >= 200 else float(np.mean(closes))
        latest_close = float(closes[-1])
        adx, ci = calc_adx_ci(highs, lows, closes)
        trend_class = classify_trend(adx, ci)
        bt = backtest_sma5_20(opens, highs, lows, closes)
        fund = fetch_fundamentals(ticker)
        weighted_return = calc_weighted_return(closes)
        vcp = analyze_vcp(closes, volumes)
        gap = analyze_gaps(opens, closes)

        # EMA BUY SIGNALS用（ema_fast/ema_slowはfetch_and_analyzeの引数で変更可能）
        daily_buy_signal = detect_daily_buy_signal(opens, highs, lows, closes, ema_fast=ema_fast, ema_slow=ema_slow)
        weekly_status = detect_weekly_status(df.index, closes)
        momentum_63d = calc_momentum_63d(closes)
        signal_date = df.index[-1].strftime("%Y-%m-%d")

        # ==== trend.html用：ウォークフォワード検証（前半IS→後半OOS） ====
        split = N // 2
        bt_is = bt_oos = None
        trend_class_is = None
        if split >= 60 and (N - split) >= 60:
            adx_is, ci_is = calc_adx_ci(highs[:split], lows[:split], closes[:split])
            trend_class_is = classify_trend(adx_is, ci_is)
            bt_is = backtest_sma5_20(opens[:split], highs[:split], lows[:split], closes[:split])
            bt_oos = backtest_sma5_20(opens[split:], highs[split:], lows[split:], closes[split:])

        # 過熱感フィルター用：MA200からの乖離率、52週高値からの乖離率
        extension_from_ma200_pct = round((latest_close / ma200 - 1) * 100, 2) if ma200 else None
        lookback_52w = closes[-252:] if N >= 252 else closes
        high_52w = float(np.max(lookback_52w))
        dist_from_52w_high_pct = round((latest_close / high_52w - 1) * 100, 2)

        def r2(v):
            return round(v, 3) if v is not None else None

        rec = {
            "ticker": ticker,
            "rsi": round(rsi, 1),
            "ma200": round(ma200, 2),
            "latest_close": round(latest_close, 2),
            "above_ma200": bool(latest_close > ma200),
            "adx": round(adx, 1) if adx is not None else None,
            "ci": round(ci, 1) if ci is not None else None,
            "trend_class": trend_class,
            "pf": r2(bt["pf"]) if bt else None,
            "trade_count": bt["trade_count"] if bt else None,
            "win_rate": bt["win_rate"] if bt else None,
            "avg_bars_held": r2(bt["avg_bars_held"]) if bt else None,
            "avg_return_pct": r2(bt["avg_return_pct"]) if bt else None,
            "trend_class_is": trend_class_is,
            "pf_is": r2(bt_is["pf"]) if bt_is else None,
            "trades_is": bt_is["trade_count"] if bt_is else None,
            "avg_return_pct_is": r2(bt_is["avg_return_pct"]) if bt_is else None,
            "pf_oos": r2(bt_oos["pf"]) if bt_oos else None,
            "trades_oos": bt_oos["trade_count"] if bt_oos else None,
            "avg_return_pct_oos": r2(bt_oos["avg_return_pct"]) if bt_oos else None,
            "weighted_return": weighted_return,
            "extension_from_ma200_pct": extension_from_ma200_pct,
            "dist_from_52w_high_pct": dist_from_52w_high_pct,
            "daily_buy_signal": daily_buy_signal,
            "weekly_status": weekly_status,
            "momentum_63d": round(momentum_63d, 2) if momentum_63d is not None else None,
            "signal_date": signal_date,
            "closes": [round(float(c), 2) for c in closes],
            "opens": [round(float(o), 2) for o in opens],
            "highs": [round(float(h), 2) for h in highs],
            "lows": [round(float(l), 2) for l in lows],
            "volumes": [int(v) for v in volumes],
        }
        rec.update(fund)
        if vcp:
            rec.update(vcp)
        else:
            rec.update({
                "vcp_stage2_pass": False, "vcp_above_52w_low_pct": None,
                "vcp_below_52w_high_pct": None, "vcp_contraction_count": 0,
                "vcp_contractions_pct": [], "vcp_has_enough_contractions": False,
                "vcp_is_monotonic": None, "vcp_is_final_tight": None,
                "vcp_is_vol_dry_up": None, "vcp_is_near_pivot": None,
                "vcp_dist_to_pivot_pct": None, "vcp_score": 0,
            })
        if gap:
            rec.update(gap)
        else:
            rec.update({
                "gap_type": None, "gap_score": None, "gap_days_ago": None,
                "gap_pct_at_event": None, "gap_down_recovery_pct": None,
                "gap_up_continuation_pct": None, "gap_up_filled": None,
            })
        return rec
    except Exception as e:
        print(f"  エラー: {ticker} - {e}")
        return None


# ---------- 新規：ギャップ分析（下げた後の戻り／上げた後の伸び） ----------

def analyze_gaps(opens, closes, lookback_days=60, pctile_down=2, pctile_up=98,
                  trough_window=10, abs_floor_pct=5.0):
    """直近lookback_days以内で、その銘柄にとって統計的に大きい部類のギャップ
    （下位pctile_down% or 上位pctile_up%、かつ絶対値でabs_floor_pct%以上）のうち、
    ギャップ幅の絶対値が最大のものを1つ選び、下ギャップなら戻り率、上ギャップなら継続率を点数化する。
    abs_floor_pctは、値動きの穏やかな銘柄で「統計的には上位2%だが実際は1%程度の些細な変動」を
    誤検出しないための下限フィルター。
    【重要】下ギャップの戻りスコアと上ギャップの継続スコアを直接比較して選ぶと、
    些細な下ギャップでも分母（下げ幅）が小さいと見かけ上高得点になってしまうバグがあったため、
    「まずギャップ幅の絶対値が最大のイベントを1つ選ぶ→その種類に応じてスコアを計算する」
    という順序にしている。"""
    N = len(closes)
    if N < 300:
        return None
    opens = np.asarray(opens, dtype=float)
    closes = np.asarray(closes, dtype=float)

    gap_pct = np.full(N, np.nan)
    for i in range(1, N):
        prev_close = closes[i - 1]
        if prev_close != 0:
            gap_pct[i] = (opens[i] - prev_close) / prev_close * 100

    valid_gaps = gap_pct[~np.isnan(gap_pct)]
    if len(valid_gaps) < 200:
        return None

    down_threshold = np.percentile(valid_gaps, pctile_down)
    up_threshold = np.percentile(valid_gaps, pctile_up)

    start = max(1, N - lookback_days)

    best_idx = None
    best_type = None
    best_abs = -1.0
    for i in range(start, N):
        g = gap_pct[i]
        if np.isnan(g):
            continue
        if g <= down_threshold and abs(g) >= abs_floor_pct and abs(g) > best_abs:
            best_idx, best_type, best_abs = i, "down", abs(g)
        elif g >= up_threshold and abs(g) >= abs_floor_pct and abs(g) > best_abs:
            best_idx, best_type, best_abs = i, "up", abs(g)

    result = {
        "gap_type": None, "gap_score": None, "gap_days_ago": None, "gap_pct_at_event": None,
        "gap_down_recovery_pct": None, "gap_up_continuation_pct": None, "gap_up_filled": None,
    }
    if best_idx is None:
        return result

    latest_close = float(closes[-1])
    idx = best_idx
    result["gap_days_ago"] = N - 1 - idx
    result["gap_pct_at_event"] = round(float(gap_pct[idx]), 2)

    if best_type == "down" and idx >= 1:
        pre_gap_close = closes[idx - 1]
        trough_end = min(N, idx + trough_window)
        trough = float(np.min(closes[idx:trough_end]))
        if pre_gap_close > trough:
            recovery_pct = (latest_close - trough) / (pre_gap_close - trough) * 100
        else:
            recovery_pct = 100.0
        score = max(0.0, min(100.0, recovery_pct))
        result["gap_type"] = "down_recovery"
        result["gap_score"] = round(score, 1)
        result["gap_down_recovery_pct"] = round(recovery_pct, 1)
    elif best_type == "up":
        pre_gap_close = closes[idx - 1]
        gap_day_close = closes[idx]
        gap_filled = bool(latest_close < pre_gap_close)
        continuation_pct = ((latest_close - gap_day_close) / gap_day_close * 100) if gap_day_close != 0 else 0.0
        score = 0.0 if gap_filled else max(0.0, min(100.0, continuation_pct * 3))
        result["gap_type"] = "up_continuation"
        result["gap_score"] = round(score, 1)
        result["gap_up_continuation_pct"] = round(continuation_pct, 1)
        result["gap_up_filled"] = gap_filled

    return result


# ---------- 新規：統合スコア（Phase5） ----------

def compute_integrated_score(records, market_regime_score):
    """fund_score・rs_percentile・vcp_scoreを統合し、地合いスコアで最終調整する。
    tech_score = RS 50% + VCP 50%
    integrated_score = sqrt(fund_score × tech_score) （幾何平均、どちらかが極端に低いと全体も下がる）
    final_score = integrated_score × (0.5 + 0.5×地合いスコア/100)  地合いが悪くても最大50%引きまで（ゼロにはしない）
    """
    df = pd.DataFrame(records)
    fund = (df["fund_score"].fillna(50) / 100).clip(lower=0, upper=1)
    rs = (df["rs_percentile"].fillna(50) / 100).clip(lower=0, upper=1)
    vcp = (df["vcp_score"].fillna(0) / 100).clip(lower=0, upper=1)

    tech = 0.5 * rs + 0.5 * vcp
    integrated = np.sqrt(fund * tech) * 100
    regime_multiplier = 0.5 + 0.5 * (market_regime_score / 100)
    final = integrated * regime_multiplier

    df["tech_score"] = (tech * 100).round(1)
    df["integrated_score"] = integrated.round(1)
    df["regime_multiplier"] = round(regime_multiplier, 3)
    df["final_score"] = final.round(1)
    return df.to_dict(orient="records")


# ---------- 新規：RSモメンタムのウォークフォワード検証（Phase5） ----------

def calc_weighted_return_at(closes, end_idx, w=(0.4, 0.2, 0.2, 0.2)):
    """calc_weighted_returnの、任意の時点(end_idx)版。"""
    def ret(days):
        base_idx = end_idx - days
        if base_idx < 0:
            return None
        base = closes[base_idx]
        if base == 0:
            return None
        return closes[end_idx] / base - 1
    r3, r6, r9, r12 = ret(63), ret(126), ret(189), ret(252)
    if any(v is None for v in (r3, r6, r9, r12)):
        return None
    return w[0]*r3 + w[1]*r6 + w[2]*r9 + w[3]*r12


def forward_return(closes, start_idx, horizon=63):
    end_idx = start_idx + horizon
    if end_idx >= len(closes):
        return None
    base = closes[start_idx]
    if base == 0:
        return None
    return closes[end_idx] / base - 1


def collect_rs_samples(closes, sample_every=21, horizon=63):
    """1銘柄の全期間から、sample_every日おきに(加重リターン, その後horizon日のフォワードリターン)の
    ペアをサンプリングする。"""
    n = len(closes)
    samples = []
    i = 252
    while i + horizon < n:
        wr = calc_weighted_return_at(closes, i)
        fr = forward_return(closes, i, horizon)
        if wr is not None and fr is not None:
            samples.append((i, n, wr, fr))
        i += sample_every
    return samples


def walkforward_rs_validation(closes_by_ticker, sample_every=21, horizon=63):
    """全銘柄からサンプルを集め、各銘柄の価格データ期間内での相対位置（前半/後半）で
    IS/OOSに分割。IS側でRS上位25%だったサンプル群と、それ以外のサンプル群について、
    OOS側での平均フォワードリターンを比較する。

    【注意】これは真の意味での「同じ日に他の391銘柄と比較した相対力」の検証ではなく、
    各銘柄「自身の」過去の強い時期がその後の自分自身のリターンを予測できているか、
    という簡易的な代理検証です（日付を横断して全銘柄をそろえるにはデータ取得方法の
    変更が必要なため、今回はこの近似にとどめています）。
    """
    rows = []
    for ticker, closes in closes_by_ticker.items():
        for idx, n, wr, fr in collect_rs_samples(closes, sample_every, horizon):
            rows.append({"ticker": ticker, "idx": idx, "n": n, "wr": wr, "fr": fr})
    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["rel_pos"] = df["idx"] / df["n"]
    is_df = df[df["rel_pos"] < 0.5]
    oos_df = df[df["rel_pos"] >= 0.5]

    def quintile_stats(d, label):
        if len(d) < 20:
            return {"note": f"{label}: サンプル数不足"}
        thresh = d["wr"].quantile(0.75)
        top = d[d["wr"] >= thresh]
        rest = d[d["wr"] < thresh]
        return {
            "sample_count": len(d),
            "top25pct_n": len(top),
            "top25pct_avg_forward_return_pct": round(top["fr"].mean() * 100, 2),
            "rest_n": len(rest),
            "rest_avg_forward_return_pct": round(rest["fr"].mean() * 100, 2),
        }

    return {
        "in_sample": quintile_stats(is_df, "IS"),
        "out_of_sample": quintile_stats(oos_df, "OOS"),
        "note": ("銘柄ごとの過去12ヶ月加重リターン(RS)上位25%サンプル vs それ以外について、"
                 "その後約3ヶ月(63営業日)のフォワードリターンを比較。in_sample/out_of_sampleは"
                 "各銘柄の価格データ期間の前半/後半（相対位置）で分割。真の同一時点での"
                 "全銘柄横断比較ではなく、銘柄ごとの自己参照的な簡易検証である点に注意。"),
    }


def sanitize_nan(obj):
    """PythonのNaNをJSON標準のnullに変換する（ブラウザのJSON.parseは裸のNaNトークンを
    受け付けないため。json.dumpのデフォルト(allow_nan=True)はPython独自拡張でNaNをそのまま
    書き出してしまい、それが原因でブラウザ側でJSONパースエラーになっていた）。"""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    return obj


def main(ema_fast=DEFAULT_DAILY_EMA_FAST, ema_slow=DEFAULT_DAILY_EMA_SLOW):
    JST = timezone(timedelta(hours=9))
    print(f"データ取得開始: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}")
    print(f"日足EMA BUY SIGNAL設定: EMA{ema_fast}/{ema_slow}")
    os.makedirs("docs", exist_ok=True)
    all_results = []
    total = len(JPX400_TICKERS)

    print(f"ベンチマーク({BENCHMARK_TICKER})取得中...")
    bench_df = fetch_benchmark_data()
    bench_return = None
    if bench_df is not None:
        bench_closes = bench_df["Close"].astype(float).values
        bench_return = calc_weighted_return(bench_closes)
    print(f"  -> ベンチマーク加重リターン: {bench_return}")
    market_regime = compute_market_regime(bench_df)
    print(f"  -> 地合いスコア: {market_regime['regime_score']} "
          f"(trend_class={market_regime['trend_class']}, above_ma200={market_regime['above_ma200']})")

    for i, ticker in enumerate(JPX400_TICKERS):
        print(f"[{i+1}/{total}] {ticker} 取得中...")
        result = fetch_and_analyze(ticker, ema_fast=ema_fast, ema_slow=ema_slow)
        if result:
            all_results.append(result)
            print(f"  -> OK RSI={result['rsi']} 終値={result['latest_close']} "
                  f"PER={result.get('trailing_pe')} 加重リターン={result.get('weighted_return')}")
        else:
            print(f"  -> スキップ")
        # 連続リクエストによるYahoo Finance側のレート制限を避けるための小休止
        time.sleep(INTER_TICKER_SLEEP_SEC)

    success_rate = len(all_results) / total if total else 0
    print(f"取得成功率: {len(all_results)}/{total} ({success_rate:.0%})")
    if success_rate < MIN_FETCH_SUCCESS_RATE:
        print(
            f"[FATAL] 取得成功率が閾値({MIN_FETCH_SUCCESS_RATE:.0%})を下回りました。"
            f"Yahoo Financeからのブロック/レート制限や、yfinanceの仕様変更が疑われます。"
            f"docs/data.jsonは書き換えずに異常終了します（直前の正常なデータを保持）。"
        )
        sys.exit(1)

    print("ファンダ複合スコアを計算中...")
    all_results = compute_fund_scores(all_results)

    print("RSランキングを計算中...")
    all_results, rs_basis = compute_rs_scores(all_results, bench_return)
    print(f"  -> RS算出方式: {rs_basis}")

    print("統合スコアを計算中...")
    all_results = compute_integrated_score(all_results, market_regime["regime_score"])

    print("RSモメンタムのウォークフォワード検証を実行中...")
    closes_by_ticker = {r["ticker"]: r["closes"] for r in all_results if r.get("closes")}
    rs_validation = walkforward_rs_validation(closes_by_ticker)
    if rs_validation:
        print(f"  -> IS : {rs_validation['in_sample']}")
        print(f"  -> OOS: {rs_validation['out_of_sample']}")

    # ==== EMA9/21 BUY SIGNALS 用サマリー ====
    buy_signals = [r for r in all_results if r.get("daily_buy_signal")]
    weekly_ok_count = sum(1 for r in buy_signals if r.get("weekly_status") == "OK")
    signal_dates = [r.get("signal_date") for r in all_results if r.get("signal_date")]
    ema_signal_date = max(set(signal_dates), key=signal_dates.count) if signal_dates else None
    ema_summary = {
        "signal_date": ema_signal_date,
        "daily_ema_fast": ema_fast,
        "daily_ema_slow": ema_slow,
        "buy_signal_count": len(buy_signals),
        "weekly_ok_count": weekly_ok_count,
        "fetch_success": len(all_results),
        "fetch_total": total,
    }
    print(f"  -> EMA{ema_fast}/{ema_slow} BUY: {len(buy_signals)}銘柄（うち週足OK {weekly_ok_count}銘柄）シグナル日={ema_signal_date}")

    run_time = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    output = {
        "run_time": run_time,
        "ticker_count": len(all_results),
        "benchmark_weighted_return": bench_return,
        "rs_basis": rs_basis,
        "market_regime": market_regime,
        "rs_momentum_validation": rs_validation,
        "ema_signal_summary": ema_summary,
        "data": all_results,
    }
    output = sanitize_nan(output)
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, allow_nan=False)

    print(f"\n完了: {len(all_results)}銘柄 -> docs/data.json")

def parse_args():
    parser = argparse.ArgumentParser(description="サテライト銘柄スクリーナー（EMA BUY SIGNALのEMA期間は変更可能）")
    parser.add_argument("--daily-ema-fast", type=int, default=DEFAULT_DAILY_EMA_FAST,
                         help=f"日足BUYシグナルの速いEMA期間（デフォルト{DEFAULT_DAILY_EMA_FAST}）")
    parser.add_argument("--daily-ema-slow", type=int, default=DEFAULT_DAILY_EMA_SLOW,
                         help=f"日足BUYシグナルの遅いEMA期間（デフォルト{DEFAULT_DAILY_EMA_SLOW}）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(ema_fast=args.daily_ema_fast, ema_slow=args.daily_ema_slow)
