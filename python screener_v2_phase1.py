"""
サテライト銘柄スクリーナー v2 - Phase 1
ファンダメンタルズ取得＋複合スコア算出のみを追加した段階。
既存の価格取得・テクニカル計算ロジック（calc_rsi / calc_adx_ci / classify_trend）はそのまま流用。
backtest_sma5_20 のみ、ストップロスのギャップダウン対応を修正済み。

このPhaseでは表示用HTMLは作らない（Phase 5で対応）。
docs/data.json に fund_* フィールドが正しく載るかをまず確認するためのスクリプト。
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
import os

JPX400_TICKERS = [
    "1332","1414","1419","1518","1605","1662","1719","1721","1801","1802",
    "1808","1812","1878","1911","1925","1928","1942","1951","1959","1969",
    "2124","2127","2154","2168","2181","2201","2222","2229","2264","2267",
    "2269","2317","2327","2371","2379","2384","2413","2502","2503","2531",
    "2587","2670","2678","2685","2702","2726","2760","2767","2768","2801",
    "2802","2811","2871","2875","2897","2914","3003","3038","3064","3086",
    "3088","3092","3107","3116","3132","3141","3148","3186","3231","3288",
    "3289","3291","3349","3360","3382","3391","3402","3405","3436","3465",
    "3549","3563","3626","3635","3659","3697","3765","3769","3774","3861",
    "3923","4004","4021","4042","4062","4063","4088","4091","4151","4182",
    "4183","4186","4188","4194","4202","4203","4204","4307","4385","4401",
    "4403","4452","4503","4507","4516","4519","4523","4527","4528","4543",
    "4568","4578","4587","4612","4613","4626","4661","4666","4680","4681",
    "4684","4686","4689","4704","4716","4722","4732","4768","4812","4816",
    "4901","4912","4967","4980","5019","5020","5021","5032","5076","5101",
    "5105","5108","5332","5333","5334","5344","5393","5401","5406","5411",
    "5423","5444","5471","5480","5706","5713","5714","5802","5803","5805",
    "5857","5929","5947","5991","6005","6028","6098","6101","6113","6141",
    "6146","6201","6254","6269","6273","6301","6305","6323","6326","6361",
    "6367","6368","6383","6417","6432","6436","6448","6460","6465","6479",
    "6501","6503","6504","6506","6532","6544","6586","6590","6632","6645",
    "6670","6701","6702","6707","6723","6724","6728","6752","6758","6762",
    "6787","6806","6841","6845","6849","6856","6857","6861","6869","6902",
    "6920","6951","6954","6965","6966","6981","6988","7003","7004","7011",
    "7012","7013","7014","7105","7148","7164","7167","7186","7202","7203",
    "7211","7259","7261","7267","7269","7270","7272","7276","7282","7309",
    "7419","7453","7459","7532","7550","7564","7599","7649","7701","7716",
    "7729","7733","7735","7740","7741","7744","7747","7751","7762","7832",
    "7846","7867","7906","7912","7936","7944","7951","7974","7988","7994",
    "8001","8002","8015","8020","8031","8035","8053","8056","8058","8060",
    "8078","8088","8098","8111","8113","8130","8133","8136","8154","8174",
    "8194","8227","8252","8253","8279","8306","8308","8309","8316","8331",
    "8354","8410","8411","8424","8425","8439","8473","8515","8572","8584",
    "8591","8593","8601","8604","8630","8697","8725","8750","8766","8801",
    "8802","8804","8830","8848","8850","8876","8919","8923","9005","9006",
    "9007","9008","9009","9021","9022","9024","9041","9064","9065","9069",
    "9090","9101","9104","9107","9110","9119","9142","9143","9147","9201",
    "9202","9302","9418","9432","9433","9434","9435","9449","9502","9503",
    "9506","9507","9508","9509","9513","9531","9532","9602","9684","9697",
    "9719","9735","9744","9759","9766","9843","9962","9983","9984","9989",
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
    """
    【修正済み】ストップロスの約定価格が、当日始値がストップを下回ってギャップダウンした
    場合はその始値で約定するように修正（従来は理論上のストップ価格で約定した前提だった）。
    """
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
                # ★修正点：始値がストップより下でギャップダウンしていたら始値で約定
                exit_price = min(stop_price, opens[i])
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


# ---------- 新規：ファンダメンタルズ取得 ----------

FUND_KEYS = ["trailing_pe", "price_to_book", "roe", "profit_margin",
             "revenue_growth", "earnings_growth"]

def fetch_fundamentals(ticker):
    """yfinanceの.infoからファンダメンタルズ指標を取得。
    JP株は項目が欠損することが多いので、取れない項目はNoneのまま返す
    （複合スコア側で母集団平均補完する）。"""
    try:
        t = f"{ticker}.T"
        info = yf.Ticker(t).info
        pe = info.get("trailingPE")
        # 赤字企業のマイナスPERは「割安」ではなく単に無意味なので欠損扱いにする
        if pe is not None and pe <= 0:
            pe = None
        return {
            "trailing_pe": pe,
            "price_to_book": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
        }
    except Exception as e:
        print(f"  ファンダ取得エラー: {ticker} - {e}")
        return {k: None for k in FUND_KEYS}


# ---------- 価格取得＋テクニカル（既存＋ファンダ統合） ----------

def fetch_and_analyze(ticker):
    try:
        t = f"{ticker}.T"
        df = yf.download(t, period="10y", progress=False, auto_adjust=True)
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
            "avg_return_pct": r2(bt["avg_return_pct"]) if bt else None,
            "closes": [round(float(c), 2) for c in closes],
            "volumes": [int(v) for v in volumes],
        }
        rec.update(fund)
        return rec
    except Exception as e:
        print(f"  エラー: {ticker} - {e}")
        return None


# ---------- 新規：ファンダ複合スコア（パーセンタイル方式） ----------

def compute_fund_scores(records, weights=(0.30, 0.40, 0.30)):
    """
    JPX400内でのパーセンタイル順位に基づく複合スコア（0〜100）を付与する。
    実数の標準化ではなく順位ベースにすることで、外れ値（極端なPERなど）の影響を抑える。
    欠損値はその指標だけ0.5（中央値相当）で補完し、複合スコア全体が歪まないようにする。
    weights = (割安性, 収益の質, 成長性)
    """
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


def main():
    JST = timezone(timedelta(hours=9))
    print(f"データ取得開始: {datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S JST')}")
    os.makedirs("docs", exist_ok=True)
    all_results = []
    total = len(JPX400_TICKERS)

    for i, ticker in enumerate(JPX400_TICKERS):
        print(f"[{i+1}/{total}] {ticker} 取得中...")
        result = fetch_and_analyze(ticker)
        if result:
            all_results.append(result)
            print(f"  -> OK RSI={result['rsi']} 終値={result['latest_close']} "
                  f"PER={result.get('trailing_pe')} ROE={result.get('roe')}")
        else:
            print(f"  -> スキップ")

    print("ファンダ複合スコアを計算中...")
    all_results = compute_fund_scores(all_results)

    run_time = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    output = {"run_time": run_time, "ticker_count": len(all_results), "data": all_results}
    with open("docs/data_v2_phase1.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"\n完了: {len(all_results)}銘柄 -> docs/data_v2_phase1.json")
    print("（既存のdocs/data.jsonは上書きしていません。中身を確認してから本番に反映してください）")

if __name__ == "__main__":
    main()