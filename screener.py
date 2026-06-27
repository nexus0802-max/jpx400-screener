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

def fetch_and_analyze(ticker):
    try:
        t = f"{ticker}.T"
        df = yf.download(t, period="10y", progress=False, auto_adjust=True)
        if df is None or len(df) < 120:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) < 120:
            return None

        closes = df["Close"].astype(float).values
        volumes = df["Volume"].astype(float).values
        N = len(closes)

        # 移動平均
        ma20  = float(np.mean(closes[-20:]))
        ma50  = float(np.mean(closes[-50:]))
        ma100 = float(np.mean(closes[-100:]))

        # 移動平均の傾き（右肩上がり判定）
        ma20_slope  = float(np.mean(closes[-20:]) - np.mean(closes[-25:-5]))
        ma50_slope  = float(np.mean(closes[-50:]) - np.mean(closes[-60:-10]))
        ma100_slope = float(np.mean(closes[-100:]) - np.mean(closes[-110:-10]))

        # 平均出来高（20日）
        avg_vol20 = float(np.mean(volumes[-20:]))

        # 52週高値
        weeks52 = min(252, N)
        high52w = float(np.max(closes[-weeks52:]))

        # 最新株価
        latest = float(closes[-1])

        # 時価総額（株数取得）
        info = yf.Ticker(t).fast_info
        shares = getattr(info, 'shares', None)
        market_cap = float(shares * latest) if shares else None

        # 直近1ヶ月の値幅（持ち合い判定）
        month_closes = closes[-21:]
        month_high = float(np.max(month_closes))
        month_low = float(np.min(month_closes))
        month_range_pct = (month_high - month_low) / month_low * 100 if month_low > 0 else 999

        # RSI
        rsi = calc_rsi(closes)

        # MA200（パターンスクリーナー用）
        ma200 = float(np.mean(closes[-200:])) if N >= 200 else float(np.mean(closes))

        return {
            "ticker": ticker,
            "latest": round(latest, 2),
            "ma20": round(ma20, 2),
            "ma50": round(ma50, 2),
            "ma100": round(ma100, 2),
            "ma20_slope": round(ma20_slope, 4),
            "ma50_slope": round(ma50_slope, 4),
            "ma100_slope": round(ma100_slope, 4),
            "avg_vol20": round(avg_vol20, 0),
            "high52w": round(high52w, 2),
            "market_cap": round(market_cap / 1e8, 1) if market_cap else None,  # 億円
            "month_range_pct": round(month_range_pct, 2),
            "rsi": round(rsi, 1),
            "ma200": round(ma200, 2),
            "above_ma200": bool(latest > ma200),
            "closes": [round(float(c), 2) for c in closes],
        }
    except Exception as e:
        print(f"  エラー: {ticker} - {e}")
        return None

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
            print(f"  -> OK RSI={result['rsi']} 終値={result['latest']}")
        else:
            print(f"  -> スキップ")

    run_time = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    output = {
        "run_time": run_time,
        "ticker_count": len(all_results),
        "data": all_results,
    }
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<title>JPX400 スクリーナー</title>
<style>body{{font-family:sans-serif;padding:2rem;background:#f5f5f0}}
.card{{background:#fff;border-radius:12px;border:1px solid #e5e5e5;padding:1.5rem;max-width:500px}}
a{{color:#378ADD;display:block;margin:.5rem 0;font-size:15px}}</style></head>
<body><div class="card">
<h2>📊 JPX400 スクリーナー</h2>
<p style="color:#666;margin:.5rem 0 1rem">更新: {run_time} / {len(all_results)}銘柄</p>
<a href="screener.html">▶ パターンマッチング スクリーナー</a>
<a href="swing.html">▶ スイングトレード スクリーナー</a>
</div></body></html>""")

    print(f"\n完了: {len(all_results)}銘柄 -> docs/data.json")

if __name__ == "__main__":
    main()
