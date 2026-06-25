import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

JPX400_TICKERS = [
    "1332","1333","1375","1379","1605","1721","1801","1802","1803","1804",
    "1808","1812","1820","1925","1928","1942","1944","1951","1963","1975",
    "2002","2003","2004","2006","2009","2031","2060","2108","2114","2121",
    "2201","2202","2206","2207","2267","2269","2281","2282","2284","2292",
    "2301","2303","2307","2309","2315","2317","2321","2325","2371","2375",
    "2413","2427","2432","2433","2462","2464","2471","2479","2501","2502",
    "2503","2531","2579","2601","2602","2607","2651","2653","2670","2674",
    "2678","2681","2701","2702","2712","2726","2768","2784","2801","2802",
    "2871","2872","2875","2901","2902","3003","3004","3005","3101","3103",
    "3201","3289","3291","3292","3401","3402","3405","3407","3501","3502",
    "3601","3602","3701","3702","3801","3802","3861","3863","4004","4005",
    "4041","4042","4043","4151","4152","4183","4184","4188","4208","4209",
    "4301","4302","4401","4402","4403","4502","4503","4504","4506","4507",
    "4601","4602","4661","4662","4663","4751","4755","4901","4902","4911",
    "5001","5002","5019","5101","5108","5201","5202","5214","5301","5332",
    "5333","5401","5402","5406","5411","5412","5541","5542","5631","5703",
    "5706","5711","5713","5714","5715","5801","5802","5803","5901","5938",
    "6098","6103","6104","6113","6146","6178","6273","6301","6302","6305",
    "6306","6361","6367","6383","6412","6417","6471","6472","6473","6481",
    "6501","6503","6504","6506","6532","6541","6586","6594","6645","6657",
    "6674","6701","6702","6703","6723","6724","6752","6753","6758","6762",
    "6767","6770","6773","6807","6841","6857","6861","6902","6952","6954",
    "6971","6981","6988","7003","7004","7011","7013","7201","7202","7203",
    "7205","7211","7261","7267","7269","7270","7272","7731","7733","7735",
    "7741","7751","7752","7762","7832","7911","7912","7951","7974","8001",
    "8002","8003","8008","8011","8015","8031","8035","8036","8053","8058",
    "8097","8113","8114","8233","8252","8267","8301","8304","8306","8308",
    "8309","8316","8331","8354","8355","8377","8411","8473","8591","8601",
    "8604","8628","8630","8697","8725","8729","8750","8766","8795","8801",
    "8802","8803","8804","8830","9001","9005","9007","9008","9009","9020",
    "9021","9022","9064","9101","9104","9107","9201","9202","9301","9432",
    "9433","9434","9501","9502","9503","9531","9532","9602","9613","9619",
    "9633","9684","9697","9735","9766","9983","9984",
]
JPX400_TICKERS = list(dict.fromkeys(JPX400_TICKERS))

def normalize_return(arr):
    rets = []
    for i in range(1, len(arr)):
        prev = arr[i-1]
        rets.append((arr[i] - prev) / prev if prev != 0 else 0)
    return np.array(rets)

def cosine_sim(a, b):
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return dot / (na * nb) if na > 0 and nb > 0 else 0

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    ag = np.mean(gains[:period])
    al = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        ag = (ag * (period-1) + gains[i]) / period
        al = (al * (period-1) + losses[i]) / period
    return 100 if al == 0 else 100 - 100 / (1 + ag/al)

def fetch_and_analyze(ticker):
    try:
        df = yf.download(f"{ticker}.T", period="10y", progress=False, auto_adjust=True)
        if df is None or len(df) < 100:
            return None
        df = df.dropna(subset=["Close"])
        closes = df["Close"].values.flatten()
        highs = df["High"].values.flatten()
        lows = df["Low"].values.flatten()
        vols = df["Volume"].values.flatten()
        N = len(closes)

        rsi = calc_rsi(closes)
        ma200 = float(np.mean(closes[-200:])) if N >= 200 else float(np.mean(closes))
        latest_close = float(closes[-1])

        # 直近20日の騰落率パターン（ブラウザ側で任意windowに対応するため長めに保存）
        pattern_window = 20
        recent_returns = []
        if N >= pattern_window + 1:
            seg = closes[-(pattern_window+1):]
            for i in range(1, len(seg)):
                recent_returns.append(round(float((seg[i]-seg[i-1])/seg[i-1]*100), 4))

        # 過去全データの日次リターンを保存（ブラウザ側でマッチング計算）
        all_closes = [round(float(c), 2) for c in closes]

        return {
            "ticker": ticker,
            "rsi": round(float(rsi), 1),
            "ma200": round(ma200, 2),
            "latest_close": round(latest_close, 2),
            "above_ma200": latest_close > ma200,
            "closes": all_closes,
            "recent_returns": recent_returns,
        }
    except Exception as e:
        print(f"  エラー: {e}")
        return None

def main():
    print(f"データ取得開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs("docs", exist_ok=True)
    all_results = []
    total = len(JPX400_TICKERS)

    for i, ticker in enumerate(JPX400_TICKERS):
        print(f"[{i+1}/{total}] {ticker} 取得中...")
        result = fetch_and_analyze(ticker)
        if result:
            all_results.append(result)
            print(f"  → OK RSI={result['rsi']} 終値={result['latest_close']}")

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    output = {
        "run_time": run_time,
        "ticker_count": len(all_results),
        "data": all_results,
    }
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    print(f"\n完了: {len(all_results)}銘柄のデータを docs/data.json に保存しました")

    # シンプルなindex.htmlも生成
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head><meta charset="UTF-8"><meta http-equiv="refresh" content="0;url=screener.html">
<title>JPX400 スクリーナー</title></head>
<body><p>リダイレクト中...</p></body>
</html>"""
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("docs/index.html (リダイレクト) を生成しました")

if __name__ == "__main__":
    main()
