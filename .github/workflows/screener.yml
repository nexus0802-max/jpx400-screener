import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

# JPX400 主要銘柄（代表的な400銘柄）
JPX400_TICKERS = [
    "1332","1333","1375","1379","1383","1384","1385","1386","1387","1388",
    "1605","1721","1801","1802","1803","1804","1808","1812","1815","1820",
    "1925","1928","1942","1944","1951","1952","1954","1963","1975","1976",
    "2002","2003","2004","2006","2009","2012","2031","2035","2060","2062",
    "2108","2109","2114","2117","2120","2121","2122","2127","2130","2132",
    "2201","2202","2203","2204","2206","2207","2208","2215","2220","2222",
    "2267","2268","2269","2270","2281","2282","2284","2286","2288","2292",
    "2301","2303","2305","2307","2309","2315","2317","2321","2325","2326",
    "2371","2372","2375","2376","2378","2379","2384","2385","2389","2393",
    "2413","2414","2415","2417","2418","2419","2427","2428","2432","2433",
    "2462","2463","2464","2469","2471","2475","2477","2479","2484","2485",
    "2501","2502","2503","2531","2533","2537","2539","2540","2579","2580",
    "2601","2602","2607","2612","2613","2614","2615","2616","2617","2621",
    "2651","2653","2659","2660","2670","2674","2678","2681","2685","2689",
    "2701","2702","2705","2706","2712","2713","2715","2719","2726","2729",
    "2768","2769","2784","2785","2786","2788","2789","2790","2791","2792",
    "2801","2802","2804","2805","2806","2809","2810","2811","2812","2813",
    "2871","2872","2874","2875","2876","2877","2878","2879","2880","2882",
    "2901","2902","2903","2904","2905","2906","2907","2908","2909","2910",
    "3003","3004","3005","3006","3007","3008","3009","3010","3011","3012",
    "3101","3103","3104","3105","3106","3107","3108","3109","3110","3111",
    "3201","3202","3203","3204","3205","3206","3207","3208","3209","3210",
    "3289","3291","3292","3293","3294","3295","3296","3297","3298","3299",
    "3401","3402","3403","3404","3405","3406","3407","3408","3409","3410",
    "3501","3502","3503","3504","3505","3506","3507","3508","3509","3510",
    "3601","3602","3603","3604","3605","3606","3607","3608","3609","3610",
    "3701","3702","3703","3704","3705","3706","3707","3708","3709","3710",
    "3801","3802","3803","3804","3805","3806","3807","3808","3809","3810",
    "3861","3863","3864","3865","3866","3867","3868","3869","3870","3871",
    "4004","4005","4006","4007","4008","4009","4010","4011","4012","4013",
    "4041","4042","4043","4044","4045","4046","4047","4048","4049","4050",
    "4151","4152","4153","4154","4155","4156","4157","4158","4159","4160",
    "4183","4184","4185","4186","4187","4188","4189","4190","4191","4192",
    "4208","4209","4210","4211","4212","4213","4214","4215","4216","4217",
    "4301","4302","4303","4304","4305","4306","4307","4308","4309","4310",
    "4401","4402","4403","4404","4405","4406","4407","4408","4409","4410",
    "4502","4503","4504","4505","4506","4507","4508","4509","4510","4511",
    "4601","4602","4603","4604","4605","4606","4607","4608","4609","4610",
    "4661","4662","4663","4664","4665","4666","4667","4668","4669","4670",
]

# 重複除去
JPX400_TICKERS = list(dict.fromkeys(JPX400_TICKERS))

PARAMS = {
    "window": 5,
    "forward": 5,
    "sim_threshold": 0.80,
    "min_matches": 10,
    "use_rsi_filter": True,
    "rsi_threshold": 40,
}

def get_price(row, price_type="close"):
    if price_type == "hl2":
        return (row["High"] + row["Low"]) / 2
    return row["Close"]

def normalize_return(arr):
    rets = []
    for i in range(1, len(arr)):
        rets.append((arr[i] - arr[i-1]) / arr[i-1] if arr[i-1] != 0 else 0)
    return np.array(rets)

def cosine_sim(a, b):
    length = min(len(a), len(b))
    a, b = a[:length], b[:length]
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0
    return dot / (na * nb)

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def fetch_data(ticker):
    try:
        t = f"{ticker}.T"
        df = yf.download(t, period="10y", progress=False, auto_adjust=True)
        if df is None or len(df) < 50:
            return None
        df = df.dropna(subset=["Close"])
        return df
    except Exception:
        return None

def screen_ticker(ticker, params):
    df = fetch_data(ticker)
    if df is None:
        return None

    closes = df["Close"].values.flatten()
    N = len(closes)
    wLen = params["window"]
    fwd = params["forward"]

    if N < wLen + fwd + 30:
        return None

    rsi = calc_rsi(closes)
    if params["use_rsi_filter"] and rsi >= params["rsi_threshold"]:
        return None

    query = normalize_return(closes[N - wLen:])
    returns = []

    for i in range(N - wLen - fwd):
        seg = normalize_return(closes[i:i + wLen])
        sim = cosine_sim(query, seg)
        if sim >= params["sim_threshold"]:
            ep = closes[i + wLen - 1]
            xp = closes[i + wLen - 1 + fwd]
            if ep > 0:
                returns.append((xp - ep) / ep * 100)

    if len(returns) < params["min_matches"]:
        return None

    returns = np.array(returns)
    mean = float(np.mean(returns))
    median = float(np.median(returns))
    win_rate = float(np.mean(returns > 0))
    std = float(np.std(returns)) or 0.01
    sharpe = mean / std
    score = win_rate * 40 + min(mean, 10) * 3 + min(sharpe, 3) * 20
    recent_prices = [float(x) for x in closes[-wLen:]]

    return {
        "ticker": ticker,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "win_rate": round(win_rate, 3),
        "match_count": len(returns),
        "std": round(std, 2),
        "sharpe": round(sharpe, 2),
        "max_dd": round(float(np.min(returns)), 2),
        "max_up": round(float(np.max(returns)), 2),
        "score": round(score, 1),
        "rsi": round(rsi, 1),
        "recent_prices": recent_prices,
        "returns": [round(float(r), 2) for r in returns],
    }

def generate_html(results, params, run_time):
    if not results:
        top_html = "<p style='color:#aaa;text-align:center;padding:2rem'>シグナル銘柄なし</p>"
    else:
        rows = ""
        for i, r in enumerate(results):
            mean_badge = (
                f'<span style="background:#e8f5e0;color:#2d6a0a;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">+{r["mean"]}%</span>'
                if r["mean"] >= 0
                else f'<span style="background:#fce8e8;color:#a32d2d;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600">{r["mean"]}%</span>'
            )
            sig = ("強" if r["mean"] > 2 and r["win_rate"] > 0.6
                   else "中" if r["mean"] > 0 and r["win_rate"] > 0.5
                   else "弱")
            sig_col = "#2d6a0a" if sig == "強" else "#888" if sig == "中" else "#a32d2d"
            score_w = min(100, int(r["score"]))
            rows += f"""<tr>
                <td style="color:#aaa;width:32px">{i+1}</td>
                <td style="font-weight:600">{r['ticker']}</td>
                <td><span style="display:inline-block;height:8px;width:{score_w}px;background:#378ADD;border-radius:4px;margin-right:4px;vertical-align:middle"></span>{score_w}</td>
                <td>{mean_badge}</td>
                <td>{r['win_rate']*100:.1f}%</td>
                <td>{r['match_count']}</td>
                <td style="color:#666">{r['median']}%</td>
                <td style="color:#a32d2d">{r['max_dd']}%</td>
                <td style="color:#2d6a0a">+{r['max_up']}%</td>
                <td style="color:{sig_col};font-weight:600">{sig}</td>
            </tr>"""
        top_html = f"""
        <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead>
                <tr style="border-bottom:2px solid #e5e5e5">
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">#</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">銘柄</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">スコア</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">期待リターン</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">勝率</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">マッチ数</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">中央値</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">最大下落</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">最大上昇</th>
                    <th style="padding:8px;text-align:left;font-size:11px;color:#888">シグナル</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        </div>"""

    params_html = "".join([
        f'<span style="font-size:12px;background:#f0f0f0;padding:3px 12px;border-radius:20px;color:#555">{k}: {v}</span>'
        for k, v in params.items()
    ])

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JPX400 スクリーナー結果 {run_time}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;background:#f5f5f0;color:#1a1a1a;margin:0;padding:0}}
.wrap{{max-width:1100px;margin:0 auto;padding:1.5rem 1rem}}
.card{{background:#fff;border-radius:12px;border:1px solid #e5e5e5;padding:1.25rem 1.5rem;margin-bottom:1.25rem}}
h1{{font-size:20px;font-weight:600;margin-bottom:.25rem}}
.meta{{font-size:13px;color:#888;margin-bottom:1rem}}
tbody tr:hover{{background:#fafafa}}
tbody td{{padding:8px;border-bottom:1px solid #f0f0f0}}
.warn{{font-size:12px;color:#999;margin-top:1rem;padding-top:.75rem;border-top:1px solid #f0f0f0;line-height:1.7}}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>📊 JPX400 パターンマッチング スクリーナー</h1>
    <div class="meta">実行日時: {run_time}　|　シグナル銘柄数: {len(results)}件</div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1rem">{params_html}</div>
    {top_html}
    <p class="warn">⚠️ 本ツールは過去データの統計分析です。将来の値動きを保証するものではありません。投資判断はご自身の責任で。</p>
  </div>
</div>
</body>
</html>"""
    return html

def main():
    print(f"スクリーニング開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = []
    total = len(JPX400_TICKERS)
    for i, ticker in enumerate(JPX400_TICKERS):
        print(f"[{i+1}/{total}] {ticker} 処理中...")
        result = screen_ticker(ticker, PARAMS)
        if result:
            results.append(result)
            print(f"  → シグナル検出: 期待リターン={result['mean']}% 勝率={result['win_rate']*100:.1f}%")

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"\n完了: {len(results)}銘柄がシグナル条件を満たしました")

    os.makedirs("docs", exist_ok=True)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M JST")
    html = generate_html(results, PARAMS, run_time)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    with open("docs/results.json", "w", encoding="utf-8") as f:
        json.dump({"run_time": run_time, "results": results}, f, ensure_ascii=False, indent=2)

    print("docs/index.html を生成しました")

if __name__ == "__main__":
    main()
