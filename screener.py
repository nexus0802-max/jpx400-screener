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

def calc_adx_ci(highs, lows, closes, period=14, lookback=60):
    """直近lookback本平均のADXとチョピネス指数(CI)を返す。データ不足時は(None, None)"""
    n = len(closes)
    if n < period * 2 + 1:
        return None, None

    tr  = np.zeros(n)
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    for i in range(1, n):
        h, l, pc = highs[i], lows[i], closes[i-1]
        ph, pl = highs[i-1], lows[i-1]
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        up, down = h - ph, pl - l
        pdm[i] = up if (up > down and up > 0) else 0
        mdm[i] = down if (down > up and down > 0) else 0

    sm_tr  = np.full(n, np.nan)
    sm_pdm = np.full(n, np.nan)
    sm_mdm = np.full(n, np.nan)
    sm_tr[period]  = tr[1:period+1].sum()
    sm_pdm[period] = pdm[1:period+1].sum()
    sm_mdm[period] = mdm[1:period+1].sum()
    for i in range(period + 1, n):
        sm_tr[i]  = sm_tr[i-1]  - sm_tr[i-1]/period  + tr[i]
        sm_pdm[i] = sm_pdm[i-1] - sm_pdm[i-1]/period + pdm[i]
        sm_mdm[i] = sm_mdm[i-1] - sm_mdm[i-1]/period + mdm[i]

    dx = np.full(n, np.nan)
    for i in range(period, n):
        if sm_tr[i] == 0:
            pdi = mdi = 0.0
        else:
            pdi = 100 * sm_pdm[i] / sm_tr[i]
            mdi = 100 * sm_mdm[i] / sm_tr[i]
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
        hh = highs[i-period+1:i+1].max()
        ll = lows[i-period+1:i+1].min()
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

        highs   = df["High"].astype(float).values
        lows    = df["Low"].astype(float).values
        closes  = df["Close"].astype(float).values
        volumes = df["Volume"].astype(float).values
        N = len(closes)

        rsi = calc_rsi(closes)
        ma200 = float(np.mean(closes[-200:])) if N >= 200 else float(np.mean(closes))
        latest_close = float(closes[-1])
        adx, ci = calc_adx_ci(highs, lows, closes)
        trend_class = classify_trend(adx, ci)

        return {
            "ticker": ticker,
            "rsi": round(rsi, 1),
            "ma200": round(ma200, 2),
            "latest_close": round(latest_close, 2),
            "above_ma200": bool(latest_close > ma200),
            "adx": round(adx, 1) if adx is not None else None,
            "ci": round(ci, 1) if ci is not None else None,
            "trend_class": trend_class,
            "closes":  [round(float(c), 2) for c in closes],
            "volumes": [int(v) for v in volumes],
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
            print(f"  -> OK RSI={result['rsi']} 終値={result['latest_close']}")
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
<a href="breakout.html">▶ MA収束ブレイクアウト スクリーナー</a>
<a href="vcp.html">▶ VCP（収縮パターン）スクリーナー</a>
<a href="trend.html">▶ トレンド/レンジ 判定スクリーナー</a>
</div></body></html>""")

    write_trend_page()

    print(f"\n完了: {len(all_results)}銘柄 -> docs/data.json")

def write_trend_page():
    html = """<!DOCTYPE html>
<html lang="ja"><head><meta charset="UTF-8">
<title>トレンド/レンジ 判定スクリーナー</title>
<style>
body{font-family:sans-serif;padding:2rem;background:#f5f5f0;color:#222}
.card{background:#fff;border-radius:12px;border:1px solid #e5e5e5;padding:1.5rem;max-width:900px;margin:0 auto}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:1rem}
th{text-align:left;padding:8px 6px;border-bottom:1px solid #ccc;cursor:pointer;color:#666}
td{padding:8px 6px;border-bottom:1px solid #eee}
.badge{font-size:12px;padding:2px 8px;border-radius:6px}
.trend{background:#EAF3DE;color:#27500A}
.range{background:#FCEBEB;color:#791F1F}
.mid{background:#F1EFE8;color:#444441}
a.back{color:#378ADD;font-size:14px}
</style></head>
<body><div class="card">
<a class="back" href="index.html">&larr; 戻る</a>
<h2>トレンド/レンジ 判定スクリーナー</h2>
<p style="color:#666" id="meta"></p>
<canvas id="scatter" style="max-height:340px"></canvas>
<table id="tbl"><thead><tr>
<th data-k="ticker">銘柄</th><th data-k="adx">平均ADX</th><th data-k="ci">平均CI</th><th data-k="trend_class">判定</th>
</tr></thead><tbody></tbody></table>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const label = c => c==='trend' ? 'トレンド向き' : c==='range' ? 'レンジ・往来' : '中間';
let sortKey='adx', sortDir=-1;
fetch('data.json').then(r=>r.json()).then(d=>{
  const rows = d.data.filter(r=>r.adx!=null && r.ci!=null);
  document.getElementById('meta').textContent = `更新: ${d.run_time} / 判定対象 ${rows.length}銘柄`;
  function render(){
    const sorted = rows.slice().sort((a,b)=> sortKey==='ticker'||sortKey==='trend_class'
      ? sortDir*String(a[sortKey]).localeCompare(String(b[sortKey]))
      : sortDir*(a[sortKey]-b[sortKey]));
    document.querySelector('#tbl tbody').innerHTML = sorted.map(r=>
      `<tr><td>${r.ticker}</td><td>${r.adx.toFixed(1)}</td><td>${r.ci.toFixed(1)}</td>`+
      `<td><span class="badge ${r.trend_class}">${label(r.trend_class)}</span></td></tr>`).join('');
  }
  document.querySelectorAll('th[data-k]').forEach(th=>th.addEventListener('click',()=>{
    const k=th.dataset.k;
    if(sortKey===k) sortDir*=-1; else {sortKey=k; sortDir = (k==='ticker'||k==='trend_class')?1:-1;}
    render();
  }));
  render();
  const colors = {trend:'#1D9E75', range:'#D85A30', mid:'#888780'};
  new Chart(document.getElementById('scatter'), {
    type:'scatter',
    data:{datasets:[{label:'銘柄', data:rows.map(r=>({x:r.ci,y:r.adx,ticker:r.ticker})),
      backgroundColor: rows.map(r=>colors[r.trend_class])}]},
    options:{plugins:{legend:{display:false}, tooltip:{callbacks:{label:c=>`${c.raw.ticker} ADX:${c.raw.y.toFixed(1)} CI:${c.raw.x.toFixed(1)}`}}},
      scales:{x:{title:{display:true,text:'チョピネス指数（低いほどトレンド）'}}, y:{title:{display:true,text:'ADX（高いほどトレンド）'}}}}
  });
});
</script>
</body></html>"""
    with open("docs/trend.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    main()
