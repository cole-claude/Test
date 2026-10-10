"""Stock Rater: enter a ticker, get bull / base / bear cases and an overall rating.

Data: free Yahoo Finance data via yfinance. Every metric is scored from -1 (bearish) to +1 (bullish)
by linear interpolation between a "bad" and a "good" threshold, then rolled up by category.
"""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

HORIZONS = {  # category weights per time horizon (missing = 0)
    "Short-term": {"Valuation": .10, "Profitability": .10, "Growth": .10, "Financial health": .10,
                   "Technicals": .35, "Analyst": .20, "Risk": .05},
    "Long-term": {"Valuation": .20, "Profitability": .20, "Growth": .20, "Financial health": .15,
                  "Track record": .15, "Analyst": .05, "Risk": .05},
}

# (category, label, yfinance info key, "good" threshold, "bad" threshold)
# If good < bad, lower is better.
INFO_METRICS = [
    ("Valuation", "Trailing P/E", "trailingPE", 15, 35),
    ("Valuation", "Forward P/E", "forwardPE", 14, 32),
    ("Valuation", "PEG ratio", "pegRatio", 1, 2.5),
    ("Valuation", "Price / Book", "priceToBook", 2, 10),
    ("Valuation", "Price / Sales", "priceToSalesTrailing12Months", 2, 10),
    ("Valuation", "EV / EBITDA", "enterpriseToEbitda", 9, 25),
    ("Valuation", "EV / Revenue", "enterpriseToRevenue", 2, 10),
    ("Profitability", "Net margin", "profitMargins", .15, .03),
    ("Profitability", "Operating margin", "operatingMargins", .20, .05),
    ("Profitability", "Gross margin", "grossMargins", .50, .20),
    ("Profitability", "EBITDA margin", "ebitdaMargins", .25, .08),
    ("Profitability", "Return on equity", "returnOnEquity", .18, .04),
    ("Profitability", "Return on assets", "returnOnAssets", .08, .01),
    ("Growth", "Revenue growth (YoY)", "revenueGrowth", .12, -.02),
    ("Growth", "Earnings growth (YoY)", "earningsGrowth", .12, -.05),
    ("Growth", "Quarterly earnings growth", "earningsQuarterlyGrowth", .12, -.05),
    ("Financial health", "Current ratio", "currentRatio", 2, .9),
    ("Financial health", "Quick ratio", "quickRatio", 1.2, .6),
    ("Financial health", "Debt / Equity (%)", "debtToEquity", 40, 180),
    ("Analyst", "Analyst rating (1=Strong Buy, 5=Sell)", "recommendationMean", 2.0, 3.5),
    ("Risk", "Beta", "beta", .9, 1.8),
    ("Risk", "Short interest (% of float)", "shortPercentOfFloat", .02, .12),
    ("Risk", "Institutional ownership", "heldPercentInstitutions", .70, .20),
]
NEG_IS_BAD = {"trailingPE", "forwardPE", "pegRatio", "enterpriseToEbitda"}  # loss-making = bearish


CYCLICAL = {"Energy", "Basic Materials", "Industrials", "Consumer Cyclical"}
_FIN_SKIP = {"Debt / Equity (%)", "Current ratio", "Quick ratio", "Cash / Debt", "Net debt / EBITDA", "EV / EBITDA",
             "EV / Revenue", "Price / Sales", "Gross margin", "EBITDA margin", "Operating margin",
             "Free cash flow yield", "FCF / Net income", "Interest coverage", "Total debt change (YoY)",
             "Share of years with positive FCF", "Gross margin change (YoY)", "Operating margin change (YoY)",
             "Operating margin stability (std dev)"}
_DEF_GROWTH = {k: (.05, -.01) for k in ("Revenue growth (YoY)", "Revenue growth (latest fiscal yr)", "Revenue 3-yr CAGR")}
# Sector rules of thumb: "over" = (good, bad) thresholds replacing the generic ones, "skip" = metrics that don't
# mean much for the sector, "exit" = plausible long-run P/E range used in 5-year scenarios.
PROFILES = {
    "Technology": {"over": {"Trailing P/E": (25, 55), "Forward P/E": (22, 40), "Price / Sales": (5, 16), "Price / Book": (6, 20),
                            "EV / EBITDA": (18, 38), "EV / Revenue": (5, 14), "PEG ratio": (1.5, 3), "Gross margin": (.65, .40),
                            "Operating margin": (.25, .08), "Debt / Equity (%)": (40, 120)}, "skip": set(), "exit": (18, 32)},
    "Financial Services": {"over": {"Price / Book": (1.0, 2.5), "Trailing P/E": (10, 20), "Forward P/E": (10, 18),
                                    "Return on equity": (.13, .06), "Return on assets": (.012, .004), "Net margin": (.25, .10)},
                           "skip": _FIN_SKIP, "exit": (9, 16)},
    "Healthcare": {"over": {"Trailing P/E": (22, 45), "Forward P/E": (18, 32), "Gross margin": (.65, .35), "Operating margin": (.20, .02),
                            "Net margin": (.15, 0), "Price / Sales": (3, 10), "EV / EBITDA": (12, 25)}, "skip": set(), "exit": (16, 28)},
    "Consumer Cyclical": {"over": {"Trailing P/E": (15, 32), "Forward P/E": (14, 28), "Net margin": (.08, .02), "Operating margin": (.10, .03),
                                   "Gross margin": (.35, .15), "EV / EBITDA": (9, 18), "Price / Sales": (1, 4),
                                   "Debt / Equity (%)": (70, 200)}, "skip": set(), "exit": (12, 22)},
    "Consumer Defensive": {"over": {"Trailing P/E": (18, 30), "Forward P/E": (17, 26), "Net margin": (.07, .02), "Operating margin": (.12, .04),
                                    "Gross margin": (.35, .15), "EV / EBITDA": (11, 18), "Price / Sales": (1, 3),
                                    "Debt / Equity (%)": (80, 200), **_DEF_GROWTH}, "skip": set(), "exit": (16, 24)},
    "Industrials": {"over": {"Trailing P/E": (17, 32), "Forward P/E": (16, 28), "Net margin": (.09, .03), "Operating margin": (.14, .05),
                             "Gross margin": (.32, .15), "EV / EBITDA": (11, 20), "Price / Sales": (1.5, 4),
                             "Debt / Equity (%)": (70, 170)}, "skip": set(), "exit": (14, 22)},
    "Energy": {"over": {"Trailing P/E": (9, 20), "Forward P/E": (9, 18), "EV / EBITDA": (4.5, 9), "Price / Sales": (1, 3), "Net margin": (.10, .02),
                        "Operating margin": (.15, .03), "Gross margin": (.40, .15), "Debt / Equity (%)": (35, 100),
                        "Price / Book": (1.2, 3)}, "skip": set(), "exit": (8, 14)},
    "Utilities": {"over": {"Trailing P/E": (16, 25), "Forward P/E": (15, 22), "Price / Book": (1.6, 3), "EV / EBITDA": (10, 15),
                           "Debt / Equity (%)": (110, 260), "Net margin": (.12, .05), "Operating margin": (.20, .10),
                           "Return on equity": (.10, .05), "Current ratio": (.9, .5), "Quick ratio": (.6, .3),
                           "Interest coverage": (3.5, 1.8), "Net debt / EBITDA": (4.5, 7), **_DEF_GROWTH},
                  "skip": {"Free cash flow yield", "FCF / Net income", "Share of years with positive FCF", "Gross margin",
                           "Gross margin change (YoY)"}, "exit": (14, 20)},
    "Real Estate": {"over": {"Price / Book": (1.3, 3), "EV / EBITDA": (14, 25), "Debt / Equity (%)": (90, 220), "Net margin": (.20, .05),
                             "Return on equity": (.08, .02), "Interest coverage": (3, 1.5)},
                    "skip": {"Trailing P/E", "Forward P/E", "PEG ratio", "Free cash flow yield", "FCF / Net income", "Gross margin",
                             "Gross margin change (YoY)", "Current ratio", "Quick ratio"}, "exit": (14, 22)},
    "Basic Materials": {"over": {"Trailing P/E": (11, 25), "Forward P/E": (11, 22), "EV / EBITDA": (6, 12), "Price / Sales": (1.2, 3.5),
                                 "Net margin": (.10, .03), "Operating margin": (.15, .05), "Gross margin": (.30, .12),
                                 "Debt / Equity (%)": (35, 100)}, "skip": set(), "exit": (10, 17)},
    "Communication Services": {"over": {"Trailing P/E": (18, 35), "Forward P/E": (16, 28), "EV / EBITDA": (9, 18), "Operating margin": (.20, .08),
                                        "Net margin": (.12, .04), "Debt / Equity (%)": (70, 200), "Price / Sales": (2.5, 8)},
                               "skip": set(), "exit": (14, 26)},
}
DEFAULT = {"over": {}, "skip": set(), "exit": (15, 25)}


def lin(v, good, bad):
    return float(np.clip(2 * (v - bad) / (good - bad) - 1, -1, 1))


def row(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna().astype(float)
            if len(s):
                return s
    return None


def growth(s):
    if s is None or len(s) < 2 or s.iloc[1] <= 0:
        return None
    return s.iloc[0] / s.iloc[1] - 1


def build_metrics(info, hist, inc, cf, bs, prof=None):
    prof = prof or DEFAULT
    rows = []

    def add(cat, label, v, good=None, bad=None, neg_bad=False, score=None):
        if label in prof["skip"]:
            return
        if label in prof["over"] and score is None:
            good, bad = prof["over"][label]
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if np.isnan(v) or np.isinf(v):
            return
        if score is None:
            score = -1.0 if (neg_bad and v <= 0) else lin(v, good, bad)
        rows.append({"Category": cat, "Metric": label, "Value": v, "Score": score})

    # --- Fundamentals from the info table ---
    for cat, label, key, good, bad in INFO_METRICS:
        add(cat, label, info.get(key), good, bad, key in NEG_IS_BAD)

    mc, fcf_ttm = info.get("marketCap"), info.get("freeCashflow")
    if mc and fcf_ttm:
        add("Valuation", "Free cash flow yield", fcf_ttm / mc, .05, 0)
    cash, debt_i, ebitda = info.get("totalCash"), info.get("totalDebt"), info.get("ebitda")
    if cash is not None and debt_i:
        add("Financial health", "Cash / Debt", cash / debt_i, 1, .2)
    if debt_i is not None and cash is not None and ebitda and ebitda > 0:
        add("Financial health", "Net debt / EBITDA", (debt_i - cash) / ebitda, 1, 4)
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    for key, label, g, b in (("targetMeanPrice", "Upside to mean analyst target", .15, -.05),
                             ("targetMedianPrice", "Upside to median analyst target", .15, -.05)):
        if price and info.get(key):
            add("Analyst", label, info[key] / price - 1, g, b)

    fe, te = info.get("forwardEps"), info.get("trailingEps")
    if fe and te and te > 0 and fe > 0:
        add("Growth", "Forward EPS growth", fe / te - 1, .12, 0)

    # --- Statement-derived trends ---
    rev = row(inc, ["Total Revenue", "Operating Revenue"])
    ni = row(inc, ["Net Income"])
    oi = row(inc, ["Operating Income"])
    gp = row(inc, ["Gross Profit"])
    ebit = row(inc, ["EBIT"])
    intx = row(inc, ["Interest Expense"])
    fcf = row(cf, ["Free Cash Flow"])
    sh = row(bs, ["Ordinary Shares Number", "Share Issued"])
    debt = row(bs, ["Total Debt"])

    add("Growth", "Revenue growth (latest fiscal yr)", growth(rev), .12, 0)
    if rev is not None and len(rev) >= 4 and rev.iloc[3] > 0 and rev.iloc[0] > 0:
        add("Growth", "Revenue 3-yr CAGR", (rev.iloc[0] / rev.iloc[3]) ** (1 / 3) - 1, .12, 0)
    add("Growth", "Net income growth (latest fiscal yr)", growth(ni), .12, 0)
    add("Growth", "Operating income growth", growth(oi), .12, 0)
    add("Growth", "Free cash flow growth", growth(fcf), .12, 0)
    if rev is not None and len(rev) >= 2:
        if gp is not None and len(gp) >= 2:
            add("Profitability", "Gross margin change (YoY)", gp.iloc[0] / rev.iloc[0] - gp.iloc[1] / rev.iloc[1], .01, -.02)
        if oi is not None and len(oi) >= 2:
            add("Profitability", "Operating margin change (YoY)", oi.iloc[0] / rev.iloc[0] - oi.iloc[1] / rev.iloc[1], .01, -.02)
    if fcf is not None and ni is not None and ni.iloc[0] > 0:
        add("Profitability", "FCF / Net income", fcf.iloc[0] / ni.iloc[0], 1, .5)
    if sh is not None and len(sh) >= 2:
        add("Financial health", "Share count change (buybacks < 0)", sh.iloc[0] / sh.iloc[1] - 1, -.01, .03)
    if debt is not None and len(debt) >= 2 and debt.iloc[1] > 0:
        add("Financial health", "Total debt change (YoY)", debt.iloc[0] / debt.iloc[1] - 1, -.05, .15)
    if ebit is not None and intx is not None and abs(intx.iloc[0]) > 0:
        add("Financial health", "Interest coverage", ebit.iloc[0] / abs(intx.iloc[0]), 8, 2)

    # --- Long-term track record from statements ---
    if rev is not None and len(rev) >= 3:
        add("Track record", "Share of years with revenue growth", (rev.diff(-1).dropna() > 0).mean(), 1, .5)
    if ni is not None and len(ni) >= 3:
        add("Track record", "Share of years profitable", (ni > 0).mean(), 1, .5)
    if fcf is not None and len(fcf) >= 3:
        add("Track record", "Share of years with positive FCF", (fcf > 0).mean(), 1, .5)
    if oi is not None and rev is not None:
        m = (oi / rev).dropna()
        if len(m) >= 3:
            add("Track record", "Operating margin stability (std dev)", m.std(), .02, .08)

    # --- Technicals and risk from price history ---
    c = hist["Close"].dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)
    if len(c) > 60:
        last = c.iloc[-1]
        r = c.pct_change().dropna()
        for n in (20, 50, 100, 200):
            if len(c) > n:
                add("Technicals", f"Price vs {n}-day average", last / c.rolling(n).mean().iloc[-1] - 1, .05, -.05)
        if len(c) > 200:
            add("Technicals", "50-day vs 200-day average",
                c.rolling(50).mean().iloc[-1] / c.rolling(200).mean().iloc[-1] - 1, .03, -.03)
        for n, lab, g, b in ((21, "1-month", .05, -.05), (63, "3-month", .10, -.10),
                             (126, "6-month", .15, -.10), (252, "12-month", .25, -.10)):
            if len(c) > n:
                add("Technicals", f"{lab} return", last / c.iloc[-n - 1] - 1, g, b)
        d = c.diff()
        up, dn = d.clip(lower=0).rolling(14).mean().iloc[-1], (-d.clip(upper=0)).rolling(14).mean().iloc[-1]
        if dn and dn > 0:
            rsi = 100 - 100 / (1 + up / dn)
            add("Technicals", "RSI (14)", rsi, score=1.0 if rsi < 30 else (-1.0 if rsi > 75 else 0.0))
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        add("Technicals", "MACD histogram / price", (macd - macd.ewm(span=9).mean()).iloc[-1] / last, .01, -.01)
        yr = c.tail(252)
        add("Technicals", "Distance from 52-week high", last / yr.max() - 1, -.05, -.30)
        vol = r.tail(252).std() * np.sqrt(252)
        add("Risk", "Annualized volatility", vol, .25, .60)
        add("Risk", "Max drawdown (1 yr)", (yr / yr.cummax() - 1).min(), -.15, -.45)
        if vol > 0:
            add("Risk", "1-yr return / volatility", r.tail(252).mean() * 252 / vol, 1, 0)
    if len(c) > 756:  # 3+ years of history
        for yrs in (3, 5, 10):
            n = 252 * yrs
            if len(c) > n:
                add("Track record", f"{yrs}-yr annualized price return", (c.iloc[-1] / c.iloc[-n - 1]) ** (1 / yrs) - 1, .15, .03)
        for yrs in (5, 10):
            n = 252 * yrs
            if len(c) > n:
                win = c.tail(n)
                add("Track record", f"Max drawdown ({yrs} yr)", (win / win.cummax() - 1).min(), -.30, -.60)
        yearly = c.groupby(c.index.year).last().pct_change().dropna()
        if len(yearly) >= 3:
            add("Track record", "Share of calendar years up", (yearly > 0).mean(), .8, .5)
    return pd.DataFrame(rows)


def composite(df, weights):
    cs = df.groupby("Category")["Score"].mean()
    w = pd.Series(weights).reindex(cs.index).fillna(0)
    if w.sum() == 0:
        w[:] = 1
    return cs, float((cs * w).sum() / w.sum())


def label(final):
    if final > .35: return "STRONG BUY"
    if final > .15: return "BUY"
    if final > -.15: return "HOLD"
    if final > -.35: return "SELL"
    return "STRONG SELL"


def analyze(df, price, vol, info):
    """Short-term (about 12 months)."""
    weights = HORIZONS["Short-term"]
    cat_scores, comp = composite(df, weights)

    tm, th, tl = info.get("targetMeanPrice"), info.get("targetHighPrice"), info.get("targetLowPrice")
    base = tm if tm else price * (1 + .10 * comp)
    bull = min(th, price * (1 + 1.5 * vol)) if th else price * (1 + vol)
    bear = max(tl, price * (1 - 1.5 * vol)) if tl else price * (1 - vol)
    bull, bear = max(bull, base), min(bear, base)

    p_bull, p_bear = .25 + .25 * comp, .25 - .25 * comp
    p_base = 1 - p_bull - p_bear
    ev = p_bull * bull + p_base * base + p_bear * bear
    exp_ret = ev / price - 1

    final = .7 * comp + .3 * float(np.clip(exp_ret / .20, -1, 1))
    return dict(cat_scores=cat_scores, comp=comp, final=final, rating=label(final), exp_ret=exp_ret, weights=weights,
                scen=[("🐂 Bull", bull, p_bull), ("⚖️ Base", base, p_base), ("🐻 Bear", bear, p_bear)])


def analyze_long(df, price, info, prof=None):
    prof = prof or DEFAULT
    """Long-term (3-5 years): business quality + 5-year growth/valuation scenarios (price only, no dividends)."""
    weights = HORIZONS["Long-term"]
    cat_scores, comp = composite(df, weights)

    def val(name):
        m = df.loc[df.Metric == name, "Value"]
        return float(m.iloc[0]) if len(m) else None
    gs = [val(n) for n in ("Revenue 3-yr CAGR", "Revenue growth (YoY)", "Forward EPS growth", "Earnings growth (YoY)")]
    gs = [float(np.clip(g, -.10, .30)) for g in gs if g is not None]
    g0 = float(np.clip(np.median(gs), -.05, .18)) if gs else .05
    pe = info.get("forwardPE") or info.get("trailingPE")
    pe = float(pe) if pe and pe > 0 and "Trailing P/E" not in prof["skip"] else None
    lo, hi = prof["exit"]

    def proj(g, lo, hi):
        mult = float(np.clip(pe, lo, hi)) / pe if pe else 1.0
        return price * (1 + g) ** 5 * mult
    base = proj(g0, lo, hi)
    bull = proj(min(g0 + .05, .30), lo + 3, hi + 7)
    bear = price * (1 + max(g0 - .08, -.10)) ** 5 * (.65 if pe else 1.0)
    bull, bear = max(bull, base), min(bear, base)

    p_bull, p_bear = .25 + .25 * comp, .25 - .25 * comp
    p_base = 1 - p_bull - p_bear
    ev = p_bull * bull + p_base * base + p_bear * bear
    ann = lambda p: (p / price) ** (1 / 5) - 1
    exp_ret = ann(ev)
    final = .7 * comp + .3 * float(np.clip((exp_ret - .08) / .08, -1, 1))  # vs. an ~8%/yr market benchmark
    note = (f"5-year price targets. Base case: ~{g0:.0%}/yr growth with the P/E settling toward {lo}-{hi}x (typical for this sector). "
            "Bull: +5 pts growth and a richer multiple. Bear: -8 pts growth and a 35% lower multiple. "
            "Rating compares expected annual return with an 8%/yr market benchmark.")
    return dict(cat_scores=cat_scores, comp=comp, final=final, rating=label(final), exp_ret=exp_ret, weights=weights,
                scen=[("🐂 Bull", bull, p_bull, ann(bull)), ("⚖️ Base", base, p_base, ann(base)),
                      ("🐻 Bear", bear, p_bear, ann(bear))], note=note)


def thesis(df, info, short, long_, sector):
    """Reads the metrics together like an analyst: archetype, interactions between signals, adjustments."""
    cs = df.groupby("Category")["Score"].mean()
    g = lambda k: float(cs.get(k, 0.0))
    V, P, G, H, T, A, K = (g(k) for k in ("Valuation", "Profitability", "Growth", "Financial health",
                                          "Technicals", "Analyst", "Track record"))

    def mv(name):
        r = df.loc[df.Metric == name, "Value"]
        return float(r.iloc[0]) if len(r) else 0.0
    if P < -.2 and H < -.2: arch = "Turnaround / financially stretched"
    elif P > .35 and G > .2 and K > .2: arch = "Quality compounder"
    elif G > .4: arch = "Growth story"
    elif V > .35 and P < .3: arch = "Value / out-of-favor"
    elif sector in {"Utilities", "Real Estate", "Consumer Defensive"}: arch = "Defensive / income"
    elif sector in CYCLICAL: arch = "Cyclical"
    else: arch = "Balanced, no standout"

    # (condition, observation, long-term adj, short-term adj, what would change my mind)
    rules = [
        (V < -.3 and G > .4 and P > .3, "The valuation is rich, but growth and profitability are strong enough that the premium is defensible.", .05, 0, "Growth slowing would undercut the case for paying up."),
        (V < -.3 and G < .1, "I'd be paying a high price for little growth, a combination that tends to disappoint.", -.10, -.05, "A lower price, or growth reaccelerating."),
        (V > .3 and (G < -.2 or P < -.1), "It looks cheap, but shrinking growth or weak margins suggest it may be cheap for a reason (value-trap risk).", -.10, 0, "Evidence that sales and margins have stopped falling."),
        (V > .3 and P > .2 and G > -.1 and H > 0, "A reasonable price for a healthy, profitable business is the mismatch I look for.", .10, .05, "Margins slipping would remove the margin of safety."),
        (T > .4 and (P < 0 or G < 0), "The stock is rising faster than the business; momentum without fundamental support tends to fade.", 0, -.08, "Earnings or margins catching up with the price."),
        (T < -.3 and P > .3 and G > .2, "A strong business whose stock is out of favor; weak price action alone is not a reason to sell.", .05, 0, "The price stabilizing above its 200-day average would improve timing."),
        (H < -.3, "The balance sheet is the main risk: leverage magnifies any slowdown, so the bear case deserves extra weight.", -.10, -.05, "Falling debt or rising interest coverage."),
        (A > .5 and (P + G) / 2 < -.1, "Analysts are more optimistic than the underlying numbers justify, so I trust the numbers.", -.05, -.05, "Actual earnings beating estimates."),
        (mv("Net margin") < 0 and G > .3 and mv("Cash / Debt") > 1, "Unprofitable but growing and cash-funded; the question is whether it reaches profitability before the cash runs out.", 0, 0, "A visible path to positive free cash flow."),
        (mv("Operating margin change (YoY)") > .01, "Operating margins are expanding, a sign of operating leverage.", .03, .02, "Margin gains reversing."),
        (mv("Operating margin change (YoY)") < -.02, "Operating margins are compressing, which usually precedes weaker earnings.", -.05, -.03, "Margins stabilizing."),
        (sector in CYCLICAL, "Earnings in this sector swing with the cycle, so a low P/E can signal peak earnings, not a bargain.", 0, 0, "Where we are in the sector's cycle."),
    ]
    notes, watch, dl, ds = [], [], 0.0, 0.0
    for cond, text, a, b, w in rules:
        if cond:
            notes.append(text + (f" (long {a:+.2f}, short {b:+.2f})" if a or b else ""))
            watch.append(w); dl += a; ds += b
    long_["final"] = float(np.clip(long_["final"] + dl, -1, 1)); long_["rating"] = label(long_["final"])
    short["final"] = float(np.clip(short["final"] + ds, -1, 1)); short["rating"] = label(short["final"])

    s, l = short["final"], long_["final"]
    if l > .15 and s > .15: rel = "Both the business and the near-term setup support owning it."
    elif l > .15 and s <= .15: rel = "A good business with a weak near-term setup. If buying, scale in rather than all at once."
    elif s > .15 and l <= .15: rel = "More of a trade than a holding: the setup is favorable, but I wouldn't count on the business over years."
    elif s < -.15 and l < -.15: rel = "Neither the business nor the setup appeals to me."
    else: rel = "The signals are mixed, so patience is reasonable."
    verdict = (f"**{info.get('shortName', '')}** reads as a **{arch.lower()}**"
               f"{' in ' + sector if sector else ''}. Short-term: **{short['rating']}**. Long-term: **{long_['rating']}**. {rel}")

    spread = float(cs.std()) if len(cs) > 1 else 0.0
    conf = "High" if len(df) >= 35 and spread < .45 else ("Low" if len(df) < 22 or spread > .7 else "Medium")
    why = f"Confidence {conf.lower()}: {len(df)} metrics scored; categories {'mostly agree' if spread < .45 else 'disagree notably'}."
    strong = [f"{k} ({v:+.2f})" for k, v in cs.sort_values(ascending=False).items() if v > .25][:3]
    weak = [f"{k} ({v:+.2f})" for k, v in cs.sort_values().items() if v < -.15][:3]
    return dict(arch=arch, verdict=verdict, notes=notes, watch=watch, conf=conf, why=why, strong=strong, weak=weak)


@st.cache_data(ttl=3600, show_spinner=False)
def load(sym):
    t = yf.Ticker(sym)

    def safe(f):
        try:
            return f()
        except Exception:
            return None
    hist = safe(lambda: t.history(period="10y", auto_adjust=True))
    return (safe(lambda: t.info) or {}, hist, safe(lambda: t.income_stmt),
            safe(lambda: t.cashflow), safe(lambda: t.balance_sheet))


def fmt(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def render(res, df, price):
    st.write(f"Score: **{res['final']:+.2f}**  (-1 very bearish, +1 very bullish)")
    st.subheader("Scenarios")
    rows = []
    for n, p, pr, *ann in res["scen"]:
        r = {"Case": n, "Target": f"${p:,.2f}", "vs. now": f"{p / price - 1:+.1%}"}
        if ann:
            r["Per year"] = f"{ann[0]:+.1%}"
        r["Probability"] = f"{pr:.0%}"
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.write(f"Probability-weighted expected {'annual return' if len(res['scen'][0]) > 3 else 'move'}: "
             f"**{res['exp_ret']:+.1%}**")
    if res.get("note"):
        st.caption(res["note"])
    used = [k for k, v in res["weights"].items() if v > 0]
    st.subheader("Category scores")
    st.bar_chart(res["cat_scores"].reindex(used).dropna().round(2))
    d = df[df.Category.isin(used)]
    st.subheader("🐂 Bull case: what's working")
    for _, m in d[d.Score > .3].sort_values("Score", ascending=False).head(6).iterrows():
        st.write(f"🟢 {m.Metric}: {fmt(m.Value)}")
    st.subheader("🐻 Bear case: what's worrying")
    for _, m in d[d.Score < -.3].sort_values("Score").head(6).iterrows():
        st.write(f"🔴 {m.Metric}: {fmt(m.Value)}")


def main():
    st.set_page_config(page_title="Stock Rater", page_icon="📈", layout="centered")
    st.title("📈 Stock Rater")
    st.caption("Type a ticker. Sector-adjusted short- and long-term ratings, with a written thesis.")

    with st.form("f"):
        sym_in = st.text_input("Ticker", placeholder="e.g. AAPL").strip().upper()
        go = st.form_submit_button("Analyze", use_container_width=True)
    if go and sym_in:
        st.session_state["sym"] = sym_in
    sym = st.session_state.get("sym")
    if not sym:
        return

    with st.spinner(f"Crunching {sym}..."):
        info, hist, inc, cf, bs = load(sym)
        sector = info.get("sector")
        prof = PROFILES.get(sector, DEFAULT)
        df = build_metrics(info, hist, inc, cf, bs, prof)
    if df.empty or hist is None or hist.empty:
        st.error(f"Couldn't get enough data for '{sym}'. Check the ticker and try again.")
        return

    price = info.get("currentPrice") or info.get("regularMarketPrice") or float(hist["Close"].iloc[-1])
    vol_row = df.loc[df.Metric == "Annualized volatility", "Value"]
    vol = float(vol_row.iloc[0]) if len(vol_row) else .35
    short, long_ = analyze(df, price, vol, info), analyze_long(df, price, info, prof)
    th = thesis(df, info, short, long_, sector)

    st.header(f"{info.get('shortName', sym)} ({sym})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Short-term (≤12 mo)", short["rating"])
    c2.metric("Long-term (3-5 yr)", long_["rating"])
    c3.metric("Price", f"${price:,.2f}")

    st.caption(f"Sector: {sector or 'n/a (generic thresholds)'} · Profile: {th['arch']} · Confidence: {th['conf']}")
    st.subheader("🧠 The thesis")
    st.write(th["verdict"])
    if th["strong"]: st.write("**Strongest areas:** " + ", ".join(th["strong"]))
    if th["weak"]: st.write("**Weakest areas:** " + ", ".join(th["weak"]))
    for n in th["notes"]:
        st.write(f"• {n}")
    if th["watch"]:
        st.write("**What would change my mind:** " + " ".join(th["watch"][:3]))
    st.caption(th["why"] + " Thresholds are sector rules of thumb, not live peer medians.")

    t1, t2 = st.tabs(["⏱ Short-term", "🌱 Long-term"])
    with t1:
        render(short, df, price)
    with t2:
        render(long_, df, price)

    with st.expander(f"All {len(df)} metrics evaluated"):
        show = df.assign(Signal=df.Score.map(lambda s: "🟢" if s > .3 else ("🔴" if s < -.3 else "🟡")))
        show = show[["Signal", "Category", "Metric", "Value", "Score"]].copy()
        show["Value"] = show["Value"].map(fmt)
        show["Score"] = show["Score"].round(2)
        st.dataframe(show, hide_index=True, use_container_width=True)

    st.caption("Educational tool, not financial advice. Based on free, sometimes incomplete Yahoo Finance "
               "data and simple rules of thumb. Metrics unavailable for a ticker (e.g. ETFs) are skipped.")


main()
