import argparse
import requests
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from dataclasses import dataclass

# =========================
# API
# =========================
API_YEAR = "https://history.macaumarksix.com/history/macaujc2/y/{year}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# =========================
# 数据结构
# =========================
@dataclass
class Draw:
    nums: list
    main6: list
    special: int

# =========================
# 数据加载
# =========================
def fetch(year):
    r = requests.get(API_YEAR.format(year=year), headers=HEADERS)
    return r.json()["data"]

def parse(data):
    out = []
    for i in data:
        nums = list(map(int, i["openCode"].split(",")))
        if len(nums) != 7:
            continue
        out.append(Draw(nums, nums[:6], nums[-1]))
    return out[::-1]

# =========================
# ===== AI核心层1：序列学习（Markov增强）
# =========================
def sequence_model(draws):

    markov1 = defaultdict(float)
    markov2 = defaultdict(float)

    for i in range(2, len(draws)):
        prev1 = set(draws[i-1].nums)
        prev2 = set(draws[i-2].nums)
        cur = set(draws[i].nums)

        for p in prev1:
            for c in cur:
                markov1[c] += 1

        for p in prev2:
            for c in cur:
                markov2[c] += 1

    return markov1, markov2

# =========================
# ===== AI核心层2：贝叶斯概率
# =========================
def bayes(draws):

    c = Counter()
    for d in draws:
        c.update(d.nums)

    total = sum(c.values()) + 49

    prob = {}
    for i in range(1,50):
        prob[i] = (c[i] + 1) / total

    return prob

# =========================
# ===== AI核心层3：蒙特卡洛模拟
# =========================
def monte_carlo(draws, runs=2000):

    pool = list(range(1,50))
    score = Counter()

    for _ in range(runs):
        sample = np.random.choice(pool, 7, replace=False)
        for n in sample:
            score[n] += 1

    total = sum(score.values())
    return {k: v/total for k,v in score.items()}

# =========================
# ===== AI核心层4：遗漏模型
# =========================
def gap(draws):

    res = {}

    for n in range(1,50):
        m = 0
        for d in draws:
            if n in d.nums:
                break
            m += 1
        res[n] = m

    return res

# =========================
# ===== AI融合权重（自适应）
# =========================
def adaptive_fusion(draws):

    markov1, markov2 = sequence_model(draws)
    bayes_p = bayes(draws)
    mc = monte_carlo(draws)
    g = gap(draws)

    scores = {}

    for n in range(1,50):

        scores[n] = (
            bayes_p[n] * 2.0 +
            mc.get(n,0) * 2.5 +
            markov1[n] * 0.8 +
            markov2[n] * 1.2 +
            (1 / (g[n]+1)) * 1.5
        )

    return scores

# =========================
# 推荐系统
# =========================
def recommend(scores):

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    top = [x[0] for x in ranked]

    return {
        "main6": sorted(top[:6]),
        "special": top[0],
        "top10": top[:10],
        "top20": ranked[:20]
    }

# =========================
# 回测系统
# =========================
def backtest(draws, window=30):

    hit2 = hit3 = total = 0

    for i in range(window, len(draws)-1):

        train = draws[:i]
        test = draws[i]

        scores = adaptive_fusion(train)
        top = set(sorted(scores, key=scores.get, reverse=True)[:12])

        hit = len(top & set(test.main6))

        if hit >= 2:
            hit2 += 1
        if hit >= 3:
            hit3 += 1

        total += 1

    return hit2, hit3, total

# =========================
# CLI
# =========================
def run_cli(year, window):

    draws = parse(fetch(year))

    scores = adaptive_fusion(draws)
    rec = recommend(scores)

    print("\n=== TOP20 ===")
    print(rec["top20"])

    print("\n=== 正码6 ===")
    print(rec["main6"])

    print("\n=== 特码 ===")
    print(rec["special"])

    h2,h3,t = backtest(draws, window)

    print("\n=== 回测 ===")
    print("2中2:", h2, "/", t)
    print("3中3:", h3, "/", t)

# =========================
# WEB
# =========================
def run_web(year, window):

    st.title("🔥 Ultimate V6 AI（序列 + 贝叶斯 + 蒙特卡洛融合）")

    draws = parse(fetch(year))

    scores = adaptive_fusion(draws)
    rec = recommend(scores)

    st.subheader("正码6")
    st.write(rec["main6"])

    st.subheader("特码")
    st.write(rec["special"])

    st.subheader("TOP10")
    st.write(rec["top10"])

    df = pd.DataFrame(rec["top20"], columns=["号码","评分"])
    st.dataframe(df)

    h2,h3,t = backtest(draws, window)

    st.subheader("回测")
    st.write({
        "2中2": f"{h2}/{t}",
        "3中3": f"{h3}/{t}"
    })

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--web", action="store_true")
    args = parser.parse_args()

    if args.web:
        run_web(args.year, args.window)
    else:
        run_cli(args.year, args.window)