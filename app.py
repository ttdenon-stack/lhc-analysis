import math
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from itertools import combinations
import requests
from datetime import datetime

# =========================
# 基础工具（你必须保留）
# =========================

RED = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
BLUE = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}

ZODIAC_MAP = {
    "鼠":[7,19,31,43], "牛":[6,18,30,42], "虎":[5,17,29,41],
    "兔":[4,16,28,40], "龙":[3,15,27,39], "蛇":[2,14,26,38],
    "马":[1,13,25,37,49], "羊":[12,24,36,48], "猴":[11,23,35,47],
    "鸡":[10,22,34,46], "狗":[9,21,33,45], "猪":[8,20,32,44]
}

def get_wave(n):
    if n in RED:
        return "红"
    if n in BLUE:
        return "蓝"
    return "绿"

def get_zodiac(n):
    for z, nums in ZODIAC_MAP.items():
        if n in nums:
            return z
    return "未知"

# =========================
# 数据获取（示例接口）
# =========================
LATEST_API = "https://macaumarksix.com/api/macaujc2.com"

@st.cache_data(ttl=300)
def fetch_latest():
    try:
        r = requests.get(LATEST_API, timeout=10).json()
        return r if isinstance(r, list) else [r]
    except:
        return []

# =========================
# 数据结构解析
# =========================
def parse_history(raw):
    rows = []
    for item in raw:
        try:
            nums = list(map(int, item["openCode"].split(",")))
            if len(nums) == 7:
                rows.append({
                    "expect": item["expect"],
                    "nums": nums,
                    "normal": nums[:6],
                    "special": nums[-1],
                    "time": item.get("openTime", "")
                })
        except:
            pass
    return pd.DataFrame(rows).sort_values("expect", ascending=False)

# =========================
# 🔥 V9 Pro 核心回测系统
# =========================
def ai_engine_v9_pro(df, analyze_periods=30, backtest_depth=10):

    def core(data):
        normal = defaultdict(float)
        special = defaultdict(float)
        global_s = defaultdict(float)
        miss = defaultdict(float)
        zodiac = Counter()

        for i, r in data.iterrows():
            decay = math.exp(-(i / (len(data) * 0.4)))

            for n in r["normal"]:
                normal[n] += 3.2 * decay
                global_s[n] += 2.8 * decay
                zodiac[get_zodiac(n)] += 1

            sp = r["special"]
            special[sp] += 4.8 * decay
            global_s[sp] += 3.2 * decay
            zodiac[get_zodiac(sp)] += 1

        for n in range(1, 50):
            m = 0
            for r in data["nums"]:
                if n in r:
                    break
                m += 1
            miss[n] = min(m, 25) * 0.38

        short = data.head(min(len(data), 5))
        flat = []
        for _, r in short.iterrows():
            flat += r["normal"]

        hot = Counter(flat)
        kill = []

        for n, c in hot.items():
            if c >= 3:
                normal[n] -= 6.5
                global_s[n] -= 4.5
                kill.append(n)
            elif 1 <= c <= 2:
                normal[n] += 2.2
                global_s[n] += 1.6

        final = defaultdict(float)
        weak = min(zodiac, key=zodiac.get) if zodiac else None

        for n in range(1, 50):
            bonus = 0.45 if get_zodiac(n) == weak else 0
            final[n] = max(0.0001, normal[n] + miss[n] + bonus)

        return final, special, global_s, kill

    # =========================
    # 当前预测
    # =========================
    latest = df.head(analyze_periods)
    fn, sp, gl, kill = core(latest)

    rank = sorted(fn.items(), key=lambda x: -x[1])
    top = [x[0] for x in rank if x[0] not in kill][:8]

    special = [x[0] for x in sorted(sp.items(), key=lambda x: -x[1])[:5]]
    global_top = [x[0] for x in sorted(gl.items(), key=lambda x: -x[1])[:10]]

    danma = []
    used = set()
    for n in global_top:
        w = get_wave(n)
        if w not in used:
            danma.append(n)
            used.add(w)
        if len(danma) >= 4:
            break

    # =========================
    # 回测
    # =========================
    hits = []
    max_test = min(backtest_depth, len(df) - analyze_periods)

    for i in range(max_test):
        train = df.iloc[i+1:i+1+analyze_periods]
        test = set(df.iloc[i]["nums"])

        fn2, sp2, gl2, kill2 = core(train)
        pred = set([x[0] for x in sorted(fn2.items(), key=lambda x: -x[1]) if x[0] not in kill2][:10])

        hits.append(len(pred & test))

    avg = sum(hits)/len(hits) if hits else 0
    confidence = min(avg / 3 * 100, 100)

    # entropy
    total = sum(fn.values())
    entropy = -sum((v/total)*math.log(v/total) for v in fn.values())

    return {
        "top_normals": top,
        "special": special,
        "global": global_top,
        "danma": danma,
        "kill": kill[:6],
        "recent_avg_hits": round(avg,2),
        "confidence": round(confidence,1),
        "entropy": round(entropy,4),
        "combo2": list(combinations(top,2))[:6],
        "combo3": list(combinations(top,3))[:6]
    }

# =========================
# UI
# =========================
st.title("V9 Pro 回测系统")

data = fetch_latest()

if not data:
    st.error("API失败")
    st.stop()

df = parse_history(data)

result = ai_engine_v9_pro(df)

st.subheader("🔥 推荐号码")
st.success(" / ".join(map(lambda x: f"{x:02d}", result["top_normals"])))

st.subheader("🔥 置信度")
st.write(result["confidence"], "%")

if result["confidence"] >= 80:
    st.success("模型状态强")
elif result["confidence"] >= 50:
    st.warning("模型稳定")
else:
    st.error("模型混乱")

st.subheader("📊 回测")
st.write("平均命中:", result["recent_avg_hits"])

st.subheader("⚡ 组合")
for c in result["combo2"]:
    st.code(" - ".join(map(lambda x: f"{x:02d}", c)))