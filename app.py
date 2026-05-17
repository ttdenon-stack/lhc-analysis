import json
import os
import requests
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from datetime import datetime
import time
import random
import math
from itertools import combinations

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="澳门六合彩 AI 超级智能分析系统",
    layout="wide"
)

# ==========================================
# 自动刷新
# ==========================================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# ==========================================
# API
# ==========================================
LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"

AI_LEARN_FILE = "ai_learn.json"

# ==========================================
# 波色
# ==========================================
RED = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
BLUE = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
GREEN = {5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49}

# ==========================================
# 生肖
# ==========================================
ZODIAC_MAP = {
    "鼠":[7,19,31,43],
    "牛":[6,18,30,42],
    "虎":[5,17,29,41],
    "兔":[4,16,28,40],
    "龙":[3,15,27,39],
    "蛇":[2,14,26,38],
    "马":[1,13,25,37,49],
    "羊":[12,24,36,48],
    "猴":[11,23,35,47],
    "鸡":[10,22,34,46],
    "狗":[9,21,33,45],
    "猪":[8,20,32,44]
}

# ==========================================
# 工具函数
# ==========================================
def get_wave(num):
    if num in RED:
        return "红"
    if num in BLUE:
        return "蓝"
    return "绿"

def get_zodiac(num):
    for z, nums in ZODIAC_MAP.items():
        if num in nums:
            return z
    return "未知"

def get_tail(num):
    return num % 10

def get_zone(num):
    if num <= 16:
        return "低区"
    elif num <= 33:
        return "中区"
    return "高区"

# ==========================================
# AI权重
# ==========================================
def load_ai_weights():

    default_weights = {
        "miss": 2.0,
        "hot": 1.5,
        "cold": 2.5,
        "wave": 2.0,
        "zodiac": 2.0,
        "tail": 2.0,
        "combo": 2.5,
        "zone": 2.0,
        "special": 5.0,
        "markov": 3.0
    }

    if not os.path.exists(AI_LEARN_FILE):
        with open(AI_LEARN_FILE, "w", encoding="utf-8") as f:
            json.dump(default_weights, f, ensure_ascii=False, indent=4)
        return default_weights

    try:
        with open(AI_LEARN_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        for k, v in default_weights.items():
            if k not in data:
                data[k] = v

        return data

    except:
        return default_weights

def save_ai_weights(weights):
    with open(AI_LEARN_FILE, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=4)

# ==========================================
# API
# ==========================================
@st.cache_data(ttl=60)
def fetch_latest():
    try:
        r = requests.get(LATEST_API, timeout=10)
        data = r.json()
        return data if isinstance(data, list) else [data]
    except:
        return []

@st.cache_data(ttl=300)
def fetch_history(year):
    try:
        url = HISTORY_API.format(year)
        r = requests.get(url, timeout=20)
        data = r.json()
        return data.get("data", [])
    except:
        return []

# ==========================================
# 解析
# ==========================================
def parse_history(data):

    rows = []

    for item in data:
        try:
            nums = [int(x) for x in item["openCode"].split(",")]

            if len(nums) != 7:
                continue

            rows.append({
                "expect": item["expect"],
                "time": item["openTime"],
                "nums": nums,
                "normal": nums[:6],
                "special": nums[-1]
            })

        except:
            pass

    return pd.DataFrame(rows)

# ==========================================
# 马尔可夫
# ==========================================
def build_markov_chain(df, history_count=20):

    transitions = defaultdict(int)

    recent = df.head(history_count)

    prev = None

    for _, row in recent.iterrows():

        current = get_wave(row["special"])

        if prev:
            transitions[(prev, current)] += 1

        prev = current

    return transitions

# ==========================================
# AI核心
# ==========================================
def ai_engine(df, history_count=10):

    weights = load_ai_weights()

    recent = df.head(history_count)

    score = defaultdict(float)

    freq = Counter()
    zodiac_count = Counter()
    wave_count = Counter()
    tail_count = Counter()
    zone_count = Counter()

    total = max(len(recent), 1)

    # 主模型
    for idx, row in recent.iterrows():

        decay = math.exp(-(idx / total))

        normals = row["normal"]
        special = row["special"]

        for n in normals:

            freq[n] += 1

            score[n] += 3.5 * decay

            zodiac_count[get_zodiac(n)] += 1
            wave_count[get_wave(n)] += 1
            tail_count[get_tail(n)] += 1
            zone_count[get_zone(n)] += 1

        score[special] += weights["special"] * decay

    # 遗漏
    all_nums = df["nums"].tolist()

    for n in range(1, 50):

        miss = 0

        for row in all_nums:
            if n in row:
                break
            miss += 1

        miss = min(miss, 18)

        score[n] += miss * weights["miss"]

    # 热号降温
    for n, c in freq.items():
        if c >= 5:
            score[n] -= min(c * weights["hot"], 10)

    # 冷号增强
    for n in range(1, 50):
        if freq[n] == 0:
            score[n] += weights["cold"]

    # 波色轮动
    if wave_count:

        weak_wave = min(wave_count, key=wave_count.get)

        for n in range(1, 50):
            if get_wave(n) == weak_wave:
                score[n] += weights["wave"]

    # 生肖轮动
    if zodiac_count:

        weak_zodiac = min(zodiac_count, key=zodiac_count.get)

        for n in range(1, 50):
            if get_zodiac(n) == weak_zodiac:
                score[n] += weights["zodiac"]

    # 尾数轮动
    if tail_count:

        weak_tail = min(tail_count, key=tail_count.get)

        for n in range(1, 50):
            if get_tail(n) == weak_tail:
                score[n] += weights["tail"]

    # 区间轮动
    if zone_count:

        weak_zone = min(zone_count, key=zone_count.get)

        for n in range(1, 50):
            if get_zone(n) == weak_zone:
                score[n] += weights["zone"]

    # 连码
    for row in recent["normal"]:

        nums = sorted(row)

        for i in range(len(nums)-1):

            diff = nums[i+1] - nums[i]

            if diff == 1:
                score[nums[i]] += 2.5
                score[nums[i+1]] += 2.5

    # 杀号
    latest_nums = recent.iloc[0]["nums"]

    kill_numbers = []

    for n in range(1, 50):

        penalty = 0

        if n in latest_nums:
            penalty += 6

        if freq[n] >= 4:
            penalty += 4

        score[n] -= penalty

        if penalty >= 6:
            kill_numbers.append(n)

    # 马尔可夫
    try:

        transitions = build_markov_chain(df)

        last_wave = get_wave(recent.iloc[0]["special"])

        next_wave = defaultdict(int)

        for (a, b), v in transitions.items():
            if a == last_wave:
                next_wave[b] += v

        if next_wave:

            target_wave = max(next_wave, key=next_wave.get)

            for n in range(1, 50):
                if get_wave(n) == target_wave:
                    score[n] += weights["markov"]

    except:
        pass

    # 防负数
    for n in range(1, 50):
        score[n] = max(score[n], 0.1)

    # 概率
    total_score = sum(score.values())

    prob = {}

    for n in range(1, 50):
        prob[n] = round(score[n] / total_score * 100, 2)

    # 排序
    final_rank = sorted(
        score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_numbers = [x[0] for x in final_rank[:12]]

    combo2 = list(combinations(top_numbers[:8], 2))
    combo3 = list(combinations(top_numbers[:8], 3))

    zodiac_hot = [
        z[0]
        for z in zodiac_count.most_common(3)
    ]

    # 平特一肖
    one_zodiac = zodiac_count.most_common(1)[0][0]

    # 平特二肖
    two_zodiac = [
        z[0]
        for z in zodiac_count.most_common(2)
    ]

    return {
        "numbers": top_numbers,
        "special": [x[0] for x in final_rank[:8]],
        "detail": final_rank[:20],
        "zodiac": zodiac_count.most_common(),
        "wave": wave_count.most_common(),
        "prob": prob,
        "kill": kill_numbers[:6],
        "combo2": combo2[:10],
        "combo3": combo3[:10],
        "triple_zodiac": zodiac_hot,
        "one_zodiac": one_zodiac,
        "two_zodiac": two_zodiac
    }

# ==========================================
# 页面
# ==========================================
st.title("澳门六合彩 AI 超级智能分析系统")

history_count = st.sidebar.slider(
    "分析最近期数",
    5,
    30,
    12
)

with st.spinner("正在获取数据..."):

    latest = fetch_latest()

    history = fetch_history(datetime.now().year)

if not latest:
    st.error("获取开奖失败")
    st.stop()

latest_item = sorted(
    latest,
    key=lambda x:x["expect"],
    reverse=True
)[0]

# 去重
seen = set()
clean_history = []

for item in history:

    if item["expect"] not in seen:
        seen.add(item["expect"])
        clean_history.append(item)

if latest_item["expect"] not in seen:
    clean_history.insert(0, latest_item)

df = parse_history(clean_history)

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    by="time",
    ascending=False
).reset_index(drop=True)

# AI预测
result = ai_engine(df, history_count)

# 最新开奖
st.header("最新开奖")
st.success(latest_item["openCode"])

# AI推荐号码
st.header("AI推荐号码")

st.success(
    " / ".join([
        f"{n:02d}"
        for n in result["numbers"]
    ])
)

# 杀码
st.header("AI杀码")

st.error(
    " / ".join([
        f"{n:02d}"
        for n in result["kill"]
    ])
)

# 二中二
st.header("AI二中二")

for item in result["combo2"]:
    st.info(
        " - ".join([
            f"{x:02d}"
            for x in item
        ])
    )

# 三中三
st.header("AI三中三")

for item in result["combo3"]:
    st.success(
        " - ".join([
            f"{x:02d}"
            for x in item
        ])
    )

# 平特一肖
st.header("AI平特一肖")
st.warning(result["one_zodiac"])

# 平特二肖
st.header("AI平特二肖")
st.warning(" / ".join(result["two_zodiac"]))

# 三连肖
st.header("AI平特三连肖")

st.warning(
    " → ".join(result["triple_zodiac"])
)

# 特码
st.header("AI推荐特码")

st.error(
    " / ".join([
        f"{n:02d}"
        for n in result["special"]
    ])
)

# 评分详情
st.header("AI号码评分")

detail_df = pd.DataFrame(
    result["detail"],
    columns=["号码","评分"]
)

detail_df["生肖"] = detail_df["号码"].apply(get_zodiac)
detail_df["波色"] = detail_df["号码"].apply(get_wave)
detail_df["概率%"] = detail_df["号码"].apply(
    lambda x: result["prob"][x]
)

st.dataframe(detail_df)

st.caption("系统每60秒自动刷新")