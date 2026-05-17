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

# ==========================================
# AI学习文件
# ==========================================
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
# AI动态权重
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
        return default_weights

    try:

        with open(AI_LEARN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except:
        return default_weights

def save_ai_weights(weights):

    with open(AI_LEARN_FILE, "w", encoding="utf-8") as f:

        json.dump(
            weights,
            f,
            ensure_ascii=False,
            indent=4
        )

# ==========================================
# API获取
# ==========================================
@st.cache_data(ttl=60)
def fetch_latest():

    try:

        r = requests.get(LATEST_API, timeout=10)

        data = r.json()

        if isinstance(data, list):
            return data

        return [data]

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
# 数据解析
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
# 马尔可夫链
# ==========================================
def build_markov_chain(df, history_count=20):

    transitions = defaultdict(int)

    recent = df.head(history_count)

    prev = None

    for _, row in recent.iterrows():

        current = get_wave(row["special"])

        if prev is not None:
            transitions[(prev, current)] += 1

        prev = current

    return transitions

# ==========================================
# AI超级引擎
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

    if recent.empty:

        return {
            "numbers": [],
            "special": [],
            "detail": [],
            "zodiac": [],
            "wave": [],
            "prob": {}
        }

    total = len(recent)

    # ==========================================
    # 主循环
    # ==========================================
    for idx, row in recent.iterrows():

        decay = (total - idx) / total

        normals = row["normal"]
        special = row["special"]

        # 正码
        for n in normals:

            freq[n] += 1

            score[n] += 2.5 * decay

            zodiac_count[get_zodiac(n)] += 1
            wave_count[get_wave(n)] += 1
            tail_count[get_tail(n)] += 1
            zone_count[get_zone(n)] += 1

        # 特码
        score[special] += weights["special"] * decay

        zodiac_count[get_zodiac(special)] += 2
        wave_count[get_wave(special)] += 2
        tail_count[get_tail(special)] += 2

    # ==========================================
    # 遗漏模型
    # ==========================================
    all_nums = df["nums"].tolist()

    for n in range(1,50):

        miss = 0

        for row in all_nums:

            if n in row:
                break

            miss += 1

        score[n] += min(
            miss * weights["miss"],
            15
        )

    # ==========================================
    # 热号降温
    # ==========================================
    for n, c in freq.items():

        if c >= 4:
            score[n] -= c * weights["hot"]

    # ==========================================
    # 冷号增强
    # ==========================================
    for n in range(1,50):

        if freq[n] == 0:
            score[n] += weights["cold"]

    # ==========================================
    # 波色轮动
    # ==========================================
    if wave_count:

        weak_wave = min(
            wave_count,
            key=wave_count.get
        )

        for n in range(1,50):

            if get_wave(n) == weak_wave:
                score[n] += weights["wave"]

    # ==========================================
    # 生肖轮动
    # ==========================================
    if zodiac_count:

        weak_zodiac = min(
            zodiac_count,
            key=zodiac_count.get
        )

        for n in range(1,50):

            if get_zodiac(n) == weak_zodiac:
                score[n] += weights["zodiac"]

    # ==========================================
    # 尾数轮动
    # ==========================================
    if tail_count:

        weak_tail = min(
            tail_count,
            key=tail_count.get
        )

        for n in range(1,50):

            if get_tail(n) == weak_tail:
                score[n] += weights["tail"]

    # ==========================================
    # 区间轮动
    # ==========================================
    if zone_count:

        weak_zone = min(
            zone_count,
            key=zone_count.get
        )

        for n in range(1,50):

            if get_zone(n) == weak_zone:
                score[n] += weights["zone"]

    # ==========================================
    # 连码模型
    # ==========================================
    for row in recent["normal"]:

        nums = sorted(row)

        for i in range(len(nums)-1):

            diff = nums[i+1] - nums[i]

            if diff == 1:

                score[nums[i]] += weights["combo"]
                score[nums[i+1]] += weights["combo"]

            elif diff == 2:

                score[nums[i]] += 1
                score[nums[i+1]] += 1

    # ==========================================
    # 马尔可夫波色预测
    # ==========================================
    transitions = build_markov_chain(df)

    last_wave = get_wave(
        recent.iloc[0]["special"]
    )

    next_wave = defaultdict(int)

    for (a, b), v in transitions.items():

        if a == last_wave:
            next_wave[b] += v

    if next_wave:

        target_wave = max(
            next_wave,
            key=next_wave.get
        )

        for n in range(1,50):

            if get_wave(n) == target_wave:
                score[n] += weights["markov"]

    # ==========================================
    # 杀最新开奖号
    # ==========================================
    latest_nums = recent.iloc[0]["nums"]

    for n in latest_nums:
        score[n] -= 5

    # ==========================================
    # 概率归一化
    # ==========================================
    min_score = min(score.values())

    if min_score < 0:

        for n in score:
            score[n] += abs(min_score)

    total_score = sum(score.values())

    prob = {}

    if total_score <= 0:
        total_score = 1

    for n in range(1,50):

        prob[n] = round(
            score[n] / total_score * 100,
            2
        )

    # ==========================================
    # 排序
    # ==========================================
    final_rank = sorted(
        score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    return {
        "numbers":[x[0] for x in final_rank[:12]],
        "special":[x[0] for x in final_rank[:8]],
        "detail":final_rank[:20],
        "zodiac":zodiac_count.most_common(),
        "wave":wave_count.most_common(),
        "prob":prob
    }

# ==========================================
# 命中率统计
# ==========================================
def calculate_hit_rate(df, history_count=10):

    if len(df) < 30:
        return 0, 0

    hit1 = 0
    hit2 = 0
    total = 0

    for i in range(20, len(df)-1):

        try:

            train_df = df.iloc[i:i+history_count]

            next_row = df.iloc[i-1]

            result = ai_engine(
                train_df,
                history_count
            )

            predict = result["numbers"][:10]

            real = next_row["nums"]

            hit_count = len(
                set(predict) & set(real)
            )

            if hit_count >= 1:
                hit1 += 1

            if hit_count >= 2:
                hit2 += 1

            total += 1

        except:
            pass

    if total == 0:
        return 0, 0

    return (
        round(hit1 / total * 100, 2),
        round(hit2 / total * 100, 2)
    )

# ==========================================
# 特码模型
# ==========================================
def special_model(df, history_count=10):

    recent = df.head(history_count)

    score = defaultdict(float)

    for idx, row in recent.iterrows():

        decay = (history_count - idx) / history_count

        sp = row["special"]

        score[sp] += 8 * decay

        tail = get_tail(sp)

        for n in range(1,50):

            if get_tail(n) == tail:
                score[n] += 2 * decay

    result = sorted(
        score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    return [x[0] for x in result[:10]]

# ==========================================
# 连码模型
# ==========================================
def combo_model(df, history_count=10):

    recent = df.head(history_count)

    combo_score = defaultdict(float)

    for idx, row in recent.iterrows():

        decay = (history_count - idx) / history_count

        nums = sorted(row["normal"])

        for i in range(len(nums)-1):

            diff = nums[i+1] - nums[i]

            if diff == 1:

                combo_score[nums[i]] += 3 * decay
                combo_score[nums[i+1]] += 3 * decay

            elif diff == 2:

                combo_score[nums[i]] += 1.5 * decay
                combo_score[nums[i+1]] += 1.5 * decay

    latest_nums = recent.iloc[0]["normal"]

    for n in latest_nums:
        combo_score[n] -= 2

    result = sorted(
        combo_score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    return [x[0] for x in result[:10]]

# ==========================================
# 页面开始
# ==========================================
st.title("澳门六合彩 AI 超级智能分析系统")

history_count = st.sidebar.slider(
    "分析最近期数",
    5,
    30,
    12
)

# ==========================================
# 获取数据
# ==========================================
with st.spinner("正在获取数据..."):

    latest = fetch_latest()

    history = fetch_history(
        datetime.now().year
    )

if not latest:

    st.error("获取开奖失败")
    st.stop()

latest_item = sorted(
    latest,
    key=lambda x:x["expect"],
    reverse=True
)[0]

# ==========================================
# 去重
# ==========================================
seen = set()
clean_history = []

for item in history:

    if item["expect"] not in seen:

        seen.add(item["expect"])
        clean_history.append(item)

if latest_item["expect"] not in seen:
    clean_history.insert(0, latest_item)

# ==========================================
# DataFrame
# ==========================================
df = parse_history(clean_history)

if df.empty:

    st.error("历史数据为空")
    st.stop()

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    by="time",
    ascending=False
).reset_index(drop=True)

# ==========================================
# 最新开奖
# ==========================================
st.header("最新开奖")

st.success(
    latest_item["openCode"]
)

latest_nums = [
    int(x)
    for x in latest_item["openCode"].split(",")
]

info = []

for n in latest_nums:

    info.append(
        f"{n}({get_zodiac(n)}/{get_wave(n)}波)"
    )

st.info(" | ".join(info))

# ==========================================
# AI预测
# ==========================================
result = ai_engine(
    df,
    history_count
)

# ==========================================
# 命中率
# ==========================================
hit1, hit2 = calculate_hit_rate(
    df,
    history_count
)

st.header("AI历史命中率")

st.success(f"命中1个以上概率：{hit1}%")
st.warning(f"命中2个以上概率：{hit2}%")

# ==========================================
# AI推荐号码
# ==========================================
st.header("AI推荐号码")

st.success(
    " / ".join([
        f"{n:02d}"
        for n in result["numbers"]
    ])
)

# ==========================================
# AI推荐特码
# ==========================================
st.header("AI推荐特码")

st.error(
    " / ".join([
        f"{n:02d}"
        for n in result["special"]
    ])
)

# ==========================================
# AI强化特码
# ==========================================
special_ai = special_model(
    df,
    history_count
)

st.header("AI强化特码")

st.warning(
    " / ".join([
        f"{n:02d}"
        for n in special_ai
    ])
)

# ==========================================
# AI连码强化
# ==========================================
combo_ai = combo_model(
    df,
    history_count
)

st.header("AI连码强化")

st.info(
    " / ".join([
        f"{n:02d}"
        for n in combo_ai
    ])
)

# ==========================================
# AI热门生肖
# ==========================================
st.header("AI热门生肖")

for z, s in result["zodiac"][:5]:

    st.info(f"{z} → {s}")

# ==========================================
# AI热门波色
# ==========================================
st.header("AI热门波色")

for w, s in result["wave"][:3]:

    st.warning(f"{w}波 → {s}")

# ==========================================
# AI评分详情
# ==========================================
st.header("AI号码评分")

detail_df = pd.DataFrame(
    result["detail"],
    columns=["号码","评分"]
)

detail_df["生肖"] = detail_df["号码"].apply(
    get_zodiac
)

detail_df["波色"] = detail_df["号码"].apply(
    get_wave
)

detail_df["概率%"] = detail_df["号码"].apply(
    lambda x: result["prob"][x]
)

st.dataframe(detail_df)

# ==========================================
# 最近开奖
# ==========================================
st.header(f"最近{history_count}期开奖")

show_df = df.head(history_count)[[
    "expect",
    "time",
    "nums"
]].copy()

show_df.columns = [
    "期号",
    "开奖时间",
    "开奖号码"
]

show_df["开奖号码"] = show_df["开奖号码"].apply(
    lambda x:" ".join([
        f"{n:02d}"
        for n in x
    ])
)

st.dataframe(show_df)

# ==========================================
# AI自动学习
# ==========================================
weights = load_ai_weights()

if hit2 >= 55:
    weights["special"] += 0.2

if hit1 >= 80:
    weights["markov"] += 0.1

if hit2 < 30:
    weights["cold"] += 0.2

weights["special"] = round(
    max(1, weights["special"]),
    2
)

weights["markov"] = round(
    max(1, weights["markov"]),
    2
)

weights["cold"] = round(
    max(1, weights["cold"]),
    2
)

save_ai_weights(weights)

st.sidebar.success(
    f"AI动态学习参数：{weights}"
)

st.caption("系统每60秒自动刷新")