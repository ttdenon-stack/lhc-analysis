import json
import os
import requests
import pandas as pd
import streamlit as st
import math

from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

# =====================================================
# 页面配置
# =====================================================

st.set_page_config(
    page_title="AI 超级智能分析系统 Pro Max",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# API
# =====================================================

LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

AI_FILE = "ai_learn.json"

# =====================================================
# 波色
# =====================================================

RED = {
    1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46
}

BLUE = {
    3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48
}

GREEN = {
    5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49
}

# =====================================================
# 五行
# =====================================================

ELEMENTS = {
    "金": [1,2,9,10,17,18,25,26,33,34,41,42,49],
    "木": [5,6,13,14,21,22,29,30,37,38,45,46],
    "水": [11,12,19,20,27,28,35,36,43,44],
    "火": [3,4,15,16,23,24,31,32,39,40,47,48],
    "土": [7,8]
}

# =====================================================
# 生肖
# =====================================================

ZODIAC_MAP = {
    "鼠": [7,19,31,43],
    "牛": [6,18,30,42],
    "虎": [5,17,29,41],
    "兔": [4,16,28,40],
    "龙": [3,15,27,39],
    "蛇": [2,14,26,38],
    "马": [1,13,25,37,49],
    "羊": [12,24,36,48],
    "猴": [11,23,35,47],
    "鸡": [10,22,34,46],
    "狗": [9,21,33,45],
    "猪": [8,20,32,44]
}

# =====================================================
# 工具函数
# =====================================================

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


def get_element(num):

    for e, nums in ELEMENTS.items():
        if num in nums:
            return e

    return "未知"


def get_tail(num):
    return num % 10


# =====================================================
# AI 权重
# =====================================================

def load_ai():

    default = {
        "miss": 2.0,
        "hot": 1.8,
        "cold": 2.5,
        "wave": 2.0,
        "zodiac": 2.0,
        "tail": 2.0,
        "element": 2.0,
        "consecutive": 2.5,
        "special": 5.0,
        "bayes": 2.0
    }

    if not os.path.exists(AI_FILE):

        save_ai(default)
        return default

    try:

        with open(AI_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        for k, v in default.items():

            if k not in data:
                data[k] = v

        return data

    except:
        return default


def save_ai(data):

    try:

        with open(AI_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    except:
        pass

# =====================================================
# 获取数据
# =====================================================

@st.cache_data(ttl=60)
def fetch_latest():

    try:

        r = requests.get(
            LATEST_API,
            headers=HEADERS,
            timeout=10
        )

        data = r.json()

        return data if isinstance(data, list) else [data]

    except:
        return []


@st.cache_data(ttl=300)
def fetch_history(year):

    try:

        r = requests.get(
            HISTORY_API.format(year),
            headers=HEADERS,
            timeout=10
        )

        return r.json().get("data", [])

    except:
        return []


def parse_history(data):

    rows = []

    for item in data:

        try:

            nums = [
                int(x)
                for x in item["openCode"].split(",")
            ]

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

# =====================================================
# 贝叶斯模型
# =====================================================

def bayes_model(df, score, weights):

    recent = df.head(20)

    freq = Counter([
        n
        for row in recent["nums"]
        for n in row
    ])

    total = sum(freq.values())

    for n in range(1, 50):

        p = (freq[n] + 1) / (total + 49)

        score[n] += (
            p * 100 * weights["bayes"]
        )

# =====================================================
# 一肖模型
# =====================================================

def yixiao_model(df, history_count):

    recent = df.head(history_count)

    zodiac_score = Counter()

    for idx, row in recent.iterrows():

        decay = (
            history_count - idx
        ) / history_count

        for n in row["normal"]:

            zodiac_score[
                get_zodiac(n)
            ] += 1.2 * decay

        zodiac_score[
            get_zodiac(row["special"])
        ] += 2.5 * decay

    result = sorted(
        zodiac_score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        x[0]
        for x in result[:5]
    ]

# =====================================================
# AI 核心引擎
# =====================================================

def ai_engine(df, history_count=12, learning=True):

    weights = load_ai()

    recent = df.head(history_count)

    score = defaultdict(float)

    freq = Counter()

    zodiac_count = Counter()
    wave_count = Counter()
    element_count = Counter()

    total = max(len(recent), 1)

    # =================================================
    # 基础统计
    # =================================================

    for idx, row in recent.iterrows():

        decay = math.exp(-(idx / total))

        for n in row["normal"]:

            freq[n] += 1

            score[n] += 2.5 * decay

            zodiac_count[get_zodiac(n)] += 1
            wave_count[get_wave(n)] += 1
            element_count[get_element(n)] += 1

        score[row["special"]] += (
            weights["special"] * decay
        )

    # =================================================
    # 遗漏值
    # =================================================

    all_nums = df["nums"].tolist()

    for n in range(1, 50):

        miss = 0

        for row in all_nums:

            if n in row:
                break

            miss += 1

        score[n] += (
            min(miss, 20)
            * weights["miss"]
        )

    # =================================================
    # 冷热
    # =================================================

    for n in range(1, 50):

        if freq[n] >= 4:
            score[n] -= (
                freq[n]
                * weights["hot"]
            )

        if freq[n] == 0:
            score[n] += weights["cold"]

    # =================================================
    # 属性轮动
    # =================================================

    if wave_count:

        weak_wave = min(
            wave_count,
            key=wave_count.get
        )

        for n in range(1, 50):

            if get_wave(n) == weak_wave:
                score[n] += weights["wave"]

    if zodiac_count:

        weak_zodiac = min(
            zodiac_count,
            key=zodiac_count.get
        )

        for n in range(1, 50):

            if get_zodiac(n) == weak_zodiac:
                score[n] += weights["zodiac"]

    if element_count:

        weak_element = min(
            element_count,
            key=element_count.get
        )

        for n in range(1, 50):

            if get_element(n) == weak_element:
                score[n] += weights["element"]

    # =================================================
    # 贝叶斯
    # =================================================

    bayes_model(df, score, weights)

    # =================================================
    # 连码
    # =================================================

    for row in recent["normal"]:

        nums = sorted(row)

        for i in range(len(nums) - 1):

            # 连码
            if nums[i + 1] - nums[i] == 1:

                score[nums[i]] += weights["consecutive"]
                score[nums[i + 1]] += weights["consecutive"]

            # 跳码
            elif nums[i + 1] - nums[i] == 2:

                score[nums[i]] += 1.2
                score[nums[i + 1]] += 1.2

    # =================================================
    # 杀码
    # =================================================

    latest_nums = (
        recent.iloc[0]["nums"]
        if not recent.empty
        else []
    )

    kill = []

    for n in range(1, 50):

        penalty = 0

        if n in latest_nums:
            penalty += 5

        if freq[n] >= 6:
            penalty += 4

        if penalty >= 5:

            kill.append(n)

            score[n] -= penalty

        score[n] = max(score[n], 0.1)

    # =================================================
    # 排序
    # =================================================

    final_rank = sorted(
        score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    numbers = [
        n
        for n, _ in final_rank
        if n not in kill
    ][:12]

    # =================================================
    # 胆码
    # =================================================

    danma = []

    used_wave = set()

    for n in numbers:

        w = get_wave(n)

        if w not in used_wave:

            danma.append(n)

            used_wave.add(w)

        if len(danma) >= 5:
            break

    # =================================================
    # 特码
    # =================================================

    special_score = {
        n: score[n]
        for n in range(1, 50)
    }

    if not recent.empty:

        for n in range(1, 50):

            if n in latest_nums:
                special_score[n] -= 6

            if get_tail(n) == get_tail(recent.iloc[0]["special"]):
                special_score[n] -= 3

            if get_wave(n) == get_wave(recent.iloc[0]["special"]):
                special_score[n] -= 2

    special = [
        x[0]
        for x in sorted(
            special_score.items(),
            key=lambda x: x[1],
            reverse=True
        )[:8]
    ]

    # =================================================
    # 概率
    # =================================================

    total_score = sum(score.values())

    prob = {
        n: round(
            score[n] / total_score * 100,
            2
        )
        for n in range(1, 50)
    }

    # =================================================
    # 组合
    # =================================================

    combo2 = list(
        combinations(numbers[:8], 2)
    )[:10]

    combo3 = list(
        combinations(numbers[:8], 3)
    )[:10]

    # =================================================
    # AI学习
    # =================================================

    if learning:

        try:

            if len(df) > history_count + 2:

                future_real = df.iloc[0]["normal"]

                hit = len(
                    set(numbers[:10])
                    & set(future_real)
                )

                if hit >= 3:
                    weights["miss"] += 0.03
                    weights["cold"] += 0.03

                elif hit <= 1:
                    weights["hot"] -= 0.02

                for k in weights:

                    weights[k] = round(
                        max(
                            0.5,
                            min(weights[k], 8)
                        ),
                        2
                    )

                save_ai(weights)

        except:
            pass

    return {

        "numbers": numbers,
        "danma": danma,
        "kill": kill[:6],
        "special": special,
        "prob": prob,
        "combo2": combo2,
        "combo3": combo3,
        "yixiao": yixiao_model(df, history_count),
        "detail": final_rank[:20],
        "raw_score": score
    }

# =====================================================
# 多窗口融合
# =====================================================

def ensemble_engine(df):

    windows = [8, 12, 20]

    final_score = defaultdict(float)

    weights = [0.5, 0.3, 0.2]

    for win, w in zip(windows, weights):

        result = ai_engine(
            df,
            history_count=win,
            learning=False
        )

        for n, p in result["raw_score"].items():

            final_score[n] += p * w

    return final_score

# =====================================================
# UI
# =====================================================

st.title("📊 AI 超级智能分析系统 Pro Max")

st.sidebar.title("⚙️ 控制中心")

history_count = st.sidebar.slider(
    "分析最近期数",
    min_value=1,
    max_value=30,
    value=15,
    step=1
)

if st.sidebar.button("🔄 清理缓存"):

    st.cache_data.clear()

# =====================================================
# 获取数据
# =====================================================

with st.spinner("AI正在深度分析数据..."):

    latest = fetch_latest()

    history = fetch_history(
        datetime.now().year
    )

if not latest:

    st.error("无法获取最新数据")

    st.stop()

latest_item = sorted(
    latest,
    key=lambda x: x["expect"],
    reverse=True
)[0]

seen = set()

clean = []

for item in history:

    if item["expect"] not in seen:

        seen.add(item["expect"])

        clean.append(item)

if latest_item["expect"] not in seen:

    clean.insert(0, latest_item)

df = parse_history(clean)

if df.empty:

    st.error("历史数据为空")

    st.stop()

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    by="time",
    ascending=False
).reset_index(drop=True)

# =====================================================
# AI计算
# =====================================================

ensemble_score = ensemble_engine(df)

result = ai_engine(
    df,
    history_count,
    learning=True
)

for n in result["prob"]:

    result["prob"][n] = round(
        result["prob"][n]
        + ensemble_score[n] / 100,
        2
    )

# =====================================================
# UI显示
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("🔔 最新开奖")

    st.success(
        f"{latest_item['expect']} 期\n\n"
        f"{latest_item['openCode']}"
    )

with col2:

    st.subheader("🎯 AI推荐12码")

    st.info(
        " / ".join([
            f"{n:02d}"
            for n in result["numbers"]
        ])
    )

st.markdown("---")

col3, col4, col5 = st.columns(3)

with col3:

    st.subheader("⭐ 胆码")

    st.warning(
        " / ".join([
            f"{n:02d}"
            for n in result["danma"]
        ])
    )

with col4:

    st.subheader("💎 独立特码")

    st.error(
        " / ".join([
            f"{n:02d}"
            for n in result["special"]
        ])
    )

with col5:

    st.subheader("❌ 杀码")

    st.error(
        " / ".join([
            f"{n:02d}"
            for n in result["kill"]
        ])
    )

st.markdown("---")

st.subheader("🐅 平特一肖")

cols = st.columns(5)

for i, z in enumerate(result["yixiao"]):

    with cols[i]:

        nums = " ".join([
            f"{n:02d}"
            for n in ZODIAC_MAP[z]
        ])

        st.success(f"{z}\n\n{nums}")

st.markdown("---")

tab1, tab2 = st.tabs([
    "📊 综合评分",
    "🔬 组合"
])

with tab1:

    detail_df = pd.DataFrame(
        result["detail"],
        columns=["号码", "评分"]
    )

    detail_df["生肖"] = detail_df["号码"].apply(get_zodiac)
    detail_df["波色"] = detail_df["号码"].apply(get_wave)
    detail_df["五行"] = detail_df["号码"].apply(get_element)

    detail_df["概率"] = detail_df["号码"].apply(
        lambda x: f"{result['prob'][x]}%"
    )

    st.dataframe(
        detail_df,
        use_container_width=True
    )

with tab2:

    col_a, col_b = st.columns(2)

    with col_a:

        st.subheader("二中二")

        for item in result["combo2"]:

            st.code(
                " - ".join([
                    f"{x:02d}"
                    for x in item
                ])
            )

    with col_b:

        st.subheader("三中三")

        for item in result["combo3"]:

            st.code(
                " - ".join([
                    f"{x:02d}"
                    for x in item
                ])
            )

st.markdown("---")

st.caption(
    "本系统仅用于数据分析与概率研究。"
)