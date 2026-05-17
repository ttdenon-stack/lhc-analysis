import requests
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from datetime import datetime
import time
import numpy as np

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="分析系统",
    layout="wide"
)

# =========================
# 自动刷新
# =========================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

now = time.time()

if now - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = now
    st.rerun()

# =========================
# API
# =========================
LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"

# =========================
# 波色
# =========================
RED = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
BLUE = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
GREEN = {5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49}

# =========================
# 生肖
# =========================
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

# =========================
# 工具函数
# =========================
def get_wave(num):

    if num in RED:
        return "红"

    elif num in BLUE:
        return "蓝"

    return "绿"

def get_zodiac(num):

    for k, v in ZODIAC_MAP.items():

        if num in v:
            return k

    return "未知"

def get_tail(num):

    return num % 10

# =========================
# 获取最新开奖
# =========================
@st.cache_data(ttl=60)
def fetch_latest():

    try:

        r = requests.get(LATEST_API, timeout=15)

        data = r.json()

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

        return []

    except:
        return []

# =========================
# 获取历史开奖
# =========================
@st.cache_data(ttl=300)
def fetch_history(year):

    try:

        url = HISTORY_API.format(year)

        r = requests.get(url, timeout=20)

        data = r.json()

        if "data" in data:
            return data["data"]

        return []

    except:
        return []

# =========================
# 解析数据
# =========================
def parse_history(data):

    rows = []

    for item in data:

        nums = [int(x) for x in item["openCode"].split(",")]

        rows.append({
            "expect": item["expect"],
            "time": item["openTime"],
            "nums": nums,
            "normal": nums[:6],
            "special": nums[-1]
        })

    return pd.DataFrame(rows)

# =========================
# 波色转移矩阵
# =========================
def build_wave_markov(df, count=20):

    recent = df.head(count)

    transitions = defaultdict(int)

    prev = None

    for _, row in recent.iterrows():

        current = get_wave(row["special"])

        if prev is not None:

            transitions[(prev, current)] += 1

        prev = current

    return transitions

# =========================
# AI核心
# =========================
def ai_engine(df, history_count=10):

    recent = df.head(history_count)

    score = defaultdict(float)

    zodiac_score = defaultdict(float)

    wave_score = defaultdict(float)

    tail_score = defaultdict(float)

    total = len(recent)

    # =====================
    # 波色马尔可夫链
    # =====================
    transitions = build_wave_markov(df)

    specials = recent["special"].tolist()

    last_wave = get_wave(specials[0])

    next_wave_score = defaultdict(float)

    for (a, b), v in transitions.items():

        if a == last_wave:

            next_wave_score[b] += v

    # =====================
    # 主模型
    # =====================
    for idx, row in recent.iterrows():

        weight = (total - idx) / total

        # 正码
        for n in row["normal"]:

            score[n] += 2.5 * weight

            zodiac_score[get_zodiac(n)] += 1.5 * weight

            wave_score[get_wave(n)] += 1.2 * weight

            tail_score[get_tail(n)] += 1.0 * weight

        # 特码
        sp = row["special"]

        score[sp] += 6 * weight

        zodiac_score[get_zodiac(sp)] += 4 * weight

        wave_score[get_wave(sp)] += 3 * weight

        tail_score[get_tail(sp)] += 2 * weight

    # =====================
    # 遗漏周期
    # =====================
    all_rows = df["nums"].tolist()

    for n in range(1, 50):

        miss = 0

        for row in all_rows:

            if n in row:
                break

            miss += 1

        if miss >= 5:

            score[n] += miss * 1.2

    # =====================
    # 波色冷热轮动
    # =====================
    weak_wave = min(
        wave_score,
        key=wave_score.get
    )

    for n in range(1,50):

        if get_wave(n) == weak_wave:

            score[n] += 4

    # =====================
    # 马尔可夫波色
    # =====================
    if next_wave_score:

        target_wave = max(
            next_wave_score,
            key=next_wave_score.get
        )

        for n in range(1,50):

            if get_wave(n) == target_wave:

                score[n] += 5

    # =====================
    # 生肖冷热轮动
    # =====================
    weak_zodiac = min(
        zodiac_score,
        key=zodiac_score.get
    )

    for n in range(1,50):

        if get_zodiac(n) == weak_zodiac:

            score[n] += 4

    # =====================
    # 尾数走势
    # =====================
    weak_tail = min(
        tail_score,
        key=tail_score.get
    )

    for n in range(1,50):

        if get_tail(n) == weak_tail:

            score[n] += 3

    # =====================
    # 同尾杀
    # =====================
    latest_tails = [
        get_tail(x)
        for x in recent.iloc[0]["nums"]
    ]

    for n in range(1,50):

        if get_tail(n) in latest_tails:

            score[n] -= 4

    # =====================
    # 同波杀
    # =====================
    latest_waves = [
        get_wave(x)
        for x in recent.iloc[0]["nums"]
    ]

    latest_wave_counter = Counter(latest_waves)

    kill_wave = latest_wave_counter.most_common(1)[0][0]

    for n in range(1,50):

        if get_wave(n) == kill_wave:

            score[n] -= 3

    # =====================
    # 最近开号降温
    # =====================
    latest_nums = recent.iloc[0]["nums"]

    for n in latest_nums:

        score[n] -= 5

    # =====================
    # 连码规律
    # =====================
    for row in recent["normal"]:

        sorted_nums = sorted(row)

        for i in range(len(sorted_nums)-1):

            if sorted_nums[i+1] - sorted_nums[i] == 1:

                score[sorted_nums[i]] += 1.5

                score[sorted_nums[i+1]] += 1.5

    # =====================
    # 贝叶斯概率
    # =====================
    total_score = sum(score.values()) + 1

    probability_map = {}

    for n in range(1,50):

        probability_map[n] = round(
            (score[n] / total_score) * 100,
            2
        )

    # =====================
    # 最终排序
    # =====================
    final_rank = sorted(
        score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_numbers = [
        x[0]
        for x in final_rank[:12]
    ]

    special_top = [
        x[0]
        for x in final_rank[:8]
    ]

    zodiac_rank = sorted(
        zodiac_score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    wave_rank = sorted(
        wave_score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "numbers": top_numbers,
        "special": special_top,
        "zodiac": zodiac_rank,
        "wave": wave_rank,
        "detail": final_rank[:20],
        "prob": probability_map
    }

# =========================
# 页面
# =========================
st.title("分析系统")

history_count = st.sidebar.slider(
    "分析最近期数",
    5,
    20,
    10
)

with st.spinner("正在获取开奖数据..."):

    latest = fetch_latest()

    current_year = datetime.now().year

    history = fetch_history(current_year)

if not latest:

    st.error("获取开奖失败")

    st.stop()

latest_item = sorted(
    latest,
    key=lambda x: x["expect"],
    reverse=True
)[0]

# =========================
# 去重
# =========================
seen = set()

clean_history = []

for item in history:

    if item["expect"] not in seen:

        seen.add(item["expect"])

        clean_history.append(item)

if latest_item["expect"] not in seen:

    clean_history.insert(0, latest_item)

# =========================
# DataFrame
# =========================
df = parse_history(clean_history)

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    by="time",
    ascending=False
).reset_index(drop=True)

# =========================
# 最新开奖
# =========================
st.header("最新开奖")

st.success(
    f"{latest_item['openCode']}"
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

# =========================
# AI预测
# =========================
result = ai_engine(
    df,
    history_count
)

# =========================
# AI推荐号码
# =========================
st.header("AI推荐号码")

st.success(
    " / ".join([
        f"{n:02d}"
        for n in result["numbers"]
    ])
)

# =========================
# AI推荐特码
# =========================
st.header("AI推荐特码")

st.warning(
    " / ".join([
        f"{n:02d}"
        for n in result["special"]
    ])
)

# =========================
# AI热门生肖
# =========================
st.header("AI热门生肖")

for z, s in result["zodiac"][:5]:

    st.info(
        f"{z} → {round(s,2)}"
    )

# =========================
# AI热门波色
# =========================
st.header("AI热门波色")

for w, s in result["wave"][:3]:

    st.warning(
        f"{w}波 → {round(s,2)}"
    )

# =========================
# AI评分详情
# =========================
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

# =========================
# 最近开奖
# =========================
st.header(f"最近 {history_count} 期开奖")

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
    lambda x: " ".join([
        f"{n:02d}"
        for n in x
    ])
)

st.dataframe(show_df)

st.caption("系统每60秒自动刷新")