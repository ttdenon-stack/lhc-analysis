import json
import os
import requests
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from datetime import datetime
import time

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="澳门六合彩AI分析系统",
    layout="wide"
)

# =========================
# 自动刷新
# =========================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# =========================
# API
# =========================
LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"

# =========================
# AI学习文件
# =========================
AI_LEARN_FILE = "ai_learn.json"

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

    if num in BLUE:
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
# AI权重
# =========================
def load_ai_weights():

    default_weights = {
        "wave": 4,
        "zodiac": 4,
        "tail": 3,
        "miss": 2,
        "special": 6
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

# =========================
# 获取最新开奖
# =========================
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

# =========================
# 获取历史开奖
# =========================
@st.cache_data(ttl=300)
def fetch_history(year):

    try:

        url = HISTORY_API.format(year)

        r = requests.get(url, timeout=20)

        data = r.json()

        return data.get("data", [])

    except:
        return []

# =========================
# 数据解析
# =========================
def parse_history(data):

    rows = []

    for item in data:

        try:

            nums = [int(x) for x in item["openCode"].split(",")]

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

# =========================
# AI核心算法
# =========================
def ai_engine(df, history_count=10):

    recent = df.head(history_count)

    score = defaultdict(float)

    zodiac_score = Counter()
    wave_score = Counter()
    tail_score = Counter()
    freq_score = Counter()

    total = len(recent)

    for idx, row in recent.iterrows():

        weight = (total - idx) / total

        normals = row["normal"]
        special = row["special"]

        # 正码
        for n in normals:

            freq_score[n] += 1

            score[n] += 2.5 * weight

            zodiac_score[get_zodiac(n)] += 1

            wave_score[get_wave(n)] += 1

            tail_score[get_tail(n)] += 1

        # 特码
        score[special] += 7 * weight

        zodiac_score[get_zodiac(special)] += 2

        wave_score[get_wave(special)] += 2

        tail_score[get_tail(special)] += 2

    # =========================
    # 遗漏增强
    # =========================
    all_nums = df["nums"].tolist()

    for n in range(1, 50):

        miss = 0

        for row in all_nums:

            if n in row:
                break

            miss += 1

        score[n] += min(miss * 1.2, 12)

    # =========================
    # 热号降温
    # =========================
    for n, c in freq_score.items():

        if c >= 4:
            score[n] -= c * 1.5

    # =========================
    # 冷号增强
    # =========================
    for n in range(1,50):

        if freq_score[n] == 0:
            score[n] += 5

    # =========================
    # 波色轮动
    # =========================
    if wave_score:

        weak_wave = min(wave_score, key=wave_score.get)

        for n in range(1,50):

            if get_wave(n) == weak_wave:
                score[n] += 3

    # =========================
    # 生肖轮动
    # =========================
    if zodiac_score:

        weak_zodiac = min(zodiac_score, key=zodiac_score.get)

        for n in range(1,50):

            if get_zodiac(n) == weak_zodiac:
                score[n] += 3

    # =========================
    # 尾数轮动
    # =========================
    if tail_score:

        weak_tail = min(tail_score, key=tail_score.get)

        for n in range(1,50):

            if get_tail(n) == weak_tail:
                score[n] += 2

    # =========================
    # 连码分析
    # =========================
    for row in recent["normal"]:

        nums = sorted(row)

        for i in range(len(nums)-1):

            diff = nums[i+1] - nums[i]

            if diff == 1:

                score[nums[i]] += 2
                score[nums[i+1]] += 2

            elif diff == 2:

                score[nums[i]] += 1
                score[nums[i+1]] += 1

    # =========================
    # 杀最近开奖号
    # =========================
    latest_nums = recent.iloc[0]["nums"]

    for n in latest_nums:
        score[n] -= 5

    # =========================
    # 概率
    # =========================
    min_score = min(score.values())

    if min_score < 0:

        for n in score:
            score[n] += abs(min_score)

    total_score = sum(score.values())

    prob = {}

    for n in range(1,50):

        prob[n] = round(
            score[n] / total_score * 100,
            2
        )

    # =========================
    # 排序
    # =========================
    final_rank = sorted(
        score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    return {
        "numbers":[x[0] for x in final_rank[:12]],
        "special":[x[0] for x in final_rank[:8]],
        "detail":final_rank[:20],
        "zodiac":zodiac_score.most_common(),
        "wave":wave_score.most_common(),
        "prob":prob
    }

# =========================
# 命中率统计
# =========================
def calculate_hit_rate(df, history_count=10):

    hit1 = 0
    hit2 = 0
    total = 0

    for i in range(20, len(df)-1):

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

    if total == 0:
        return 0,0

    return (
        round(hit1 / total * 100, 2),
        round(hit2 / total * 100, 2)
    )

# =========================
# 特码模型
# =========================
def special_model(df, history_count=10):

    recent = df.head(history_count)

    score = defaultdict(float)

    total = len(recent)

    for idx, row in recent.iterrows():

        weight = (total - idx) / total

        sp = row["special"]

        score[sp] += 8 * weight

        tail = get_tail(sp)

        for n in range(1,50):

            if get_tail(n) == tail:
                score[n] += 2 * weight

    result = sorted(
        score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    return [x[0] for x in result[:10]]

# =========================
# 连码模型
# =========================
def combo_model(df, history_count=10):

    recent = df.head(history_count)

    combo_score = defaultdict(float)

    for idx, row in recent.iterrows():

        weight = (history_count - idx) / history_count

        nums = sorted(row["normal"])

        for i in range(len(nums)-1):

            diff = nums[i+1] - nums[i]

            if diff == 1:

                combo_score[nums[i]] += 3 * weight
                combo_score[nums[i+1]] += 3 * weight

            elif diff == 2:

                combo_score[nums[i]] += 1.5 * weight
                combo_score[nums[i+1]] += 1.5 * weight

    latest_nums = recent.iloc[0]["normal"]

    for n in latest_nums:
        combo_score[n] -= 2

    result = sorted(
        combo_score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    return [x[0] for x in result[:10]]

# =========================
# 页面开始
# =========================
st.title("澳门六合彩AI智能分析系统")

history_count = st.sidebar.slider(
    "分析最近期数",
    5,
    20,
    10
)

with st.spinner("正在获取开奖数据..."):

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

st.success(latest_item["openCode"])

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
# 命中率
# =========================
hit1, hit2 = calculate_hit_rate(
    df,
    history_count
)

st.header("AI历史命中率")

st.success(f"命中1个以上概率：{hit1}%")

st.warning(f"命中2个以上概率：{hit2}%")

# =========================
# 特码强化
# =========================
special_ai = special_model(
    df,
    history_count
)

st.header("AI强化特码")

st.error(
    " / ".join([
        f"{n:02d}"
        for n in special_ai
    ])
)

# =========================
# 连码强化
# =========================
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
# 热门生肖
# =========================
st.header("AI热门生肖")

for z, s in result["zodiac"][:5]:

    st.info(f"{z} → {s}")

# =========================
# 热门波色
# =========================
st.header("AI热门波色")

for w, s in result["wave"][:3]:

    st.warning(f"{w}波 → {s}")

# =========================
# AI评分详情
# =========================
st.header("AI号码评分")

detail_df = pd.DataFrame(
    result["detail"],
    columns=["号码", "评分"]
)

detail_df["生肖"] = detail_df["号码"].apply(get_zodiac)

detail_df["波色"] = detail_df["号码"].apply(get_wave)

detail_df["概率%"] = detail_df["号码"].apply(
    lambda x: result["prob"][x]
)

st.dataframe(detail_df)

# =========================
# 最近开奖
# =========================
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

# =========================
# AI动态学习
# =========================
weights = load_ai_weights()

if hit2 >= 55:
    weights["special"] += 0.2

if hit1 >= 80:
    weights["wave"] += 0.1

weights["special"] = round(
    max(1, weights["special"]),
    2
)

weights["wave"] = round(
    max(1, weights["wave"]),
    2
)

save_ai_weights(weights)

st.sidebar.success(
    f"AI动态权重：{weights}"
)

st.caption("系统每60秒自动刷新")