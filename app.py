import json
import os
import requests
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
import math
import time
import random

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="六合彩 AI Pro Max V4",
    layout="wide"
)

# ==========================================
# 自动刷新
# ==========================================
if "refresh" not in st.session_state:
    st.session_state.refresh = time.time()

if time.time() - st.session_state.refresh > 60:
    st.session_state.refresh = time.time()
    st.rerun()

# ==========================================
# API
# ==========================================
LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"

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
# API获取
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

        r = requests.get(
            HISTORY_API.format(year),
            timeout=20
        )

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

# ==========================================
# 波色趋势
# ==========================================
def wave_trend(df):

    recent = df.head(12)

    waves = [
        get_wave(x)
        for x in recent["special"]
    ]

    c = Counter(waves)

    weak = min(c, key=c.get)

    return weak

# ==========================================
# 生肖趋势
# ==========================================
def zodiac_trend(df):

    recent = df.head(12)

    zlist = []

    for _, row in recent.iterrows():

        for n in row["normal"]:
            zlist.append(get_zodiac(n))

    c = Counter(zlist)

    weak = min(c, key=c.get)

    return weak

# ==========================================
# 尾数趋势
# ==========================================
def tail_trend(df):

    recent = df.head(15)

    tails = []

    for _, row in recent.iterrows():

        for n in row["nums"]:
            tails.append(get_tail(n))

    c = Counter(tails)

    weak = min(c, key=c.get)

    return weak

# ==========================================
# AI核心
# ==========================================
def ai_engine(df):

    score = defaultdict(float)

    recent5 = df.head(5)
    recent10 = df.head(10)
    recent30 = df.head(30)

    freq5 = Counter()
    freq10 = Counter()
    freq30 = Counter()

    # ======================================
    # 多周期统计
    # ======================================
    for _, row in recent5.iterrows():

        for n in row["normal"]:
            freq5[n] += 1

    for _, row in recent10.iterrows():

        for n in row["normal"]:
            freq10[n] += 1

    for _, row in recent30.iterrows():

        for n in row["normal"]:
            freq30[n] += 1

    # ======================================
    # 主评分
    # ======================================
    for n in range(1, 50):

        score[n] += freq5[n] * 5
        score[n] += freq10[n] * 3
        score[n] += freq30[n] * 1.5

    # ======================================
    # 遗漏
    # ======================================
    all_nums = df["nums"].tolist()

    for n in range(1, 50):

        miss = 0

        for row in all_nums:

            if n in row:
                break

            miss += 1

        score[n] += min(miss, 15) * 2.2

    # ======================================
    # 波色趋势
    # ======================================
    weak_wave = wave_trend(df)

    for n in range(1, 50):

        if get_wave(n) == weak_wave:
            score[n] += 4

    # ======================================
    # 生肖趋势
    # ======================================
    weak_zodiac = zodiac_trend(df)

    for n in range(1, 50):

        if get_zodiac(n) == weak_zodiac:
            score[n] += 5

    # ======================================
    # 尾数趋势
    # ======================================
    weak_tail = tail_trend(df)

    for n in range(1, 50):

        if get_tail(n) == weak_tail:
            score[n] += 3

    # ======================================
    # 连码强化
    # ======================================
    for _, row in recent10.iterrows():

        nums = sorted(row["normal"])

        for i in range(len(nums)-1):

            diff = nums[i+1] - nums[i]

            if diff == 1:

                score[nums[i]] += 3
                score[nums[i+1]] += 3

    # ======================================
    # 杀热号
    # ======================================
    latest_nums = recent5.iloc[0]["nums"]

    kill = []

    for n in range(1, 50):

        penalty = 0

        if n in latest_nums:
            penalty += 8

        if freq5[n] >= 3:
            penalty += 5

        score[n] -= penalty

        if penalty >= 8:
            kill.append(n)

    # ======================================
    # 防负数
    # ======================================
    for n in range(1, 50):

        score[n] = max(score[n], 0.1)

    # ======================================
    # 排序
    # ======================================
    rank = sorted(
        score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    top_numbers = [
        x[0]
        for x in rank[:12]
    ]

    # ======================================
    # 特码模型
    # ======================================
    special_score = defaultdict(float)

    for _, row in recent10.iterrows():

        sp = row["special"]

        special_score[sp] += 8

        tail = get_tail(sp)

        for n in range(1, 50):

            if get_tail(n) == tail:
                special_score[n] += 2

    special_rank = sorted(
        special_score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    special_numbers = [
        x[0]
        for x in special_rank[:10]
    ]

    # ======================================
    # 平特一肖
    # ======================================
    zodiac_counter = Counter()

    for _, row in recent10.iterrows():

        for n in row["nums"]:
            zodiac_counter[get_zodiac(n)] += 1

    one_zodiac = zodiac_counter.most_common(1)[0][0]

    triple_zodiac = [
        x[0]
        for x in zodiac_counter.most_common(3)
    ]

    # ======================================
    # 二中二
    # ======================================
    combo2 = list(
        combinations(top_numbers[:8], 2)
    )[:10]

    # ======================================
    # 三中三
    # ======================================
    combo3 = list(
        combinations(top_numbers[:8], 3)
    )[:10]

    return {

        "numbers": top_numbers,

        "special": special_numbers,

        "kill": kill[:6],

        "combo2": combo2,

        "combo3": combo3,

        "one_zodiac": one_zodiac,

        "triple_zodiac": triple_zodiac,

        "detail": rank[:20]
    }

# ==========================================
# 页面
# ==========================================
st.title("六合彩 AI Pro Max V4")

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

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    by="time",
    ascending=False
).reset_index(drop=True)

# ==========================================
# AI预测
# ==========================================
result = ai_engine(df)

# ==========================================
# 最新开奖
# ==========================================
st.header("最新开奖")

st.success(latest_item["openCode"])

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
# AI杀码
# ==========================================
st.header("AI杀码")

st.error(
    " / ".join([
        f"{n:02d}"
        for n in result["kill"]
    ])
)

# ==========================================
# AI平特一肖
# ==========================================
st.header("AI平特一肖")

st.warning(result["one_zodiac"])

# ==========================================
# AI平特三连肖
# ==========================================
st.header("AI平特三连肖")

st.info(
    " → ".join(result["triple_zodiac"])
)

# ==========================================
# AI二中二
# ==========================================
st.header("AI二中二")

for item in result["combo2"]:

    st.info(
        " - ".join([
            f"{x:02d}"
            for x in item
        ])
    )

# ==========================================
# AI三中三
# ==========================================
st.header("AI三中三")

for item in result["combo3"]:

    st.success(
        " - ".join([
            f"{x:02d}"
            for x in item
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
# AI评分详情
# ==========================================
st.header("AI号码评分")

detail_df = pd.DataFrame(
    result["detail"],
    columns=["号码","评分"]
)

detail_df["生肖"] = detail_df["号码"].apply(get_zodiac)

detail_df["波色"] = detail_df["号码"].apply(get_wave)

st.dataframe(detail_df)

st.caption("系统每60秒自动刷新")