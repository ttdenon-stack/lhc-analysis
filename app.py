import requests
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import time
import os
import json
import math

# =====================================
# 页面配置
# =====================================

st.set_page_config(
    page_title="澳门六合彩 AI 超级智能分析系统",
    layout="wide"
)

# =====================================
# 自动刷新
# =====================================

st_autorefresh = st.empty()

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

if time.time() - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = time.time()
    st.rerun()

# =====================================
# API
# =====================================

LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"

# =====================================
# 波色
# =====================================

RED = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
BLUE = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
GREEN = {5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49}

# =====================================
# 生肖
# =====================================

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

# =====================================
# 工具函数
# =====================================

def get_wave(n):

    if n in RED:
        return "红"

    if n in BLUE:
        return "蓝"

    return "绿"


def get_zodiac(n):

    for k, v in ZODIAC_MAP.items():
        if n in v:
            return k

    return "未知"

# =====================================
# 获取数据
# =====================================

@st.cache_data(ttl=60)
def fetch_latest():

    try:

        r = requests.get(LATEST_API, timeout=10)
        data = r.json()

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

        return []

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

# =====================================
# 解析历史
# =====================================

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

# =====================================
# AI 热度分析
# =====================================

def analyze_numbers(history):

    score = defaultdict(float)

    total = len(history)

    recent_all = []

    for row in history[:5]:
        recent_all.extend(row)

    for idx, row in enumerate(history):

        weight = (total - idx) / total

        for n in row:

            # 热度
            score[n] += weight * 2.2

            # 周期加权
            if idx % 7 == 0:
                score[n] += 1.5

            # 波色轮动
            wave = get_wave(n)

            if wave == "绿":
                score[n] += 0.8

            # 生肖轮动
            zodiac = get_zodiac(n)

            if zodiac in ["龙", "马", "猴"]:
                score[n] += 0.6

    # 最近开号降温
    recent_counter = Counter(recent_all)

    for n, c in recent_counter.items():

        score[n] -= c * 1.8

    # 遗漏回补
    for n in range(1, 50):

        miss = 0

        for row in history:

            if n in row:
                break

            miss += 1

        score[n] += miss * 0.45

    result = sorted(
        score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return result

# =====================================
# AI 杀号
# =====================================

def kill_numbers(history):

    freq = Counter()

    for row in history[:20]:
        freq.update(row)

    cold = sorted(freq.items(), key=lambda x: x[1])

    result = []

    for n, _ in cold:

        if n not in history[0]:
            result.append(n)

        if len(result) == 6:
            break

    return result

# =====================================
# 2中2 AI
# =====================================

def predict_2in2(history):

    counter = defaultdict(float)

    total = len(history)

    for idx, row in enumerate(history):

        weight = (total - idx) / total

        nums = row[:6]

        for pair in combinations(nums, 2):

            pair = tuple(sorted(pair))

            counter[pair] += weight

    result = sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final = []

    for pair, _ in result:

        if pair not in final:
            final.append(pair)

        if len(final) >= 12:
            break

    return final

# =====================================
# 3中3 AI
# =====================================

def predict_3in3(history):

    counter = defaultdict(float)

    total = len(history)

    for idx, row in enumerate(history):

        weight = (total - idx) / total

        nums = row[:6]

        for group in combinations(nums, 3):

            group = tuple(sorted(group))

            counter[group] += weight

    result = sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final = []

    for group, _ in result:

        final.append(group)

        if len(final) >= 10:
            break

    return final

# =====================================
# 页面
# =====================================

st.title("🔥 澳门六合彩 AI 超级智能分析系统")

history_count = st.sidebar.slider(
    "分析期数",
    10,
    50,
    20
)

with st.spinner("AI 正在分析数据..."):

    latest = fetch_latest()

    year = datetime.now().year

    history = fetch_history(year)

if not latest:

    st.error("获取开奖失败")

    st.stop()

latest_item = latest[0]

df = parse_history(history)

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    by="time",
    ascending=False
).reset_index(drop=True)

history_data = df["normal"].tolist()

# =====================================
# 最新开奖
# =====================================

st.subheader("最新开奖")

st.success(
    f"{latest_item['expect']}期\n\n"
    f"{latest_item['openCode']}"
)

# =====================================
# AI综合分析
# =====================================

scores = analyze_numbers(
    history_data[:history_count]
)

top_numbers = [x[0] for x in scores[:12]]

st.subheader("AI综合推荐")

st.success(
    " / ".join(
        [f"{n:02d}" for n in top_numbers]
    )
)

# =====================================
# 胆码
# =====================================

st.subheader("AI胆码")

danma = top_numbers[:4]

st.warning(
    " / ".join(
        [f"{n:02d}" for n in danma]
    )
)

# =====================================
# 杀号
# =====================================

kill = kill_numbers(
    history_data[:history_count]
)

st.subheader("AI杀号")

st.error(
    " / ".join(
        [f"{n:02d}" for n in kill]
    )
)

# =====================================
# 2中2
# =====================================

st.subheader("AI 2中2")

for x in predict_2in2(history_data[:history_count]):

    st.success(
        " / ".join(
            [f"{n:02d}" for n in x]
        )
    )

# =====================================
# 3中3
# =====================================

st.subheader("AI 3中3")

for x in predict_3in3(history_data[:history_count]):

    st.info(
        " / ".join(
            [f"{n:02d}" for n in x]
        )
    )

# =====================================
# 平特一肖
# =====================================

zodiac_counter = Counter()

for row in history_data[:history_count]:

    zodiac_counter.update(
        [get_zodiac(n) for n in row]
    )

best_zodiac = zodiac_counter.most_common(3)

st.subheader("平特一肖")

st.success(
    " / ".join(
        [x[0] for x in best_zodiac]
    )
)

# =====================================
# 波色分析
# =====================================

wave_counter = Counter()

for row in history_data[:history_count]:

    wave_counter.update(
        [get_wave(n) for n in row]
    )

st.subheader("波色趋势")

st.info(str(dict(wave_counter)))

# =====================================
# 最近开奖
# =====================================

st.subheader("最近开奖")

show = df.head(20)[
    ["expect", "nums"]
]

show.columns = ["期号", "号码"]

show["号码"] = show["号码"].apply(
    lambda x: " ".join(
        [f"{n:02d}" for n in x]
    )
)

st.dataframe(show)

# =====================================
# 保存AI记录
# =====================================

save = {
    "time": str(datetime.now()),
    "top": top_numbers,
    "kill": kill
}

try:

    old = []

    if os.path.exists("ai_history.json"):

        with open(
            "ai_history.json",
            "r",
            encoding="utf-8"
        ) as f:

            old = json.load(f)

    old.append(save)

    old = old[-500:]

    with open(
        "ai_history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            old,
            f,
            ensure_ascii=False,
            indent=4
        )

except:
    pass

st.caption("AI系统每60秒自动更新")