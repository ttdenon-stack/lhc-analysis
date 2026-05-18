import os
import json
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st

from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="六合彩 AI Ultra v3 自学习版",
    layout="wide"
)

st.title("六合彩 AI Ultra v3 自学习版")

# ==========================================
# API
# ==========================================
LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# ==========================================
# AI学习文件
# ==========================================
AI_FILE = "ai_weights.json"

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
# 五行
# ==========================================
ELEMENT_MAP = {
    "金":[1,2,15,16,23,24,31,32,45,46],
    "木":[5,6,13,14,27,28,35,36,43,44],
    "水":[9,10,17,18,25,26,39,40,47,48],
    "火":[3,4,11,12,19,20,33,34,41,42,49],
    "土":[7,8,21,22,29,30,37,38]
}

# ==========================================
# 工具函数
# ==========================================
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

def get_tail(n):
    return n % 10

def get_zone(n):

    if n <= 16:
        return "低"

    elif n <= 33:
        return "中"

    return "高"

def get_element(n):

    for e, nums in ELEMENT_MAP.items():

        if n in nums:
            return e

    return "未知"

# ==========================================
# AI权重
# ==========================================
def load_weights():

    default = {
        "miss":2.5,
        "hot":1.8,
        "cold":3,
        "wave":2,
        "zodiac":2,
        "tail":2,
        "zone":1.5,
        "element":2,
        "consecutive":2.5,
        "special":5
    }

    if not os.path.exists(AI_FILE):

        with open(AI_FILE,"w",encoding="utf-8") as f:
            json.dump(default,f,ensure_ascii=False,indent=4)

        return default

    try:

        with open(AI_FILE,"r",encoding="utf-8") as f:
            return json.load(f)

    except:
        return default

def save_weights(weights):

    with open(AI_FILE,"w",encoding="utf-8") as f:

        json.dump(
            weights,
            f,
            ensure_ascii=False,
            indent=4
        )

# ==========================================
# API
# ==========================================
@st.cache_data(ttl=180)
def fetch_latest():

    try:

        r = requests.get(
            LATEST_API,
            headers=HEADERS,
            timeout=15
        )

        data = r.json()

        return data if isinstance(data,list) else [data]

    except:
        return []

@st.cache_data(ttl=300)
def fetch_history(year):

    try:

        url = HISTORY_API.format(year)

        r = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        return r.json().get("data",[])

    except:
        return []

# ==========================================
# 解析
# ==========================================
def parse_data(data):

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
                "expect":item["expect"],
                "time":item["openTime"],
                "nums":nums,
                "normal":nums[:6],
                "special":nums[-1]
            })

        except:
            pass

    return pd.DataFrame(rows)

# ==========================================
# AI核心
# ==========================================
def ai_engine(df, history_count=12):

    weights = load_weights()

    recent = df.head(history_count)

    score = defaultdict(float)

    freq = Counter()
    zodiac_count = Counter()
    wave_count = Counter()
    tail_count = Counter()
    zone_count = Counter()
    element_count = Counter()

    # ======================================
    # 时间衰减
    # ======================================
    for idx, row in recent.iterrows():

        decay = math.exp(-(idx / max(history_count,1)))

        for n in row["normal"]:

            freq[n] += 1

            score[n] += 3.5 * decay

            zodiac_count[get_zodiac(n)] += 1
            wave_count[get_wave(n)] += 1
            tail_count[get_tail(n)] += 1
            zone_count[get_zone(n)] += 1
            element_count[get_element(n)] += 1

        score[row["special"]] += (
            weights["special"] * decay
        )

    # ======================================
    # 遗漏
    # ======================================
    all_nums = df["nums"].tolist()

    for n in range(1,50):

        miss = 0

        for row in all_nums:

            if n in row:
                break

            miss += 1

        score[n] += (
            min(miss,18)
            * weights["miss"]
        )

    # ======================================
    # 热冷
    # ======================================
    for n in range(1,50):

        if freq[n] >= 5:

            score[n] -= (
                freq[n]
                * weights["hot"]
            )

        if freq[n] == 0:

            score[n] += weights["cold"]

    # ======================================
    # 波色轮动
    # ======================================
    if wave_count:

        weak_wave = min(
            wave_count,
            key=wave_count.get
        )

        for n in range(1,50):

            if get_wave(n) == weak_wave:

                score[n] += weights["wave"]

    # ======================================
    # 生肖轮动
    # ======================================
    if zodiac_count:

        weak_zodiac = min(
            zodiac_count,
            key=zodiac_count.get
        )

        for n in range(1,50):

            if get_zodiac(n) == weak_zodiac:

                score[n] += weights["zodiac"]

    # ======================================
    # 尾数轮动
    # ======================================
    if tail_count:

        weak_tail = min(
            tail_count,
            key=tail_count.get
        )

        for n in range(1,50):

            if get_tail(n) == weak_tail:

                score[n] += weights["tail"]

    # ======================================
    # 区间轮动
    # ======================================
    if zone_count:

        weak_zone = min(
            zone_count,
            key=zone_count.get
        )

        for n in range(1,50):

            if get_zone(n) == weak_zone:

                score[n] += weights["zone"]

    # ======================================
    # 五行轮动
    # ======================================
    if element_count:

        weak_element = min(
            element_count,
            key=element_count.get
        )

        for n in range(1,50):

            if get_element(n) == weak_element:

                score[n] += weights["element"]

    # ======================================
    # 连码趋势
    # ======================================
    for row in recent["normal"]:

        nums = sorted(row)

        for i in range(len(nums)-1):

            diff = nums[i+1] - nums[i]

            if diff == 1:

                score[nums[i]] += (
                    weights["consecutive"]
                )

                score[nums[i+1]] += (
                    weights["consecutive"]
                )

    # ======================================
    # 连尾
    # ======================================
    try:

        last_tails = [
            get_tail(x)
            for x in recent.iloc[0]["nums"]
        ]

        for n in range(1,50):

            if get_tail(n) in last_tails:

                score[n] += 1.2

    except:
        pass

    # ======================================
    # 杀码
    # ======================================
    kill = []

    try:

        latest_nums = recent.iloc[0]["nums"]

        for n in range(1,50):

            penalty = 0

            if n in latest_nums:
                penalty += 6

            if freq[n] >= 4:
                penalty += 4

            score[n] -= penalty

            if penalty >= 6:
                kill.append(n)

    except:
        pass

    # ======================================
    # 防负数
    # ======================================
    for n in range(1,50):

        score[n] = max(score[n],0.1)

    # ======================================
    # 排序
    # ======================================
    final_rank = sorted(
        score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    top_numbers = [
        x[0]
        for x in final_rank[:12]
    ]

    # ======================================
    # 胆码
    # ======================================
    danma = top_numbers[:4]

    # ======================================
    # 平特一肖
    # ======================================
    zodiac_score = Counter()

    for n in top_numbers:

        zodiac_score[
            get_zodiac(n)
        ] += score[n]

    yixiao = zodiac_score.most_common(5)

    # ======================================
    # 二中二
    # ======================================
    combo2 = list(
        combinations(top_numbers[:8],2)
    )[:10]

    # ======================================
    # 三中三
    # ======================================
    combo3 = list(
        combinations(top_numbers[:8],3)
    )[:10]

    # ======================================
    # 概率
    # ======================================
    total_score = sum(score.values())

    prob = {}

    for n in range(1,50):

        prob[n] = round(
            score[n]
            / total_score
            * 100,
            2
        )

    return {

        "numbers":top_numbers,
        "special":top_numbers[:8],
        "kill":kill[:6],
        "combo2":combo2,
        "combo3":combo3,
        "danma":danma,
        "yixiao":yixiao,
        "detail":final_rank[:20],
        "prob":prob
    }

# ==========================================
# AI自学习系统
# ==========================================
def auto_learn(df, history_count):

    weights = load_weights()

    if len(df) < history_count + 8:
        return weights, 0

    hit_total = 0
    total = 0

    for i in range(history_count, min(len(df)-1, 40)):

        train_df = df.iloc[i-history_count:i]

        real = df.iloc[i]["nums"]

        result = ai_engine(
            train_df,
            history_count
        )

        predict = result["numbers"][:10]

        hit = len(
            set(predict)
            & set(real)
        )

        hit_total += hit
        total += 1

    if total == 0:
        return weights, 0

    avg_hit = hit_total / total

    # ======================================
    # AI动态学习
    # ======================================
    if avg_hit >= 2.5:

        weights["wave"] += 0.05
        weights["zodiac"] += 0.05
        weights["consecutive"] += 0.03

    else:

        weights["cold"] += 0.08
        weights["miss"] += 0.08

    # ======================================
    # 限制范围
    # ======================================
    for k in weights:

        weights[k] = round(
            min(
                max(weights[k],0.5),
                8
            ),
            2
        )

    save_weights(weights)

    return weights, round(avg_hit,2)

# ==========================================
# 刷新
# ==========================================
if st.button("刷新最新数据"):

    st.cache_data.clear()

# ==========================================
# 滑块
# ==========================================
history_count = st.sidebar.slider(
    "分析最近期数",
    1,
    25,
    12
)

# ==========================================
# 获取数据
# ==========================================
with st.spinner("AI超级学习分析中..."):

    latest = fetch_latest()

    history = fetch_history(
        datetime.now().year
    )

if not latest:

    st.error("API获取失败")
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

clean = []

for item in history:

    if item["expect"] not in seen:

        seen.add(item["expect"])

        clean.append(item)

if latest_item["expect"] not in seen:

    clean.insert(0,latest_item)

# ==========================================
# DataFrame
# ==========================================
df = parse_data(clean)

if df.empty:

    st.error("历史数据为空")
    st.stop()

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    by="time",
    ascending=False
).reset_index(drop=True)

# ==========================================
# AI分析
# ==========================================
result = ai_engine(
    df,
    history_count
)

# ==========================================
# AI自动学习
# ==========================================
weights, avg_hit = auto_learn(
    df,
    history_count
)

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
        f"{x:02d}"
        for x in result["numbers"]
    ])
)

# ==========================================
# 胆码
# ==========================================
st.header("AI最强胆码")

st.warning(
    " / ".join([
        f"{x:02d}"
        for x in result["danma"]
    ])
)

# ==========================================
# 平特一肖
# ==========================================
st.header("AI平特一肖")

for z, s in result["yixiao"]:

    nums = " ".join([
        f"{x:02d}"
        for x in ZODIAC_MAP[z]
    ])

    st.info(f"{z} → {nums}")

# ==========================================
# 杀码
# ==========================================
st.header("AI杀码")

st.error(
    " / ".join([
        f"{x:02d}"
        for x in result["kill"]
    ])
)

# ==========================================
# 二中二
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
# 三中三
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
# 特码
# ==========================================
st.header("AI推荐特码")

st.error(
    " / ".join([
        f"{x:02d}"
        for x in result["special"]
    ])
)

# ==========================================
# AI回测统计
# ==========================================
st.header("AI历史回测")

st.success(
    f"平均每期命中：{avg_hit} 个"
)

# ==========================================
# AI权重
# ==========================================
st.header("AI动态学习权重")

weight_df = pd.DataFrame(
    list(weights.items()),
    columns=["模型","权重"]
)

st.dataframe(
    weight_df,
    use_container_width=True
)

# ==========================================
# 评分详情
# ==========================================
st.header("AI评分详情")

detail_df = pd.DataFrame(
    result["detail"],
    columns=["号码","评分"]
)

detail_df["生肖"] = detail_df["号码"].apply(get_zodiac)
detail_df["波色"] = detail_df["号码"].apply(get_wave)
detail_df["五行"] = detail_df["号码"].apply(get_element)

detail_df["概率%"] = detail_df["号码"].apply(
    lambda x: result["prob"][x]
)

st.dataframe(
    detail_df,
    use_container_width=True
)

# ==========================================
# 最近开奖
# ==========================================
st.header(f"最近{history_count}期开奖")

show_df = df.head(history_count)[[
    "expect",
    "time",
    "nums"
]]

show_df.columns = [
    "期号",
    "时间",
    "号码"
]

show_df["号码"] = show_df["号码"].apply(
    lambda x:" ".join([
        f"{n:02d}"
        for n in x
    ])
)

st.dataframe(
    show_df,
    use_container_width=True
)

# ==========================================
# 结束
# ==========================================
st.caption("六合彩 AI Ultra v3 自学习版")