import requests
import pandas as pd
import numpy as np
import streamlit as st
from collections import Counter
from itertools import combinations
from datetime import datetime
import time
import os
import json
from collections import defaultdict

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="分析系统",
    layout="wide"
)

# =========================
# 自动刷新（60秒）
# =========================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

now = time.time()
if now - st.session_state.last_refresh > 60:
    st.session_state.last_refresh = now
    st.experimental_rerun()

# =========================
# API 配置
# =========================
LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"
# =========================
# 预测结果保存目录
# =========================
PREDICT_DIR = "prediction_results"
os.makedirs(PREDICT_DIR, exist_ok=True)

# =========================
# 波色表
# =========================
RED = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
BLUE = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
GREEN = {5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49}

# =========================
# 遗漏值分析
# =========================

def analyze_missing(history):
    missing = {}

    for n in range(1, 50):
        missing[n] = 0

    reversed_history = history[::-1]

    for n in range(1, 50):
        miss = 0

        for row in reversed_history:
            if n in row:
                break

            miss += 1

        missing[n] = miss

    sorted_missing = sorted(
        missing.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in sorted_missing[:10]]


# =========================
# 2中2 AI
# =========================

def predict_2in2(history):

    counter = defaultdict(float)

    total = len(history)

    for idx, row in enumerate(history):

        nums = row[:6]

        weight = (total - idx) / total

        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):

                pair = tuple(sorted([
                    nums[i],
                    nums[j]
                ]))

                counter[pair] += weight

    result = sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in result[:15]]


# =========================
# 3中3 AI
# =========================

def predict_3in3(history):

    counter = defaultdict(float)

    total = len(history)

    for idx, row in enumerate(history):

        nums = row[:6]

        # 越近权重越高
        weight = (total - idx) / total

        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):

                    group = tuple(sorted([
                        nums[i],
                        nums[j],
                        nums[k]
                    ]))

                    counter[group] += weight

    result = sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in result[:15]]


# =========================
# AI综合评分（增强版）
# =========================

def smart_score(hot, missing, cold):

    score = defaultdict(float)

    # 热号
    for i, n in enumerate(hot):
        score[n] += (10 - i) * 1.8

    # 遗漏号
    for i, n in enumerate(missing):
        score[n] += (10 - i) * 1.5

    # 冷号回补
    for i, n in enumerate(cold):
        score[n] += (10 - i) * 1.2

    # 最近开号降权
    recent_nums = history_data[0]

    for n in recent_nums:
        score[n] -= 4

    # 连开号轻微降温
    last3 = []

    for row in history_data[:3]:
        last3.extend(row)

    recent_counter = Counter(last3)

    for n, c in recent_counter.items():

        if c >= 2:
            score[n] -= 2

    final = sorted(
        score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [x[0] for x in final[:15]]

# =========================
# 生肖映射
# =========================
ZODIAC_MAP = {
    "鼠":[7,19,31,43],"牛":[6,18,30,42],"虎":[5,17,29,41],"兔":[4,16,28,40],
    "龙":[3,15,27,39],"蛇":[2,14,26,38],"马":[1,13,25,37,49],"羊":[12,24,36,48],
    "猴":[11,23,35,47],"鸡":[10,22,34,46],"狗":[9,21,33,45],"猪":[8,20,32,44]
}

def get_zodiac(num):
    for k, v in ZODIAC_MAP.items():
        if num in v:
            return k
    return "未知"

def get_wave(num):
    if num in RED:
        return "红"
    elif num in BLUE:
        return "蓝"
    return "绿"

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

        if isinstance(data, dict):
            return [data]

        return []

    except:
        return []

# =========================
# 获取历史数据
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
# 热号冷号分析
# =========================
def analyze_hot_cold(history):
    freq = defaultdict(int)

    for row in history:
        for num in row:
            freq[num] += 1

    sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    hot = [x[0] for x in sorted_nums[:10]]
    cold = [x[0] for x in sorted_nums[-10:]]

    return hot, cold

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
# 平特一肖分析（6区+特码） 
# =========================
def one_zodiac(df, history_count=15):
    df_recent = df.head(history_count)
    specials = []
    for row in df_recent["nums"]:  # 包括 normal+special
        specials.extend(row)
    z = [get_zodiac(x) for x in specials]
    c = Counter(z)
    recent = z[:3]
    for name, _ in c.most_common():
        if name not in recent:
            return name
    return c.most_common(1)[0][0]

# =========================
# 平特三连肖分析
# =========================
def three_zodiac(df, history_count=15):
    df_recent = df.head(history_count)
    specials = []
    for row in df_recent["nums"]:
        specials.extend(row)
    z = [get_zodiac(x) for x in specials]
    c = Counter(z)
    result = []
    recent = z[:3]
    for name, _ in c.most_common():
        if name not in recent:
            result.append(name)
        if len(result) == 3:
            break
    return result

# =========================
# 不分6区/特码的号码概率分析
# =========================
def predict_hot_numbers(df, history_count=15, top_n=3):
    df_recent = df.head(history_count)
    all_nums = []
    for row in df_recent["nums"]:
        all_nums.extend(row)
    counter = Counter(all_nums)
    sorted_nums = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    top_nums = [num for num, _ in sorted_nums[:top_n]]
    return top_nums

# =========================
# 联合波色+生肖+号码评分分析
# =========================
def combined_analysis(df, history_count=15, top_n=5):

    df_recent = df.head(history_count)

    score_map = {n:0 for n in range(1,50)}

    all_nums = []
    all_zodiac = []
    all_wave = []

    for row in df_recent["nums"]:

        all_nums.extend(row)

        all_zodiac.extend([
            get_zodiac(n)
            for n in row
        ])

        all_wave.extend([
            get_wave(n)
            for n in row
        ])

    freq = Counter(all_nums)
    zc = Counter(all_zodiac)
    wc = Counter(all_wave)

    # 最弱波色
    weak_wave = min(wc, key=wc.get)

    # 最弱生肖
    weak_zodiac = min(zc, key=zc.get)

    # 最近一期号码
    latest_nums = df_recent.iloc[0]["nums"]

    for n in range(1,50):

        s = 0

        # 频率
        s += freq.get(n,0) * 1.5

        # 生肖
        s += zc.get(get_zodiac(n),0)

        # 波色
        if get_wave(n) == weak_wave:
            s += 4

        # 弱生肖加成
        if get_zodiac(n) == weak_zodiac:
            s += 3

        # 最近开号降权
        if n in latest_nums:
            s -= 5

        score_map[n] = s

    sorted_score = sorted(
        score_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_nums = [
        num for num,_ in sorted_score[:top_n]
    ]

    return top_nums, sorted_score[:top_n]

# =========================
# 页面逻辑
# =========================
st.sidebar.header("分析参数")
history_count = st.sidebar.slider("选择历史期数", 5, 25, 15)

st.title("澳门六合彩 AI 平特分析系统")

with st.spinner("正在获取开奖数据..."):
    latest = fetch_latest()
    current_year = datetime.now().year
    history = fetch_history(current_year)

if not latest:
    st.error("无法获取最新开奖")
    st.stop()

latest_item = sorted(
    latest,
    key=lambda x: x["expect"],
    reverse=True
)[0]

# 去重+插入最新一期
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
df = df.sort_values(by="time", ascending=False).reset_index(drop=True)

history_data = df["normal"].tolist()

# 最新开奖
st.subheader("最新开奖")
st.success(f"期号：{latest_item['expect']}\n开奖号码：{latest_item['openCode']}")
st.info("对应生肖+波色：" + " / ".join([f"{n}:{get_zodiac(n)}/{get_wave(n)}" for n in [int(x) for x in latest_item['openCode'].split(",")]]))

# 平特一肖
st.subheader("平特一肖分析（6区+特码）")
zodiac_one = one_zodiac(df, history_count)
st.success(zodiac_one)

# 平特三连肖
st.subheader("平特三连肖分析（6区+特码）")
zodiac_three = three_zodiac(df, history_count)
st.success(" / ".join(zodiac_three))

# 不分6区/特码号码概率分析
st.subheader("号码概率分析（不分6区/特码）")
top_numbers = predict_hot_numbers(df, history_count, top_n=3)
st.success("推荐号码： " + " / ".join([f"{n:02d}" for n in top_numbers]))

# 联合波色+生肖+号码分析
st.subheader("联合波色+生肖+号码综合评分分析")
combined_top, combined_detail = combined_analysis(df, history_count, top_n=3)
st.success("推荐号码： " + " / ".join([f"{n:02d}" for n in combined_top]))
st.write("Top评分详情（号码:分数）:", ", ".join([f"{n}:{s}" for n,s in combined_detail]))

# 最近开奖表
st.subheader(f"最近 {history_count}期开奖")
show_df = df.head(history_count)[["expect","time","nums"]].copy()
show_df.columns = ["期号","开奖时间","开奖号码"]
show_df["开奖号码"] = show_df["开奖号码"].apply(lambda x: " ".join(str(n) for n in x))
st.dataframe(show_df)

# 自动刷新提示
st.caption("系统每60秒自动刷新一次")

# =========================
# AI 深度预测
# =========================

st.header("🔥 AI 深度预测")

hot, cold = analyze_hot_cold(history_data)

missing = analyze_missing(history_data)

smart_nums = smart_score(
    hot,
    missing,
    cold
)

# AI杀号
kill_nums = []

for n in cold:

    if n not in missing[:5]:
        kill_nums.append(n)

if len(kill_nums) < 6:

    for n in cold:

        if n not in kill_nums:
            kill_nums.append(n)

kill_nums = kill_nums[:6]

# 过滤杀号
smart_nums = [
    n for n in smart_nums
    if n not in kill_nums
]

st.subheader("热号")
st.success(hot)

st.subheader("冷号")
st.info(cold)

st.subheader("高遗漏号码")
st.warning(missing)

st.subheader("AI杀号")
st.error(kill_nums)

st.subheader("AI综合推荐")
st.success(
    " / ".join([f"{n:02d}" for n in smart_nums[:10]])
)
st.subheader("AI胆码")

danma = smart_nums[:3]

st.warning(
    " / ".join([f"{n:02d}" for n in danma])
)

# =========================
# 2中2
# =========================

result_2in2 = predict_2in2(history_data)

st.subheader("2中2推荐")

for x in result_2in2:
    st.success(" / ".join([f"{n:02d}" for n in x]))

# =========================
# 3中3
# =========================

result_3in3 = predict_3in3(history_data)

st.subheader("3中3推荐")

for x in result_3in3:
    st.success(" / ".join([f"{n:02d}" for n in x]))

# =========================
# AI预测记录保存（自学习）
# =========================

history_file = "history_predictions.json"

save_data = {
    "time": str(datetime.now()),
    "expect": latest_item["expect"],
    "hot": hot,
    "cold": cold,
    "missing": missing,
    "smart": smart_nums,
    "2in2": result_2in2,
    "3in3": result_3in3
}

# 读取旧记录
old_data = []

if os.path.exists(history_file):

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            old_data = json.load(f)

    except:
        old_data = []

# 添加新预测
old_data.append(save_data)

# 最多保存300期
old_data = old_data[-300:]

# 写入
with open(history_file, "w", encoding="utf-8") as f:

    json.dump(
        old_data,
        f,
        ensure_ascii=False,
        indent=4
    )