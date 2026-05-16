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
    page_title="澳门六合彩 AI 超级智能分析系统",
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
# 获取历史
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
# 超级AI预测
# =========================
def ai_super_predict(df, history_count=10):

    recent_df = df.head(history_count)

    score_map = {n:0 for n in range(1,50)}

    zodiac_score = defaultdict(float)

    wave_score = defaultdict(float)

    total = len(recent_df)

    # =====================
    # 主分析
    # =====================
    for idx, row in recent_df.iterrows():

        weight = (total - idx) / total

        normal_nums = row["normal"]

        special = row["special"]

        # -----------------
        # 6区
        # -----------------
        for n in normal_nums:

            score_map[n] += 2.5 * weight

            zodiac_score[get_zodiac(n)] += 1.5 * weight

            wave_score[get_wave(n)] += 1.2 * weight

        # -----------------
        # 特码
        # -----------------
        score_map[special] += 5 * weight

        zodiac_score[get_zodiac(special)] += 4 * weight

        wave_score[get_wave(special)] += 3 * weight

    # =====================
    # 遗漏分析
    # =====================
    history_nums = df["nums"].tolist()

    for n in range(1,50):

        miss = 0

        for row in history_nums:

            if n in row:
                break

            miss += 1

        if miss >= 5:
            score_map[n] += miss * 0.8

    # =====================
    # 最近一期降温
    # =====================
    latest_nums = recent_df.iloc[0]["nums"]

    for n in latest_nums:

        score_map[n] -= 4

    # =====================
    # 最近3期重复降温
    # =====================
    last3 = []

    for row in recent_df.head(3)["nums"]:

        last3.extend(row)

    counter = Counter(last3)

    for n, c in counter.items():

        if c >= 2:
            score_map[n] -= c * 2

    # =====================
    # 波色回补
    # =====================
    weak_wave = min(wave_score, key=wave_score.get)

    for n in range(1,50):

        if get_wave(n) == weak_wave:
            score_map[n] += 3

    # =====================
    # 生肖回补
    # =====================
    weak_zodiac = min(zodiac_score, key=zodiac_score.get)

    for n in range(1,50):

        if get_zodiac(n) == weak_zodiac:
            score_map[n] += 3

    # =====================
    # 最终排序
    # =====================
    final_rank = sorted(
        score_map.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top号码
    top_numbers = [x[0] for x in final_rank[:10]]

    # Top生肖
    zodiac_rank = sorted(
        zodiac_score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top波色
    wave_rank = sorted(
        wave_score.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "numbers": top_numbers,
        "zodiac": zodiac_rank,
        "wave": wave_rank,
        "detail": final_rank[:15]
    }

# =========================
# 页面
# =========================
st.title("🔥 澳门六合彩 AI 超级智能分析系统")

history_count = st.sidebar.slider(
    "分析最近期数",
    5,
    15,
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
    f"期号：{latest_item['expect']}"
)

st.success(
    f"号码：{latest_item['openCode']}"
)

latest_nums = [
    int(x)
    for x in latest_item["openCode"].split(",")
]

info_text = []

for n in latest_nums:

    info_text.append(
        f"{n}({get_zodiac(n)}/{get_wave(n)}波)"
    )

st.info(" | ".join(info_text))

# =========================
# AI预测
# =========================
result = ai_super_predict(
    df,
    history_count
)

# =========================
# AI号码
# =========================
st.header("🔥 AI最强号码")

st.success(
    " / ".join([
        f"{n:02d}"
        for n in result["numbers"]
    ])
)

# =========================
# AI生肖
# =========================
st.header("🐲 AI最强生肖")

top_zodiac = result["zodiac"][:5]

for z, s in top_zodiac:

    st.info(
        f"{z}  →  AI评分：{round(s,2)}"
    )

# =========================
# AI波色
# =========================
st.header("🌈 AI最强波色")

top_wave = result["wave"][:3]

for w, s in top_wave:

    st.warning(
        f"{w}波  →  AI评分：{round(s,2)}"
    )

# =========================
# AI评分详情
# =========================
st.header("📊 AI号码评分详情")

detail_df = pd.DataFrame(
    result["detail"],
    columns=["号码", "AI评分"]
)

detail_df["生肖"] = detail_df["号码"].apply(get_zodiac)

detail_df["波色"] = detail_df["号码"].apply(get_wave)

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
    lambda x: " ".join([f"{n:02d}" for n in x])
)

st.dataframe(show_df)

# =========================
# 提示
# =========================
st.caption("系统每60秒自动刷新一次")