import json
import os
import requests
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
import math

# =====================================================
# 1. 页面配置与基础常量
# =====================================================
st.set_page_config(
    page_title="AI 超级智能数据分析系统 Ultimate",
    layout="wide"
)

st.title("📊 AI 超级智能数据分析系统 Ultimate")

LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
AI_FILE = "ai_learn.json"

# 波色定义
RED = {1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46}
BLUE = {3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48}
GREEN = {5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49}

# 五行定义
ELEMENTS = {
    "金": [1, 2, 9, 10, 17, 18, 25, 26, 33, 34, 41, 42, 49],
    "木": [5, 6, 13, 14, 21, 22, 29, 30, 37, 38, 45, 46],
    "水": [11, 12, 19, 20, 27, 28, 35, 36, 43, 44],
    "火": [3, 4, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48],
    "土": [7, 8]
}

# 生肖定义
ZODIAC_MAP = {
    "鼠": [7, 19, 31, 43], "牛": [6, 18, 30, 42], "虎": [5, 17, 29, 41],
    "兔": [4, 16, 28, 40], "龙": [3, 15, 27, 39], "蛇": [2, 14, 26, 38],
    "马": [1, 13, 25, 37, 49], "羊": [12, 24, 36, 48], "猴": [11, 23, 35, 47],
    "鸡": [10, 22, 34, 46], "狗": [9, 21, 33, 45], "猪": [8, 20, 32, 44]
}

# =====================================================
# 2. 属性特征映射工具函数
# =====================================================
def get_wave(num):
    if num in RED: return "红"
    if num in BLUE: return "蓝"
    return "绿"

def get_zodiac(num):
    for z, nums in ZODIAC_MAP.items():
        if num in nums: return z
    return "未知"

def get_element(num):
    for e, nums in ELEMENTS.items():
        if num in nums: return e
    return "未知"

def get_tail(num):
    return num % 10

def get_zone(num):
    if num <= 16: return "低区"
    elif num <= 33: return "中区"
    return "高区"

# =====================================================
# 3. AI 权重配置文件管理
# =====================================================
def load_ai():
    default = {
        "miss": 2.0, "hot": 1.8, "cold": 2.5, "wave": 2.0, "zodiac": 2.0,
        "tail": 2.0, "zone": 1.5, "element": 2.0, "consecutive": 2.5,
        "special": 5.0, "bayes": 2.0, "cycle": 2.5
    }
    if not os.path.exists(AI_FILE):
        save_ai(default)
        return default
    try:
        with open(AI_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default.items():
            if k not in data: data[k] = v
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
# 4. API 数据获取与清洗
# =====================================================
@st.cache_data(ttl=120)
def fetch_latest():
    try:
        r = requests.get(LATEST_API, headers=HEADERS, timeout=10)
        data = r.json()
        return data if isinstance(data, list) else [data]
    except:
        return []

@st.cache_data(ttl=300)
def fetch_history(year):
    try:
        url = HISTORY_API.format(year)
        r = requests.get(url, headers=HEADERS, timeout=10)
        return r.json().get("data", [])
    except:
        return []

def parse_history(data):
    rows = []
    for item in data:
        try:
            nums = [int(x) for x in item["openCode"].split(",")]
            if len(nums) != 7: continue
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
# 5. 核心算法子模型
# =====================================================
def cycle_model(df, score, weights):
    periods = [3, 5, 8]
    for p in periods:
        recent = df.head(p)
        count = Counter([n for row in recent["nums"] for n in row])
        weak = min(count.values()) if count else 0
        for n in range(1, 50):
            if count[n] <= weak:
                score[n] += 1.2 * weights.get("cycle", 1.0)

def bayes_model(df, score, weights):
    recent = df.head(20)
    freq = Counter([n for row in recent["nums"] for n in row])
    total = sum(freq.values())
    for n in range(1, 50):
        p = (freq[n] + 1) / (total + 49)
        score[n] += p * 100 * weights.get("bayes", 1.0)

def yixiao_model(df, history_count):
    recent = df.head(history_count)
    zodiac_score = Counter()
    
    for idx, row in recent.iterrows():
        decay = (history_count - idx) / history_count
        for n in row["normal"]:
            zodiac_score[get_zodiac(n)] += 1.2 * decay
        zodiac_score[get_zodiac(row["special"])] += 2.5 * decay

    last3 = recent.head(3)
    repeat_counter = Counter([get_zodiac(row["special"]) for _, row in last3.iterrows()])
    for z, c in repeat_counter.items():
        if c >= 2: zodiac_score[z] -= 4 * c

    for z in ZODIAC_MAP:
        zodiac_score[z] += 2.0

    result = sorted(zodiac_score.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in result[:5]]

# =====================================================
# 6. AI 核心引擎
# =====================================================
def ai_engine(df, history_count=12):
    weights = load_ai()
    recent = df.head(history_count)
    score = defaultdict(float)
    
    freq = Counter()
    zodiac_count, wave_count, tail_count, element_count = Counter(), Counter(), Counter(), Counter()
    total = max(len(recent), 1)

    # 基础特征统计
    for idx, row in recent.iterrows():
        decay = math.exp(-(idx / total))
        for n in row["normal"]:
            freq[n] += 1
            score[n] += 2.5 * decay  # 平衡初始分，避免后期杀码冲突
            zodiac_count[get_zodiac(n)] += 1
            wave_count[get_wave(n)] += 1
            tail_count[get_tail(n)] += 1
            element_count[get_element(n)] += 1
        score[row["special"]] += weights["special"] * decay

    # 遗漏分析
    all_nums = df["nums"].tolist()
    for n in range(1, 50):
        miss = 0
        for row in all_nums:
            if n in row: break
            miss += 1
        score[n] += min(miss, 18) * weights["miss"]

    # 冷热调控
    for n in range(1, 50):
        if freq[n] >= 4: score[n] -= freq[n] * weights["hot"]
        if freq[n] == 0: score[n] += weights["cold"]

    # 属性轮动补偿
    for count_dict, weight_key, get_func in [(wave_count, "wave", get_wave), (zodiac_count, "zodiac", get_zodiac), (tail_count, "tail", get_tail), (element_count, "element", get_element)]:
        if count_dict:
            weak = min(count_dict, key=count_dict.get)
            for n in range(1, 50):
                if get_func(n) == weak: score[n] += weights[weight_key]

    # 调用子模型
    cycle_model(df, score, weights)
    bayes_model(df, score, weights)

    # 连码加成
    for row in recent["normal"]:
        nums = sorted(row)
        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] == 1:
                score[nums[i]] += weights["consecutive"]
                score[nums[i+1]] += weights["consecutive"]

    # 智能杀码筛选 (优化：严格阈值，防止误杀高分号)
    latest_nums = recent.iloc[0]["nums"] if not recent.empty else []
    kill = []
    for n in range(1, 50):
        penalty = 0
        if n in latest_nums: penalty += 5
        if freq[n] >= 5: penalty += 4
        if penalty >= 5:
            kill.append(n)
            score[n] -= penalty

    for n in range(1, 50):
        score[n] = max(score[n], 0.1)

    final_rank = sorted(score.items(), key=lambda x: x[1], reverse=True)
    numbers = [n for n, _ in final_rank if n not in kill][:12]

    # 胆码分散选择
    danma = []
    used_wave, used_zodiac = set(), set()
    for n in numbers:
        w, z = get_wave(n), get_zodiac(n)
        if w not in used_wave and z not in used_zodiac:
            danma.append(n)
            used_wave.add(w)
            used_zodiac.add(z)
        if len(danma) >= 4: break

    # 特码独立模型
    special_score = {n: score[n] for n in range(1, 50)}
    if not recent.empty:
        for n in range(1, 50):
            if n in latest_nums: special_score[n] -= 6
            if get_tail(n) == get_tail(recent.iloc[0]["special"]): special_score[n] -= 3
    special = [x[0] for x in sorted(special_score.items(), key=lambda x: x[1], reverse=True)[:8]]

    # 概率转化与辅助输出
    total_score = sum(score.values())
    prob = {n: round(score[n] / total_score * 100, 2) for n in range(1, 50)}
    
    tail_rank = Counter([get_tail(n) for row in recent["nums"] for n in row])
    tails = [x[0] for x in tail_rank.most_common(5)]
    yixiao = yixiao_model(df, history_count)
    combo2 = list(combinations(numbers[:8], 2))[:10]
    combo3 = list(combinations(numbers[:8], 3))[:10]

    # 反馈自学习机制
    try:
        if len(recent) > 1:
            last_real = recent.iloc[0]["nums"]
            prev_df = df.iloc[1:].reset_index(drop=True)
            prev_predict = ai_engine(prev_df, history_count)["numbers"][:10]
            hit = len(set(prev_predict) & set(last_real))
            if hit >= 3:
                weights["miss"] = min(weights["miss"] + 0.02, 5.0)
                weights["cold"] = min(weights["cold"] + 0.02, 5.0)
            else:
                weights["hot"] = max(weights["hot"] - 0.01, 0.5)
            save_ai(weights)
    except:
        pass

    return {
        "numbers": numbers, "danma": danma, "kill": kill[:6], "special": special,
        "prob": prob, "combo2": combo2, "combo3": combo3, "tails": tails,
        "yixiao": yixiao, "detail": final_rank[:20], "weights": weights
    }

# =====================================================
# 7. 页面渲染与交互逻辑
# =====================================================
if st.button("🔄 手动刷新最新数据"):
    st.cache_data.clear()

history_count = st.sidebar.slider("分析最近期数", min_value=5, max_value=25, value=12, step=1)

with st.spinner("AI数据大模型深度分析中..."):
    latest = fetch_latest()
    history = fetch_history(datetime.now().year)

if not latest:
    st.error("无法获取最新奖项数据，请检查网络连接。")
    st.stop()

latest_item = sorted(latest, key=lambda x: x["expect"], reverse=True)[0]
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
    st.error("历史数据解析失败")
    st.stop()

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(by="time", ascending=False).reset_index(drop=True)

# 运算生成结果
result = ai_engine(df, history_count)

# --- 布局展现 ---
col1, col2 = st.columns(2)
with col1:
    st.header("🔔 最新开奖结果")
    st.success(f"期号 {latest_item['expect']} ➡️ 开奖号码：{latest_item['openCode']}")
with col2:
    st.header("🎯 AI 核心推荐号 (12码)")
    st.success(" / ".join([f"{n:02d}" for n in result["numbers"]]))

st.markdown("---")

col3, col4, col5 = st.columns(3)
with col3:
    st.header("⭐ 核心胆码")
    st.warning(" / ".join([f"{n:02d}" for n in result["danma"]]))
with col4:
    st.header("🔮 连尾预测")
    st.warning(" / ".join([str(x) for x in result["tails"]]))
with col5:
    st.header("❌ 智能杀码")
    st.error(" / ".join([f"{n:02d}" for n in result["kill"]]) if result["kill"] else "暂无建议杀码")

st.markdown("---")

st.header("🐅 平特一肖方案")
sz_cols = st.columns(5)
for i, z in enumerate(result["yixiao"]):
    with sz_cols[i]:
        nums = " ".join([f"{n:02d}" for n in ZODIAC_MAP[z]])
        st.info(f"**{z}**\n\n{nums}")

st.markdown("---")

col_c1, col_c2 = st.columns(2)
with col_c1:
    st.header("🥈 组合二中二 (前10组)")
    for item in result["combo2"]:
        st.code(" - ".join([f"{x:02d}" for x in item]))
with col_c2:
    st.header("🥉 组合三中三 (前10组)")
    for item in result["combo3"]:
        st.code(" - ".join([f"{x:02d}" for x in item]))

st.markdown("---")

st.header("💎 独立特码建议")
st.error(" / ".join([f"{n:02d}" for n in result["special"]]))

st.markdown("---")

st.header("📊 候选号码综合特征评分")
detail_df = pd.DataFrame(result["detail"], columns=["号码", "评分"])
detail_df["生肖"] = detail_df["号码"].apply(get_zodiac)
detail_df["波色"] = detail_df["号码"].apply(get_wave)
detail_df["五行"] = detail_df["号码"].apply(get_element)
detail_df["综合概率%"] = detail_df["号码"].apply(lambda x: result["prob"][x])
st.dataframe(detail_df, use_container_width=True)

st.markdown("---")

st.header("📈 权重分布反馈")
weight_df = pd.DataFrame({"特征模型": list(result["weights"].keys()), "当前权重分": list(result["weights"].values())})
st.dataframe(weight_df, use_container_width=True)
