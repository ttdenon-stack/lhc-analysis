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
# 1. 页面配置与专业级 UI 常量
# =====================================================
st.set_page_config(
    page_title="AI 超级智能数据透视系统 Ultimate",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 AI 超级智能数据分析系统 Ultimate")

LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
AI_FILE = "ai_learn.json"

# --- 核心属性字典库 ---
RED = {1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46}
BLUE = {3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48}
GREEN = {5, 6, 11, 16, 17, 21, 22, 27, 28, 32, 33, 38, 39, 43, 44, 49}

ELEMENTS = {
    "金": [1, 2, 9, 10, 17, 18, 25, 26, 33, 34, 41, 42, 49],
    "木": [5, 6, 13, 14, 21, 22, 29, 30, 37, 38, 45, 46],
    "水": [11, 12, 19, 20, 27, 28, 35, 36, 43, 44],
    "火": [3, 4, 15, 16, 23, 24, 31, 32, 39, 40, 47, 48],
    "土": [7, 8]
}

ZODIAC_MAP = {
    "鼠": [7, 19, 31, 43], "牛": [6, 18, 30, 42], "虎": [5, 17, 29, 41],
    "兔": [4, 16, 28, 40], "龙": [3, 15, 27, 39], "蛇": [2, 14, 26, 38],
    "马": [1, 13, 25, 37, 49], "羊": [12, 24, 36, 48], "猴": [11, 23, 35, 47],
    "鸡": [10, 22, 34, 46], "狗": [9, 21, 33, 45], "猪": [8, 20, 32, 44]
}

# =====================================================
# 2. 映射工具函数 (内存优化闭包)
# =====================================================
def get_wave(num):
    return "红" if num in RED else ("蓝" if num in BLUE else "绿")

def get_zodiac(num):
    return next((z for z, nums in ZODIAC_MAP.items() if num in nums), "未知")

def get_element(num):
    return next((e, nums for e, nums in ELEMENTS.items() if num in nums), "未知")[0] if num <= 49 else "未知"

def get_tail(num):
    return num % 10

# =====================================================
# 3. 动态权重自适应系统
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
# 4. 健壮的网络请求与清洗引擎
# =====================================================
@st.cache_data(ttl=60, show_spinner=False)
def fetch_latest():
    try:
        r = requests.get(LATEST_API, headers=HEADERS, timeout=8)
        data = r.json()
        return data if isinstance(data, list) else [data]
    except:
        return []

@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(year):
    try:
        r = requests.get(HISTORY_API.format(year), headers=HEADERS, timeout=8)
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
                "expect": str(item["expect"]),
                "time": item["openTime"],
                "nums": nums,
                "normal": nums[:6],
                "special": nums[-1]
            })
        except:
            pass
    return pd.DataFrame(rows)

# =====================================================
# 5. 核心统计算法模型 (安全无递归版)
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
    total = sum(freq.values()) or 1
    for n in range(1, 50):
        p = (freq[n] + 1) / (total + 49)
        score[n] += p * 100 * weights.get("bayes", 1.0)

def yixiao_model(df, history_count):
    recent = df.head(history_count)
    zodiac_score = Counter()
    for idx, row in recent.iterrows():
        decay = (history_count - idx) / history_count
        for n in row["normal"]: zodiac_score[get_zodiac(n)] += 1.2 * decay
        zodiac_score[get_zodiac(row["special"])] += 2.5 * decay
    
    last3 = recent.head(3)
    repeat_counter = Counter([get_zodiac(row["special"]) for _, row in last3.iterrows()])
    for z, c in repeat_counter.items():
        if c >= 2: zodiac_score[z] -= 4 * c
    for z in ZODIAC_MAP: zodiac_score[z] += 2.0
    return [x[0] for x in sorted(zodiac_score.items(), key=lambda x: x[1], reverse=True)[:5]]

# =====================================================
# 6. AI 核心计算引擎
# =====================================================
def ai_engine(df, history_count=12, learning=True):
    weights = load_ai()
    recent = df.head(history_count)
    score = defaultdict(float)
    
    freq, trend = Counter(), Counter()
    zodiac_count, wave_count, tail_count, element_count = Counter(), Counter(), Counter(), Counter()
    total = max(len(recent), 1)

    # 基础特征与趋势统计
    for idx, row in recent.iterrows():
        decay = math.exp(-(idx / total))
        for n in row["normal"]:
            freq[n] += 1
            trend[n] += 1 if idx < 3 else 0 
            score[n] += 2.5 * decay  
            zodiac_count[get_zodiac(n)] += 1
            wave_count[get_wave(n)] += 1
            tail_count[get_tail(n)] += 1
            element_count[get_element(n)] += 1
        score[row["special"]] += weights["special"] * decay

    all_nums = df["nums"].tolist()
    
    # 评分主观引擎
    for n in range(1, 50):
        miss = next((i for i, row in enumerate(all_nums) if n in row), 30)
        score[n] += min(miss, 18) * weights["miss"]
        if freq[n] >= 4: score[n] -= freq[n] * weights["hot"]
        if freq[n] == 0: score[n] += weights["cold"]
        if trend[n] >= 3: score[n] += 1.5
        elif trend[n] == 0: score[n] += 2.2
        elif trend[n] == 1: score[n] += 0.8

    # 属性轮动补偿
    for c_dict, w_key, g_func in [(wave_count, "wave", get_wave), (zodiac_count, "zodiac", get_zodiac), (tail_count, "tail", get_tail), (element_count, "element", get_element)]:
        if c_dict:
            weak = min(c_dict, key=c_dict.get)
            for n in range(1, 50):
                if g_func(n) == weak: score[n] += weights[w_key]

    cycle_model(df, score, weights)
    bayes_model(df, score, weights)

    # 连码加成
    for row in recent["normal"]:
        nums = sorted(row)
        for i in range(len(nums)-1):
            if nums[i+1] - nums[i] == 1:
                score[nums[i]] += weights["consecutive"]
                score[nums[i+1]] += weights["consecutive"]

    # 智能杀码筛选
    latest_nums = recent.iloc[0]["nums"] if not recent.empty else []
    kill = []
    for n in range(1, 50):
        penalty = 5 if n in latest_nums else 0
        penalty += 4 if freq[n] >= 6 else 0
        if penalty >= 5:
            kill.append(n)
            score[n] -= penalty
        score[n] = max(score[n], 0.1)

    final_rank = sorted(score.items(), key=lambda x: x[1], reverse=True)
    numbers = [n for n, _ in final_rank if n not in kill][:12]

    # 特码独立筛选网络
    special_score = {n: score[n] for n in range(1, 50)}
    if not recent.empty:
        for n in range(1, 50):
            if n in latest_nums: special_score[n] -= 6
            if get_tail(n) == get_tail(recent.iloc[0]["special"]): special_score[n] -= 3
            if get_wave(n) == get_wave(recent.iloc[0]["special"]): special_score[n] -= 2
    special = [x[0] for x in sorted(special_score.items(), key=lambda x: x[1], reverse=True)[:8]]

    # 连尾计算
    tail_rank = Counter([get_tail(n) for row in recent["nums"] for n in row])
    tails = [x[0] for x in tail_rank.most_common(5)]

    # 概率转化
    total_score = sum(score.values()) or 1
    prob = {n: round((score[n] / total_score) * 100, 2) for n in range(1, 50)}

    # 自学习权重迭代
    if learning and len(df) > history_count + 2:
        try:
            future_real = df.iloc[0]["normal"]
            old_predict = numbers[:10]
            hit = len(set(old_predict) & set(future_real))

            if hit >= 3:
                weights["miss"] += 0.03
                weights["cold"] += 0.03
                weights["bayes"] += 0.02
                weights["cycle"] += 0.02
            elif hit <= 1:
                weights["hot"] -= 0.02
                weights["special"] -= 0.01

            for k in weights:
                weights[k] = round(max(0.5, min(weights[k], 8)), 2)
            save_ai(weights)
        except:
            pass

    return {
        "numbers": numbers,
        "danma": [n for i, n in enumerate(numbers) if get_wave(n) not in [get_wave(x) for x in numbers[:i]]][:5],
        "kill": kill[:6],
        "special": special,
        "prob": prob,
        "combo2": list(combinations(numbers[:8], 2))[:10],
        "combo3": list(combinations(numbers[:8], 3))[:10],
        "tails": tails,
        "yixiao": yixiao_model(df, history_count),
        "detail": final_rank[:20],
        "weights": weights,
        "raw_score": score
    }

# =====================================================
# 多模型融合与命中率回测引擎
# =====================================================
def ensemble_engine(df):
    final_score = defaultdict(float)
    for win, w in zip([8, 12, 20], [0.5, 0.3, 0.2]):
        result = ai_engine(df, history_count=win, learning=False)
        for n, p in result["raw_score"].items(): final_score[n] += p * w
    return final_score

def backtest(df, history_count=12, test_period=15):
    hit_history = []
    max_test = min(test_period, len(df)-2)
    for i in range(max_test):
        test_df = df.iloc[i+1:].reset_index(drop=True)
        result = ai_engine(test_df, history_count=history_count, learning=False)
        predict = result["numbers"][:10]
        real = df.iloc[i]["normal"]
        hit = len(set(predict) & set(real))
        hit_history.append(hit)
    # 将列表倒序，使得图表从左到右代表从过去到现在的时间流向
    return hit_history[::-1] 

# =====================================================
# 7. 可视化交互与前端渲染
# =====================================================
st.sidebar.title("⚙️ 数据控制中心")
history_count = st.sidebar.slider("分析数据深度 (期)", 1, 30, 15, 1)
if st.sidebar.button("🔄 同步并清理数据缓存", use_container_width=True):
    st.cache_data.clear()

with st.spinner("系统正在进行高并发数据融合分析..."):
    latest = fetch_latest()
    history = fetch_history(datetime.now().year)

if not latest:
    st.error("网络通信拦截：无法获取最新标量数据。")
    st.stop()

latest_item = sorted(latest, key=lambda x: x["expect"], reverse=True)[0]
seen, clean = set(), []
for item in history:
    if item["expect"] not in seen:
        seen.add(item["expect"])
        clean.append(item)
if latest_item["expect"] not in seen: clean.insert(0, latest_item)

df = parse_history(clean)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values(by="time", ascending=False).reset_index(drop=True)

# 核心推理计算与回测
ensemble_score = ensemble_engine(df)
res = ai_engine(df, history_count, learning=True)
hit_history = backtest(df, history_count, 15)

for n in res["prob"]:
    res["prob"][n] = round(res["prob"][n] + (ensemble_score[n] / 100), 2)

# ====== 界面呈现 ======
st.markdown("### 🔔 最新开奖核对")
m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("当期期号", latest_item['expect'])
m_col2.metric("开奖时间", latest_item['openTime'][:16])
m_col3.metric("基础开奖结构", f"{latest_item['openCode']}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 核心指标体系", "🔬 组合衍生与回测图表", "⚙️ 模型健康度与底表"])

with tab1:
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.subheader("高频推荐池 (Top 12)")
        st.success("  |  ".join([f"**{n:02d}**" for n in res["numbers"]]))
        
        st.subheader("平特一肖雷达")
        sz_cols = st.columns(5)
        for i, z in enumerate(res["yixiao"]):
            sz_cols[i].info(f"**{z}**\n\n{' '.join([f'{n:02d}' for n in ZODIAC_MAP[z]])}")

    with col_b:
        st.subheader("分流指引")
        st.warning(f"**核心主胆**: {' / '.join([f'{n:02d}' for n in res['danma']])}")
        st.error(f"**智能杀码**: {' / '.join([f'{n:02d}' for n in res['kill']]) if res['kill'] else '暂无强烈杀码'}")
        st.error(f"**特码建议**: {' / '.join([f'{n:02d}' for n in res['special']])}")
        st.warning(f"**连尾预测**: {' / '.join([str(x) for x in res['tails']])}")

with tab2:
    st.subheader("📈 AI 历史模拟命中率追踪 (近15期趋势)")
    chart_df = pd.DataFrame({"命中数量": hit_history})
    st.line_chart(chart_df, height=200)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("🥈 组合二中二 (系统前10优选)")
        for item in res["combo2"]:
            st.code(" - ".join([f"{x:02d}" for x in item]))
    with col_c2:
        st.subheader("🥉 组合三中三 (系统前10优选)")
        for item in res["combo3"]:
            st.code(" - ".join([f"{x:02d}" for x in item]))

with tab3:
    c_tab1, c_tab2, c_tab3 = st.columns(3)
    
    with c_tab1:
        st.subheader("权重分布反馈")
        weight_df = pd.DataFrame({"特征模型": list(res["weights"].keys()), "当前权重": list(res["weights"].values())})
        st.dataframe(weight_df, use_container_width=True, hide_index=True)
        
    with c_tab2:
        st.subheader("概率透视详表 (前20)")
        detail_df = pd.DataFrame(res["detail"], columns=["号码", "综合得分"])
        detail_df["生肖"] = detail_df["号码"].apply(get_zodiac)
        detail_df["波色"] = detail_df["号码"].apply(get_wave)
        detail_df["综合概率"] = detail_df["号码"].apply(lambda x: f"{res['prob'][x]} %")
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    with c_tab3:
        st.subheader("底层数据框追踪")
        st.dataframe(df[["expect", "openCode"]].head(15), use_container_width=True, hide_index=True)