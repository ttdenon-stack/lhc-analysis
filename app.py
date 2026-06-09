import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import os
import json

from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations

# =====================================================
# 页面配置
# =====================================================

st.set_page_config(
    page_title="LHC Ultimate V2",
    layout="wide",
    page_icon="🎯"
)

st.title("🎯 Ultimate V2")

st.caption(
    "贝叶斯 + 马尔可夫链 + 遗漏周期 + 冷热轮动 + 生肖轮动 + 多窗口融合"
)

# =====================================================
# API
# =====================================================

LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_YEAR_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"
HISTORY_EXPECT_API = "https://history.macaumarksix.com/history/macaujc2/expect/{}"

# 备用数据源，可按需扩展
API_SOURCES = {
    'latest': [
        "https://macaumarksix.com/api/macaujc2.com",
    ],
    'history': [
        "https://history.macaumarksix.com/history/macaujc2/y/{}",
    ],
    'expect': [
        "https://history.macaumarksix.com/history/macaujc2/expect/{}",
    ]
}

HEADERS = {
    "User-Agent":
    "Mozilla/5.0"
}

LOG_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.csv")

# =====================================================
# 波色
# =====================================================

RED = {
    1,2,7,8,12,13,18,19,
    23,24,29,30,34,35,
    40,45,46
}

BLUE = {
    3,4,9,10,14,15,
    20,25,26,31,36,37,
    41,42,47,48
}

GREEN = {
    5,6,11,16,17,
    21,22,27,28,
    32,33,38,39,
    43,44,49
}

# =====================================================
# 生肖
# =====================================================

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

# =====================================================
# 五行
# =====================================================

ELEMENTS = {

    "金":[1,2,9,10,17,18,25,26,33,34,41,42,49],

    "木":[5,6,13,14,21,22,29,30,37,38,45,46],

    "水":[11,12,19,20,27,28,35,36,43,44],

    "火":[3,4,15,16,23,24,31,32,39,40,47,48],

    "土":[7,8]
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
# API获取
# =====================================================

@st.cache_data(ttl=60)
def fetch_latest():

    for url in API_SOURCES.get('latest', []):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
            return [data]
        except Exception:
            continue

    return []


@st.cache_data(ttl=300)
def fetch_history():
    year = datetime.now().year

    # 先尝试当年历史数据
    for offset in range(0, 3):
        try_year = year - offset
        for template in API_SOURCES.get('history', []):
            try:
                url = template.format(try_year)
                r = requests.get(url, headers=HEADERS, timeout=20)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict):
                    parsed = data.get('data', [])
                else:
                    parsed = data
                if parsed:
                    return parsed
            except Exception:
                continue

    # 历史接口不可用时，退回最新数据
    latest = fetch_latest()
    return latest


@st.cache_data(ttl=300)
def fetch_record_by_issue(issue):
    """按期号查询单期记录"""
    for template in API_SOURCES.get('expect', []):
        try:
            url = template.format(issue)
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            return data.get('data', data) if isinstance(data, dict) else data
        except Exception:
            continue
    return None

    # =====================================================
# 贝叶斯模型
# =====================================================

def bayes_score(df):

    score = defaultdict(float)

    recent = df.head(20)

    freq = Counter()

    total = 0

    for row in recent["nums"]:

        for n in row:

            freq[n] += 1
            total += 1

    for n in range(1,50):

        p = (freq[n] + 1) / (total + 49)

        score[n] = p * 100

    return score


# =====================================================
# 遗漏模型
# =====================================================

def miss_score(df):

    score = defaultdict(float)

    rows = df["nums"].tolist()

    for n in range(1,50):

        miss = 0

        for row in rows:

            if n in row:
                break

            miss += 1

        score[n] = min(miss,20)

    return score


# =====================================================
# 冷热模型
# =====================================================

def hot_cold_score(df):

    score = defaultdict(float)

    recent = df.head(15)

    freq = Counter()

    for row in recent["nums"]:

        for n in row:

            freq[n] += 1

    for n in range(1,50):

        if freq[n] == 0:

            score[n] += 5

        elif freq[n] >= 4:

            score[n] -= freq[n]

    return score

# =====================================================
# 马尔可夫链（特码）
# =====================================================

def markov_special(df):

    transitions = defaultdict(int)

    recent = df.head(30)

    specials = recent["special"].tolist()

    for i in range(len(specials)-1):

        a = get_wave(specials[i])

        b = get_wave(specials[i+1])

        transitions[(a,b)] += 1

    current = get_wave(specials[0])

    target = defaultdict(int)

    for (a,b),v in transitions.items():

        if a == current:

            target[b] += v

    if not target:

        return None

    return max(
        target,
        key=target.get
    )

# =====================================================
# 生肖轮动
# =====================================================

def zodiac_rotation(df):

    score = defaultdict(float)

    recent = df.head(12)

    zodiac_count = Counter()

    for _,row in recent.iterrows():

        for n in row["nums"]:

            zodiac_count[
                get_zodiac(n)
            ] += 1

    weak = min(
        zodiac_count,
        key=zodiac_count.get
    )

    for n in range(1,50):

        if get_zodiac(n) == weak:

            score[n] += 4

    return score


# =====================================================
# 波色轮动
# =====================================================

def wave_rotation(df):

    score = defaultdict(float)

    recent = df.head(12)

    wave_count = Counter()

    for _,row in recent.iterrows():

        for n in row["nums"]:

            wave_count[
                get_wave(n)
            ] += 1

    weak = min(
        wave_count,
        key=wave_count.get
    )

    for n in range(1,50):

        if get_wave(n) == weak:

            score[n] += 3

    return score

# =====================================================
# 尾数轮动
# =====================================================

def tail_rotation(df):

    score = defaultdict(float)

    recent = df.head(12)

    tail_count = Counter()

    for _,row in recent.iterrows():

        for n in row["nums"]:

            tail_count[
                get_tail(n)
            ] += 1

    weak = min(
        tail_count,
        key=tail_count.get
    )

    for n in range(1,50):

        if get_tail(n) == weak:

            score[n] += 2

    return score

# =====================================================
# Ultimate V2 AI核心
# =====================================================

def generate_prediction(df, history_count=20):

    recent = df.head(history_count)

    final_score = defaultdict(float)

    bayes = bayes_score(df)

    miss = miss_score(df)

    hotcold = hot_cold_score(df)

    zodiac = zodiac_rotation(df)

    wave = wave_rotation(df)

    tail = tail_rotation(df)

    markov_wave = markov_special(df)

    for n in range(1,50):

        final_score[n] += bayes[n] * 2.5

        final_score[n] += miss[n] * 2.0

        final_score[n] += hotcold[n] * 2.0

        final_score[n] += zodiac[n] * 1.8

        final_score[n] += wave[n] * 1.5

        final_score[n] += tail[n] * 1.2

        if markov_wave:

            if get_wave(n) == markov_wave:

                final_score[n] += 5

    ranking = sorted(
        final_score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    numbers = [
        x[0]
        for x in ranking[:12]
    ]

    danma = numbers[:4]

    special = [
        x[0]
        for x in ranking[:8]
    ]

    kill = [
        x[0]
        for x in ranking[-6:]
    ]

    combo2 = list(
        combinations(
            numbers[:8],
            2
        )
    )[:10]

    combo3 = list(
        combinations(
            numbers[:8],
            3
        )
    )[:10]

    zodiac_rank = Counter()

    for n in numbers:

        zodiac_rank[
            get_zodiac(n)
        ] += 1

    yixiao = [

        x[0]

        for x in zodiac_rank.most_common(5)

    ]

    return {

        "numbers":numbers,

        "danma":danma,

        "special":special,

        "kill":kill,

        "combo2":combo2,

        "combo3":combo3,

        "yixiao":yixiao,

        "detail":ranking[:20]
    }


# =====================================================
# 数据解析与界面
# =====================================================


def parse_record(rec):
    """从 API 记录中解析出期号、日期、6个号码和特码（如有）。返回 dict 或 None。"""

    try:
        # 常见字段映射
        issue = rec.get("issue") or rec.get("expect") or rec.get("period") or rec.get("drawNo") or rec.get("qishu") or rec.get("期号")

        date = rec.get("date") or rec.get("openTime") or rec.get("drawDate") or rec.get("time") or rec.get("datetime") or rec.get("dateTime")

        # 号码：优先寻找 list 字段和 openCode 字符串
        nums = None

        if "nums" in rec and isinstance(rec["nums"], (list, tuple)):
            nums = [int(x) for x in rec["nums"]]

        if not nums and "openCode" in rec and isinstance(rec["openCode"], str):
            codes = [p for p in rec["openCode"].replace(';',',').split(',') if p.strip()]
            parsed = [int(''.join(ch for ch in p if ch.isdigit())) for p in codes]
            if parsed:
                nums = parsed

        if not nums and "numbers" in rec:
            v = rec["numbers"]
            if isinstance(v, str):
                parts = [p for p in v.replace('+',',').replace('\n',',').replace(';',',').split(',') if p.strip()]
                nums = [int(''.join(ch for ch in p if ch.isdigit())) for p in parts[:6]]
            elif isinstance(v, (list, tuple)):
                nums = [int(x) for x in v[:6]]

        # 支持 n1..n6 + special 的格式
        if not nums:
            keys = [f for f in rec.keys()]
            nkeys = [k for k in keys if k.lower().startswith('n') and k[1:].isdigit()]
            if nkeys:
                try:
                    nkeys_sorted = sorted(nkeys, key=lambda x: int(x[1:]))
                    nums = [int(rec[k]) for k in nkeys_sorted[:6]]
                except:
                    nums = None

        # 备用：result 字符串，如 "01 02 03 04 05 06 + 07"
        if not nums and "result" in rec and isinstance(rec["result"], str):
            parts = [p for p in rec["result"].replace('+',',').split() if p.strip()]
            nums = [int(''.join(ch for ch in p if ch.isdigit())) for p in parts[:6]]

        # 特码
        special = None
        for key in ("special", "s", "te", "特码", "sp"):
            if key in rec:
                try:
                    special = int(rec[key])
                    break
                except:
                    pass

        # 如果没有单独 special，尝试从 numbers 字符串末尾获取
        if special is None and "numbers" in rec and isinstance(rec["numbers"], str):
            s = rec["numbers"].split('+')
            if len(s) > 1:
                try:
                    special = int(''.join(ch for ch in s[1] if ch.isdigit()))
                except:
                    special = None

        if special is None and "openCode" in rec and isinstance(rec["openCode"], str):
            s = [p for p in rec["openCode"].split(',') if p.strip()]
            if len(s) >= 7:
                try:
                    special = int(''.join(ch for ch in s[6] if ch.isdigit()))
                except:
                    special = None

        # 如果仍然没有 special，但有 nums，取第7个（如果存在）
        if special is None and nums and len(nums) >= 7:
            special = nums[6]
            nums = nums[:6]

        if not nums or len(nums) < 6:
            return None

        return {
            "issue": issue,
            "date": date,
            "nums": nums,
            "special": int(special) if special is not None else nums[-1]
        }

    except Exception:
        return None


@st.cache_data(ttl=300)
def build_dataframe(records):
    parsed = []

    for r in records:
        p = parse_record(r)
        if p:
            parsed.append(p)

    if not parsed:
        return pd.DataFrame()

    df = pd.DataFrame(parsed)

    # 尝试按日期排序（降序），否则保留原始顺序并将最近的放在前面
    if "date" in df.columns and df["date"].notnull().any():
        try:
            df["_date_parsed"] = pd.to_datetime(df["date"], errors='coerce')
            df = df.sort_values("_date_parsed", ascending=False)
            df = df.drop(columns=["_date_parsed"])
        except:
            df = df.iloc[::-1].reset_index(drop=True)
    else:
        df = df.iloc[::-1].reset_index(drop=True)

    return df.reset_index(drop=True)


def show_numbers_list(nums):
    cols = st.columns(len(nums))
    for i, n in enumerate(nums):
        cols[i].metric(label=str(n), value=" ")


def save_prediction_log(log_row):
    """Append one prediction record to CSV log if not already recorded."""
    if not os.path.exists(LOG_PATH):
        df = pd.DataFrame([log_row])
        df.to_csv(LOG_PATH, index=False, encoding='utf-8-sig')
        return True

    try:
        existing = pd.read_csv(LOG_PATH, dtype=str)
        if str(log_row.get('issue')) in existing.get('issue', pd.Series([], dtype=str)).astype(str).values:
            return False
        existing = pd.concat([existing, pd.DataFrame([log_row])], ignore_index=True)
        existing.to_csv(LOG_PATH, index=False, encoding='utf-8-sig')
        return True
    except Exception:
        try:
            pd.DataFrame([log_row]).to_csv(LOG_PATH, index=False, encoding='utf-8-sig')
            return True
        except Exception:
            return False


def load_prediction_log():
    if os.path.exists(LOG_PATH):
        try:
            return pd.read_csv(LOG_PATH, dtype=str)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def main_ui():
    st.sidebar.header("设置")
    st.sidebar.markdown(
        "调整说明：\n"
        "- 历史窗口：参考最近多少期的数据来进行分析。\n"
        "- 权重：数值越大该特征越重要。\n"
        "- 组合分析：可以打开/关闭不同策略。\n"
        "- 回测：先用历史窗口验证命中，再按截止日期/期号模拟。"
    )
    history_count = st.sidebar.slider("历史窗口 (条)", min_value=10, max_value=200, value=20, step=10)

    # 模型权重（可调整以优化回测）
    st.sidebar.subheader("模型权重调整")
    bayes_w = st.sidebar.slider("贝叶斯权重", 0.0, 5.0, 2.5, 0.1)
    miss_w = st.sidebar.slider("遗漏权重", 0.0, 5.0, 2.0, 0.1)
    hotcold_w = st.sidebar.slider("冷热权重", 0.0, 5.0, 2.0, 0.1)
    zodiac_w = st.sidebar.slider("生肖权重", 0.0, 5.0, 1.8, 0.1)
    wave_w = st.sidebar.slider("波色权重", 0.0, 5.0, 1.5, 0.1)
    tail_w = st.sidebar.slider("尾数权重", 0.0, 3.0, 1.2, 0.1)
    markov_bonus = st.sidebar.slider("马尔可夫波色加分", 0.0, 10.0, 5.0, 0.5)

    # 分析选项
    st.sidebar.subheader("分析类型")
    analyze_special_flag = st.sidebar.checkbox("分析特码（仅最后一位）", value=True)
    analyze_pingte_flag = st.sidebar.checkbox("分析平特一肖（前6区/整区）", value=True)
    analyze_sanlianxiao_flag = st.sidebar.checkbox("分析平特三连肖（前6区）", value=True)
    analyze_erzher_flag = st.sidebar.checkbox("分析二中二（前6区）", value=True)
    analyze_danma_flag = st.sidebar.checkbox("胆码概率（全区）", value=True)

    st.sidebar.subheader("记录设置")
    auto_log = st.sidebar.checkbox("启用自动记录预测结果", value=False)
    show_log = st.sidebar.checkbox("显示历史预测记录", value=False)

    # 回测设置
    st.sidebar.subheader("回测")
    do_backtest = st.sidebar.checkbox("启用回测（对最近若干期进行回测）", value=False)
    backtest_rounds = st.sidebar.number_input("回测轮数（最近多少期）", min_value=5, max_value=500, value=50, step=5)

    raw = fetch_history()

    df = build_dataframe(raw)

    if df.empty:
        st.warning("未能解析到开奖记录，请检查 API 或切换数据源。")
        return

    st.subheader("最近开奖")
    latest = df.iloc[0]
    st.write("期号：", latest.get("issue"))
    st.write("开奖日期：", latest.get("date"))
    st.write("开奖号码：")
    show_numbers_list(latest["nums"]) 
    st.write("特码：", latest["special"]) 

    pred = generate_prediction(df, history_count=history_count)
    # 额外分析
    def score_fusion(df_window):
        # reuse existing generate_prediction internals but return full score dict
        bayes = bayes_score(df_window)
        miss = miss_score(df_window)
        hotcold = hot_cold_score(df_window)
        zodiac = zodiac_rotation(df_window)
        wave = wave_rotation(df_window)
        tail = tail_rotation(df_window)
        markov_wave = markov_special(df_window)

        final_score = defaultdict(float)


        for n in range(1,50):
            final_score[n] += bayes[n] * bayes_w
            final_score[n] += miss[n] * miss_w
            final_score[n] += hotcold[n] * hotcold_w
            final_score[n] += zodiac[n] * zodiac_w
            final_score[n] += wave[n] * wave_w
            final_score[n] += tail[n] * tail_w

            if markov_wave and get_wave(n) == markov_wave:
                final_score[n] += markov_bonus

        return final_score

    def analyze_special(df, history_count, method_pool=False):
        # method_pool: if True include all 7 numbers per period as pool, else only special field
        recent = df.head(history_count)
        freq = Counter()

        if method_pool:
            for _, row in recent.iterrows():
                for v in row["nums"]:
                    freq[v] += 1
                # include special as well
                freq[row.get("special", row["nums"][-1])] += 1
        else:
            for _, row in recent.iterrows():
                freq[row.get("special", row["nums"][-1])] += 1

        total = sum(freq.values()) or 1
        ranked = [(n, freq[n]/total*100) for n in range(1,50)]
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        return ranked

    def analyze_front6_freq(df, history_count):
        recent = df.head(history_count)
        freq = Counter()
        for _, row in recent.iterrows():
            for v in row["nums"][:6]:
                freq[v] += 1

        total = sum(freq.values()) or 1
        ranked = [(n, freq[n]/total*100) for n in range(1,50)]
        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        return ranked

    def analyze_pairs_front6(df, history_count, top_k=20):
        recent = df.head(history_count)
        pair_cnt = Counter()
        for _, row in recent.iterrows():
            nums = row["nums"][:6]
            for a, b in combinations(sorted(nums), 2):
                pair_cnt[(a,b)] += 1

        ranked = pair_cnt.most_common(top_k)
        return ranked

    def analyze_sanlianxiao(df, history_count):
        # identify zodiacs with 3-period consecutive appearance in front6
        recent = df.head(history_count)
        zseq = []
        for _, row in recent.iterrows():
            zset = set(get_zodiac(n) for n in row["nums"][:6])
            zseq.append(zset)

        consec = defaultdict(int)
        # sliding window for 3 consecutive
        for i in range(len(zseq)-2):
            common = zseq[i] & zseq[i+1] & zseq[i+2]
            for z in common:
                consec[z] += 1

        return sorted(consec.items(), key=lambda x: x[1], reverse=True)

    def rank_danma(df, history_count):
        scores = score_fusion(df.head(history_count))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    # run requested analyses
    results = {}

    st.subheader("预测结果（Top 候选）")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("**候选号码：**")
        st.markdown("<div style='font-size:24px; font-weight:bold; letter-spacing:0.3em;'>%s</div>" % " ".join(str(x) for x in pred["numbers"]), unsafe_allow_html=True)
        st.markdown("**单码（胆码）Top4：** %s" % ", ".join(str(x) for x in pred["danma"]))
        st.markdown("**特码候选 Top8：** %s" % ", ".join(str(x) for x in pred["special"]))
    with col_b:
        st.metric("最新期号", latest.get("issue", "N/A"))
        st.metric("特码（最后一位）", latest.get("special", "N/A"))
        st.metric("历史窗口", f"{history_count} 期")
        st.metric("当前模式", "实时分析")

    st.markdown("---")
    st.subheader("组合示例")
    combo_col1, combo_col2 = st.columns(2)
    with combo_col1:
        st.markdown("**2 选组合 Top10**")
        df_combo2 = pd.DataFrame(pred["combo2"][:10], columns=["号码1", "号码2"])
        st.table(df_combo2)
    with combo_col2:
        st.markdown("**3 选组合 Top10**")
        df_combo3 = pd.DataFrame(pred["combo3"][:10], columns=["号码1", "号码2", "号码3"])
        st.table(df_combo3)

    st.subheader("按分数的细节（前20）")
    detail = pred.get("detail", [])
    if detail:
        df_detail = pd.DataFrame(detail, columns=["num","score"]) if isinstance(detail[0], tuple) else pd.DataFrame(detail)
        st.dataframe(df_detail.style.format({"score":"{:.2f}"}), use_container_width=True)

    # 简单可视化分数热度
    scores = {n: s for n, s in pred.get("detail", [])}
    score_series = pd.Series([scores.get(i, 0) for i in range(1,50)], index=range(1,50))
    st.subheader("号码热度（分数）")
    st.bar_chart(score_series)

    if auto_log:
        log_row = {
            'log_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'issue': latest.get('issue'),
            'date': latest.get('date'),
            'history_count': history_count,
            'bayes_w': bayes_w,
            'miss_w': miss_w,
            'hotcold_w': hotcold_w,
            'zodiac_w': zodiac_w,
            'wave_w': wave_w,
            'tail_w': tail_w,
            'markov_bonus': markov_bonus,
            'analysis_特码': analyze_special_flag,
            'analysis_平特一肖': analyze_pingte_flag,
            'analysis_三连肖': analyze_sanlianxiao_flag,
            'analysis_二中二': analyze_erzher_flag,
            'analysis_胆码': analyze_danma_flag,
            'candidates': json.dumps(pred.get('numbers', []), ensure_ascii=False),
            'danma': json.dumps(pred.get('danma', []), ensure_ascii=False),
            'specials': json.dumps(pred.get('special', []), ensure_ascii=False),
            'kill': json.dumps(pred.get('kill', []), ensure_ascii=False),
            'combo2': json.dumps(pred.get('combo2', [])[:10], ensure_ascii=False),
            'combo3': json.dumps(pred.get('combo3', [])[:10], ensure_ascii=False),
            'top_scores': json.dumps(pred.get('detail', []), ensure_ascii=False),
        }
        saved = save_prediction_log(log_row)
        if saved:
            st.success('已将本次预测结果记录到 prediction_log.csv')
        else:
            st.info('本期预测结果已存在，未重复记录')

    # 执行额外分析并展示
    if analyze_special_flag:
        st.subheader("特码 分析")
        sp_rank = analyze_special(df, history_count, method_pool=False)
        st.markdown("**严格特码频率（仅统计最后一位）Top10**")
        st.table(pd.DataFrame(sp_rank[:10], columns=["号码","频率"]))
        sp_rank_pool = analyze_special(df, history_count, method_pool=True)
        st.markdown("**池化统计（每期7个号码）Top10**")
        st.table(pd.DataFrame(sp_rank_pool[:10], columns=["号码","频率"]))
        results['special'] = sp_rank

    if analyze_pingte_flag:
        st.subheader("平特一肖（前6区统计）")
        pf = analyze_front6_freq(df, history_count)
        st.table(pd.DataFrame(pf[:12], columns=["号码","频率"]))
        results['pingte'] = pf

    if analyze_sanlianxiao_flag:
        st.subheader("平特三连肖（前6区连续3期共同生肖）")
        sx = analyze_sanlianxiao(df, history_count)
        if sx:
            st.table(pd.DataFrame(sx[:10], columns=["生肖","出现次数"]))
        else:
            st.write("暂无连续三期共同生肖")
        results['sanlianxiao'] = sx

    if analyze_erzher_flag:
        st.subheader("二中二（前6区同期组合频次）")
        pairs = analyze_pairs_front6(df, history_count, top_k=30)
        pair_df = pd.DataFrame(pairs, columns=["组合","频次"])
        pair_df["组合"] = pair_df["组合"].apply(lambda x: f'{x[0]}-{x[1]}')
        st.table(pair_df)
        results['erzher'] = pairs

    if analyze_danma_flag:
        st.subheader("胆码（全区不分区）概率排名")
        dan = rank_danma(df, history_count)
        st.table(pd.DataFrame(dan[:12], columns=["号码","得分"]).style.format({"得分":"{:.2f}"}))
        results['danma'] = dan

    # 导出当前预测结果
    if st.button("导出当前预测为 CSV"):
        cur = pred.get('detail', [])
        if cur:
            df_export = pd.DataFrame(cur, columns=['num', 'score'])
            csv_bytes = df_export.to_csv(index=False).encode('utf-8-sig')
            st.download_button("下载预测 CSV", data=csv_bytes, file_name='prediction_detail.csv', mime='text/csv')
        else:
            st.warning("无可导出预测数据")

    # 回测
    if do_backtest:
        st.subheader("回测（最近 %d 期）" % backtest_rounds)
        # prepare chronological df
        df_asc = df.iloc[::-1].reset_index(drop=True)
        history_needed = max(20, history_count)
        start_idx = max(history_needed, len(df_asc) - backtest_rounds - 1)
        end_idx = len(df_asc) - 1

        sp_hits = 0
        total = 0
        pingte_hits = 0
        erzher_hits = 0

        for t in range(start_idx, end_idx):
            window = df_asc.iloc[max(0, t-history_count+1):t+1].iloc[::-1].reset_index(drop=True)
            actual_next = df_asc.iloc[t+1]
            # predict using fusion
            scores = score_fusion(window)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top12 = [x[0] for x in ranked[:12]]

            # special prediction: top1
            pred_special = top12[0]
            actual_special = actual_next.get('special', actual_next['nums'][-1])
            if pred_special == actual_special:
                sp_hits += 1

            # pingte: check how many top12 appear in actual front6
            front6 = set(actual_next['nums'][:6])
            hit_count = len(set(top12) & front6)
            if hit_count >= 1:
                pingte_hits += 1

            # erzher: check if any top pair appears fully in front6
            pairs = list(combinations(top12[:8], 2))
            pair_hit = any(set(p).issubset(front6) for p in pairs)
            if pair_hit:
                erzher_hits += 1

            total += 1

        if total > 0:
            st.write(f"特码 Top1 命中率: {sp_hits}/{total} = {sp_hits/total:.2%}")
            st.write(f"平特一肖 Top12 覆盖率（任一区出现）: {pingte_hits}/{total} = {pingte_hits/total:.2%}")
            st.write(f"二中二（基于 Top8 组合）命中率: {erzher_hits}/{total} = {erzher_hits/total:.2%}")
        else:
            st.write("样本不足，无法回测")

    # 按截止日期/期号回测入口
    st.sidebar.subheader("按截止点回测")
    cutoff_by = st.sidebar.radio("截止方式", options=["日期","期号"], index=0)
    cutoff_value = None
    if cutoff_by == "日期":
        cutoff_value = st.sidebar.date_input("选择截止日期（包含该日开奖）")
    else:
        cutoff_value = st.sidebar.text_input("输入截止期号（包含该期）", value="")

    if st.sidebar.button("运行按截止点回测"):
        # build df with parsed dates
        df2 = df.copy()
        df2["_date_parsed"] = pd.to_datetime(df2["date"], errors='coerce')

        if cutoff_by == "日期":
            if isinstance(cutoff_value, (str,)):
                st.error("请选择有效日期")
            else:
                cutoff_dt = pd.to_datetime(cutoff_value)
                # include draws on cutoff date
                df_cut = df2[df2["_date_parsed"] <= cutoff_dt]
        else:
            if not cutoff_value:
                st.error("请输入有效期号")
                df_cut = pd.DataFrame()
            else:
                df_cut = df2[df2["issue"] <= str(cutoff_value)]

        if df_cut.empty or len(df_cut) < 5:
            st.warning("截止点数据不足，无法回测")
        else:
            # chronological ascending
            df_cut_asc = df_cut.iloc[::-1].reset_index(drop=True)
            # simulate over all available points except last
            rows = []
            for t in range(0, len(df_cut_asc)-1):
                window = df_cut_asc.iloc[max(0, t-history_count+1):t+1].iloc[::-1].reset_index(drop=True)
                actual_next = df_cut_asc.iloc[t+1]
                scores = score_fusion(window)
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top12 = [x[0] for x in ranked[:12]]
                pred_special = top12[0]
                actual_special = actual_next.get('special', actual_next['nums'][-1])
                front6 = set(actual_next['nums'][:6])
                sp_hit = int(pred_special == actual_special)
                pingte_hit = int(len(set(top12) & front6) >= 1)
                pairs = list(combinations(top12[:8], 2))
                erzher_hit = int(any(set(p).issubset(front6) for p in pairs))

                rows.append({
                    'cutoff_issue': df_cut_asc.iloc[t].get('issue'),
                    'cutoff_date': df_cut_asc.iloc[t].get('date'),
                    'pred_special': pred_special,
                    'actual_special': actual_special,
                    'pred_top12': '|'.join(map(str, top12)),
                    'actual_front6': '|'.join(map(str, actual_next['nums'][:6])),
                    'sp_hit': sp_hit,
                    'pingte_hit': pingte_hit,
                    'erzher_hit': erzher_hit
                })

            res_df = pd.DataFrame(rows)
            total = len(res_df)
            st.write(f"样本数：{total}")
            st.write(f"特码 Top1 命中率: {res_df['sp_hit'].sum()}/{total} = {res_df['sp_hit'].sum()/total:.2%}")
            st.write(f"平特一肖 Top12 覆盖率: {res_df['pingte_hit'].sum()}/{total} = {res_df['pingte_hit'].sum()/total:.2%}")
            st.write(f"二中二 命中率: {res_df['erzher_hit'].sum()}/{total} = {res_df['erzher_hit'].sum()/total:.2%}")

            st.dataframe(res_df)

            # 导出为 CSV
            csv_bytes = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("下载回测结果 CSV", data=csv_bytes, file_name="backtest_results.csv", mime='text/csv')

    if show_log:
        st.subheader('历史预测记录')
        logs = load_prediction_log()
        if logs.empty:
            st.write('暂无历史记录。')
        else:
            st.write('最近 20 条记录：')
            st.dataframe(logs.tail(20).iloc[::-1].reset_index(drop=True))
            csv_bytes = logs.to_csv(index=False).encode('utf-8-sig')
            st.download_button('下载历史预测记录 CSV', data=csv_bytes, file_name='prediction_log.csv', mime='text/csv')



if __name__ == "__main__":
    main_ui()