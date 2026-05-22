import json
import os
import requests
import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
import math
import random

# =====================================================
# 页面配置
# =====================================================
st.set_page_config(
    page_title=" AI 超级智能系统 Ultimate",
    layout="wide"
)

st.title(" AI 超级智能系统 Ultimate")

# =====================================================
# API
# =====================================================
LATEST_API = "https://macaumarksix.com/api/macaujc2.com"
HISTORY_API = "https://history.macaumarksix.com/history/macaujc2/y/{}"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

AI_FILE = "ai_learn.json"

# =====================================================
# 波色
# =====================================================
RED = {1,2,7,8,12,13,18,19,23,24,29,30,34,35,40,45,46}
BLUE = {3,4,9,10,14,15,20,25,26,31,36,37,41,42,47,48}
GREEN = {5,6,11,16,17,21,22,27,28,32,33,38,39,43,44,49}

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

def get_zone(num):

    if num <= 16:
        return "低区"

    elif num <= 33:
        return "中区"

    return "高区"

# =====================================================
# AI学习
# =====================================================
def load_ai():

    default = {

        "miss":2.0,
        "hot":1.8,
        "cold":2.5,
        "wave":2.0,
        "zodiac":2.0,
        "tail":2.0,
        "zone":1.5,
        "element":2.0,
        "consecutive":2.5,
        "special":5.0,
        "bayes":2.0,
        "cycle":2.5
    }

    if not os.path.exists(AI_FILE):

        with open(AI_FILE,"w",encoding="utf-8") as f:
            json.dump(default,f,ensure_ascii=False,indent=4)

        return default

    try:

        with open(AI_FILE,"r",encoding="utf-8") as f:
            data=json.load(f)

        for k,v in default.items():

            if k not in data:
                data[k]=v

        return data

    except:
        return default

def save_ai(data):

    with open(AI_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=4)

# =====================================================
# API
# =====================================================
@st.cache_data(ttl=300)
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

@st.cache_data(ttl=600)
def fetch_history(year):

    try:

        url = HISTORY_API.format(year)

        r = requests.get(
            url,
            headers=HEADERS,
            timeout=20
        )

        data = r.json()

        return data.get("data",[])

    except:
        return []

# =====================================================
# 解析
# =====================================================
def parse_history(data):

    rows=[]

    for item in data:

        try:

            nums=[
                int(x)
                for x in item["openCode"].split(",")
            ]

            if len(nums)!=7:
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

# =====================================================
# 周期分析
# =====================================================
def cycle_model(df,score):

    periods=[3,5,8]

    for p in periods:

        recent=df.head(p)

        count=Counter()

        for row in recent["nums"]:

            for n in row:
                count[n]+=1

        weak=min(count.values()) if count else 0

        for n in range(1,50):

            if count[n]<=weak:
                score[n]+=1.2

# =====================================================
# 贝叶斯模型
# =====================================================
def bayes_model(df,score):

    recent=df.head(20)

    total=0

    freq=Counter()

    for row in recent["nums"]:

        for n in row:

            freq[n]+=1
            total+=1

    for n in range(1,50):

        p=(freq[n]+1)/(total+49)

        score[n]+=p*100

# =====================================================
# 平特一肖（升级版）
# =====================================================
def yixiao_model(df,history_count):

    recent=df.head(history_count)

    zodiac_score=Counter()

    # ============================================
    # 1. 普通号生肖统计
    # ============================================
    for idx,row in recent.iterrows():

        decay=(history_count-idx)/history_count

        # 前6区
        for n in row["normal"]:

            z=get_zodiac(n)

            zodiac_score[z]+=1.2*decay

        # 特码区
        sp=row["special"]

        zodiac_score[
            get_zodiac(sp)
        ] += 2.5*decay

    # ============================================
    # 2. 连续出现降温
    # ============================================
    last3=recent.head(3)

    repeat_counter=Counter()

    for _,row in last3.iterrows():

        z=get_zodiac(row["special"])

        repeat_counter[z]+=1

    for z,c in repeat_counter.items():

        if c>=2:

            zodiac_score[z]-=4*c

    # ============================================
    # 3. 冷门生肖补偿
    # ============================================
    recent10=recent.head(10)

    appear=Counter()

    for _,row in recent10.iterrows():

        for n in row["nums"]:

            appear[get_zodiac(n)] += 1

    min_count=min(appear.values()) if appear else 0

    for z in ZODIAC_MAP:

        if appear[z]<=min_count:

            zodiac_score[z]+=3

    # ============================================
    # 4. 波色联动
    # ============================================
    wave_map=defaultdict(int)

    for _,row in recent.iterrows():

        sp=row["special"]

        wave=get_wave(sp)

        wave_map[wave]+=1

    weak_wave=min(
        wave_map,
        key=wave_map.get
    )

    for z,nums in ZODIAC_MAP.items():

        bonus=0

        for n in nums:

            if get_wave(n)==weak_wave:

                bonus+=0.8

        zodiac_score[z]+=bonus

    # ============================================
    # 5. 排序
    # ============================================
    result=sorted(

        zodiac_score.items(),

        key=lambda x:x[1],

        reverse=True
    )

    return [x[0] for x in result[:5]]

# =====================================================
# AI核心
# =====================================================
def ai_engine(df,history_count=12):

    weights=load_ai()

    recent=df.head(history_count)

    score=defaultdict(float)

    freq=Counter()
    zodiac_count=Counter()
    wave_count=Counter()
    tail_count=Counter()
    zone_count=Counter()
    element_count=Counter()

    total=max(len(recent),1)

    # =================================================
    # 主模型
    # =================================================
    for idx,row in recent.iterrows():

        decay=math.exp(-(idx/total))

        normals=row["normal"]

        special=row["special"]

        for n in normals:

            freq[n]+=1

            score[n]+=3.5*decay

            zodiac_count[get_zodiac(n)] += 1
            wave_count[get_wave(n)] += 1
            tail_count[get_tail(n)] += 1
            zone_count[get_zone(n)] += 1
            element_count[get_element(n)] += 1

        score[special]+=weights["special"]*decay

    # =================================================
    # 遗漏
    # =================================================
    all_nums=df["nums"].tolist()

    for n in range(1,50):

        miss=0

        for row in all_nums:

            if n in row:
                break

            miss+=1

        miss=min(miss,18)

        score[n]+=miss*weights["miss"]

    # =================================================
    # 热号降温
    # =================================================
    for n,c in freq.items():

        if c>=5:
            score[n]-=c*weights["hot"]

    # =================================================
    # 冷号
    # =================================================
    for n in range(1,50):

        if freq[n]==0:
            score[n]+=weights["cold"]

    # =================================================
    # 波色轮动
    # =================================================
    if wave_count:

        weak=min(
            wave_count,
            key=wave_count.get
        )

        for n in range(1,50):

            if get_wave(n)==weak:
                score[n]+=weights["wave"]

    # =================================================
    # 生肖轮动
    # =================================================
    if zodiac_count:

        weak=min(
            zodiac_count,
            key=zodiac_count.get
        )

        for n in range(1,50):

            if get_zodiac(n)==weak:
                score[n]+=weights["zodiac"]

    # =================================================
    # 尾数轮动
    # =================================================
    if tail_count:

        weak=min(
            tail_count,
            key=tail_count.get
        )

        for n in range(1,50):

            if get_tail(n)==weak:
                score[n]+=weights["tail"]

    # =================================================
    # 五行轮动
    # =================================================
    if element_count:

        weak=min(
            element_count,
            key=element_count.get
        )

        for n in range(1,50):

            if get_element(n)==weak:
                score[n]+=weights["element"]

    # =================================================
    # 周期模型
    # =================================================
    cycle_model(df,score)

    # =================================================
    # 贝叶斯
    # =================================================
    bayes_model(df,score)

    # =================================================
    # 连码
    # =================================================
    for row in recent["normal"]:

        nums=sorted(row)

        for i in range(len(nums)-1):

            diff=nums[i+1]-nums[i]

            if diff==1:

                score[nums[i]]+=weights["consecutive"]
                score[nums[i+1]]+=weights["consecutive"]

    # =================================================
    # 杀码
    # =================================================
    latest_nums=recent.iloc[0]["nums"]

    kill=[]

    for n in range(1,50):

        penalty=0

        if n in latest_nums:
            penalty+=6

        if freq[n]>=4:
            penalty+=4

        if tail_count[get_tail(n)]>=5:
            penalty+=2

        score[n]-=penalty

        if penalty>=6:
            kill.append(n)

    # =================================================
    # 防负数
    # =================================================
    for n in range(1,50):

        score[n]=max(score[n],0.1)

    # =================================================
    # 排序
    # =================================================
    final_rank=sorted(
        score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    # =================================================
    # 推荐号
    # =================================================
    numbers=[]

    for n,_ in final_rank:

        if n not in kill:
            numbers.append(n)

        if len(numbers)>=12:
            break

    # =================================================
    # 胆码分散
    # =================================================
    danma=[]

    used_wave=set()
    used_zodiac=set()

    for n in numbers:

        w=get_wave(n)
        z=get_zodiac(n)

        if w not in used_wave and z not in used_zodiac:

            danma.append(n)

            used_wave.add(w)
            used_zodiac.add(z)

        if len(danma)>=4:
            break

    # =================================================
    # 特码模型（独立）
    # =================================================
    special_score=defaultdict(float)

    for n in range(1,50):

        special_score[n]=score[n]

        if n in latest_nums:
            special_score[n]-=8

        if get_tail(n)==get_tail(recent.iloc[0]["special"]):
            special_score[n]-=4

    special_rank=sorted(
        special_score.items(),
        key=lambda x:x[1],
        reverse=True
    )

    special=[
        x[0]
        for x in special_rank[:8]
    ]

    # =================================================
    # 概率
    # =================================================
    total_score=sum(score.values())

    prob={}

    for n in range(1,50):

        prob[n]=round(
            score[n]/total_score*100,
            2
        )

    # =================================================
    # 连尾
    # =================================================
    tail_rank=Counter()

    for row in recent["nums"]:

        for n in row:

            tail_rank[get_tail(n)] += 1

    tails=[
        x[0]
        for x in tail_rank.most_common(5)
    ]

    # =================================================
    # 连肖树
    # =================================================
    yixiao=yixiao_model(df,history_count)

    # =================================================
    # 二中二 三中三
    # =================================================
    combo2=list(
        combinations(numbers[:8],2)
    )

    combo3=list(
        combinations(numbers[:8],3)
    )

    # =================================================
    # 自学习
    # =================================================
    try:

        last_real=recent.iloc[0]["nums"]

        hit=len(
            set(numbers[:10]) &
            set(last_real)
        )

        if hit>=3:

            weights["miss"]=min(
                weights["miss"]+0.02,
                8
            )

            weights["cold"]=min(
                weights["cold"]+0.02,
                8
            )

        else:

            weights["hot"]=max(
                weights["hot"]-0.01,
                1
            )

        save_ai(weights)

    except:
        pass

    return {

        "numbers":numbers,
        "danma":danma,
        "kill":kill[:6],
        "special":special,
        "prob":prob,
        "combo2":combo2[:10],
        "combo3":combo3[:10],
        "tails":tails,
        "yixiao":yixiao,
        "detail":final_rank[:20],
        "weights":weights
    }

# =====================================================
# 刷新按钮
# =====================================================
if st.button("手动刷新最新数据"):

    st.cache_data.clear()

# =====================================================
# 滑块
# =====================================================
history_count=st.sidebar.slider(
    "分析最近期数",
    min_value=1,
    max_value=25,
    value=12,
    step=1
)

# =====================================================
# 获取数据
# =====================================================
with st.spinner("AI超级分析中..."):

    latest=fetch_latest()

    history=fetch_history(
        datetime.now().year
    )

if not latest:

    st.error("API获取失败")
    st.stop()

latest_item=sorted(
    latest,
    key=lambda x:x["expect"],
    reverse=True
)[0]

# =====================================================
# 去重
# =====================================================
seen=set()

clean=[]

for item in history:

    if item["expect"] not in seen:

        seen.add(item["expect"])
        clean.append(item)

if latest_item["expect"] not in seen:
    clean.insert(0,latest_item)

# =====================================================
# DataFrame
# =====================================================
df=parse_history(clean)

df["time"]=pd.to_datetime(df["time"])

df=df.sort_values(
    by="time",
    ascending=False
).reset_index(drop=True)

# =====================================================
# AI分析
# =====================================================
result=ai_engine(df,history_count)

# =====================================================
# 最新开奖
# =====================================================
st.header("最新开奖")

st.success(
    latest_item["openCode"]
)

# =====================================================
# AI推荐号码
# =====================================================
st.header("AI推荐号码")

st.success(
    " / ".join([
        f"{n:02d}"
        for n in result["numbers"]
    ])
)

# =====================================================
# AI最强胆码
# =====================================================
st.header("AI最强胆码")

st.warning(
    " / ".join([
        f"{n:02d}"
        for n in result["danma"]
    ])
)

# =====================================================
# 平特一肖
# =====================================================
st.header("AI平特一肖")

for z in result["yixiao"]:

    nums=" ".join([
        f"{n:02d}"
        for n in ZODIAC_MAP[z]
    ])

    st.info(f"{z} → {nums}")

# =====================================================
# 连尾
# =====================================================
st.header("AI连尾预测")

st.warning(
    " / ".join([
        str(x)
        for x in result["tails"]
    ])
)

# =====================================================
# 杀码
# =====================================================
st.header("AI杀码")

st.error(
    " / ".join([
        f"{n:02d}"
        for n in result["kill"]
    ])
)

# =====================================================
# 二中二
# =====================================================
st.header("AI二中二")

for item in result["combo2"]:

    st.info(
        " - ".join([
            f"{x:02d}"
            for x in item
        ])
    )

# =====================================================
# 三中三
# =====================================================
st.header("AI三中三")

for item in result["combo3"]:

    st.success(
        " - ".join([
            f"{x:02d}"
            for x in item
        ])
    )

# =====================================================
# 特码
# =====================================================
st.header("AI推荐特码")

st.error(
    " / ".join([
        f"{n:02d}"
        for n in result["special"]
    ])
)

# =====================================================
# 概率详情
# =====================================================
st.header("AI号码评分")

detail_df=pd.DataFrame(
    result["detail"],
    columns=["号码","评分"]
)

detail_df["生肖"]=detail_df["号码"].apply(
    get_zodiac
)

detail_df["波色"]=detail_df["号码"].apply(
    get_wave
)

detail_df["五行"]=detail_df["号码"].apply(
    get_element
)

detail_df["概率%"]=detail_df["号码"].apply(
    lambda x: result["prob"][x]
)

st.dataframe(
    detail_df,
    use_container_width=True
)

# =====================================================
# AI动态学习
# =====================================================
st.header("AI动态学习权重")

weight_df=pd.DataFrame({

    "模型":list(result["weights"].keys()),
    "权重":list(result["weights"].values())
})

st.dataframe(
    weight_df,
    use_container_width=True
)

# =====================================================
# 最近开奖
# =====================================================
st.header(f"最近{history_count}期开奖")

show_df=df.head(history_count)[[
    "expect",
    "time",
    "nums"
]].copy()

show_df.columns=[
    "期号",
    "开奖时间",
    "开奖号码"
]

show_df["开奖号码"]=show_df["开奖号码"].apply(

    lambda x:" ".join([
        f"{n:02d}"
        for n in x
    ])
)

st.dataframe(
    show_df,
    use_container_width=True
)

# =====================================================
# 底部
# =====================================================
st.caption("AI Ultimate 超级智能学习系统")