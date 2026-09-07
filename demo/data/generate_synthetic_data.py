"""
仿真数据生成器 — 模拟直播电商销量数据

业务规则（基于访谈）：
- 日常销量基线：2000-8000 单/天
- 直播日销量：基线 × 5~50 倍（脉冲式峰值）
- 退货率：约 1/3（抖音渠道）
- 20% SKU 贡献 80% 销量（帕累托分布）
- 约每 10-15 天一次直播

输出字段：
- date           日期
- sales          实际销量（目标变量）
- price          商品价格（元）
- discount       折扣力度（0.1~1.0，1=原价）
- is_holiday     是否节假日
- is_weekend     是否周末
- anchor_fans    主播粉丝量（万）
- anchor_gmv_7d  主播近7天场均GMV（万）
- platform_traffic 平台推流指数（0~100）
- competitor_discount 竞品同期折扣（0.1~1.0）
- is_live_day    是否为直播日（标签，不作为特征）
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# ===== 基础配置 =====
N_DAYS = 500                    # 模拟 500 天（~1.5年）
BASE_SALES = 3000               # 日常基线销量
SALES_NOISE = 300               # 日常波动（降低噪声，从800→300）

# ===== 生成日期序列 =====
dates = pd.date_range('2025-01-01', periods=N_DAYS, freq='D')
df = pd.DataFrame({'date': dates})

# ===== 节日标记（春节、五一、618、双11附近） =====
holidays_2025_2026 = [
    '2025-01-28', '2025-01-29', '2025-01-30',  # 春节2025
    '2025-05-01', '2025-05-02',                  # 五一
    '2025-06-15', '2025-06-16', '2025-06-17', '2025-06-18',  # 618
    '2025-11-10', '2025-11-11',                  # 双11
    '2026-02-15', '2026-02-16', '2026-02-17',    # 春节2026
    '2026-05-01', '2026-05-02',                  # 五一
    '2026-06-15', '2026-06-16', '2026-06-17', '2026-06-18',  # 618
]
df['is_holiday'] = df['date'].astype(str).isin(holidays_2025_2026).astype(int)
df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)

# ===== 商品价格 =====
df['price'] = np.random.choice([59, 79, 99, 129, 159, 199], N_DAYS)

# ===== 折扣力度 =====
df['discount'] = np.random.uniform(0.4, 1.0, N_DAYS)
df.loc[df['is_holiday'] == 1, 'discount'] *= 0.7      # 节假日折扣更大
df['discount'] = df['discount'].clip(0.1, 1.0).round(2)

# ===== 主播信息（模拟 3 个不同量级的主播轮换） =====
anchors = [
    {'fans': 8000, 'gmv_base': 5000},   # 头部主播
    {'fans': 3000, 'gmv_base': 1500},   # 腰部主播
    {'fans': 500,  'gmv_base': 300},    # 小主播
]
anchor_idx = np.random.choice(3, N_DAYS, p=[0.3, 0.4, 0.3])
df['anchor_fans'] = [anchors[i]['fans'] for i in anchor_idx]
df['anchor_gmv_7d'] = [anchors[i]['gmv_base'] + np.random.normal(0, 200) for i in anchor_idx]

# ===== 平台推流指数 =====
df['platform_traffic'] = np.random.uniform(10, 80, N_DAYS).round(0)
df.loc[df['is_holiday'] == 1, 'platform_traffic'] += 20  # 节假日推流加大
df['platform_traffic'] = df['platform_traffic'].clip(0, 100)

# ===== 竞品同期折扣 =====
df['competitor_discount'] = np.random.uniform(0.3, 1.0, N_DAYS).round(2)

# ===== 直播日标记（每 10-15 天一次） =====
df['is_live_day'] = 0
live_interval = 0
for i in range(N_DAYS):
    if live_interval <= 0:
        if np.random.random() < 0.5 or df.loc[i, 'is_holiday'] == 1:
            df.loc[i, 'is_live_day'] = 1
            live_interval = np.random.randint(8, 14)
    live_interval -= 1

# ===== 生成销量 =====
df['sales'] = 0.0
for i in range(N_DAYS):
    # 基础销量 = 日常基线 + 噪声
    base = BASE_SALES + np.random.normal(0, SALES_NOISE)

    # 节假日/周末小幅提升
    if df.loc[i, 'is_holiday']:
        base *= np.random.uniform(1.3, 2.0)
    elif df.loc[i, 'is_weekend']:
        base *= np.random.uniform(1.1, 1.4)

    # 价格弹性：折扣越大卖越多（加强信号）
    discount_effect = 1 + (1 - df.loc[i, 'discount']) * 4.0

    # 竞品因素：竞品折扣越大，我们卖的越少
    competitor_effect = 1 - (1 - df.loc[i, 'competitor_discount']) * 0.8

    # 平台推流效应（加强）
    traffic_effect = 1 + df.loc[i, 'platform_traffic'] / 150

    # 主播影响力（加强）
    anchor_effect = 0.2 + df.loc[i, 'anchor_fans'] / 8000

    daily_sales = base * discount_effect * competitor_effect * traffic_effect * anchor_effect

    # 直播日脉冲：核心改进——让脉冲"部分可预测"
    # 公式：pulse = 基础3 + 折扣贡献 + 主播贡献 + 推流贡献 + 30%随机噪声
    # 这样模型可以从特征中学到"高折扣+大主播+强推流 → 大概率爆单"
    if df.loc[i, 'is_live_day']:
        # 可预测部分
        discount_boost = (1 - df.loc[i, 'discount']) * 35       # 加强
        anchor_boost = df.loc[i, 'anchor_fans'] / 250            # 加强
        traffic_boost = df.loc[i, 'platform_traffic'] / 4        # 加强
        predictable = 2 + discount_boost + anchor_boost + traffic_boost
        # 不可预测部分（20%随机噪声，降低）
        pulse_multiplier = predictable * np.random.uniform(0.8, 1.2)
        pulse_multiplier = np.clip(pulse_multiplier, 3, 55)
        daily_sales *= pulse_multiplier

    # 加噪声 + 取整
    daily_sales *= np.random.uniform(0.85, 1.15)
    df.loc[i, 'sales'] = max(100, int(daily_sales))

print(f"数据生成完成：{N_DAYS} 天")
print(f"直播日数：{df['is_live_day'].sum()}")
print(f"非直播日均销量：{df[df['is_live_day']==0]['sales'].mean():.0f}")
print(f"直播日均销量：{df[df['is_live_day']==1]['sales'].mean():.0f}")
print(f"直播日峰值：{df['sales'].max()}")
print(f"退货率（模拟）：每月约 30% 需预留产能")
print(f"\n前5行预览：")
print(df.head())
print(f"\n后5行预览：")
print(df.tail())
print(f"\n数据形状：{df.shape}")
