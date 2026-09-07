"""
特征工程模块 — 直播电商销量预测

第一阶段特征（~30个）：
  - 时序滞后 (6)
  - 滚动统计 (7)
  - 趋势变化 (3)
  - 事件时间 (3)
  - 日期特征 (6)
  - 价格折扣 (4)
  - 交叉特征 (5)
  - 原始保留 (4)

注意：is_live_day 不入模（数据泄露）
"""
import pandas as pd
import numpy as np


class LiveStreamFeatureEngineer:
    """直播电商销量预测特征工程"""

    def __init__(self):
        self.feature_names_ = None
        self.categorical_features_ = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """读取原始数据 → 构造 y + 全部特征 → 拆分 X, y"""
        df = df.copy()

        # ===== 1. 构造目标变量 y =====
        df['y'] = df['sales'].shift(-1)

        # ===== 2. 时序滞后特征 =====
        df['sales_lag_1'] = df['sales'].shift(1)
        df['sales_lag_2'] = df['sales'].shift(2)
        df['sales_lag_3'] = df['sales'].shift(3)
        df['sales_lag_5'] = df['sales'].shift(5)
        df['sales_lag_7'] = df['sales'].shift(7)
        df['sales_lag_14'] = df['sales'].shift(14)

        # ===== 3. 滚动统计特征 =====
        df['sales_roll3_mean'] = df['sales'].rolling(3).mean()
        df['sales_roll7_mean'] = df['sales'].rolling(7).mean()
        df['sales_roll14_mean'] = df['sales'].rolling(14).mean()
        df['sales_roll3_std'] = df['sales'].rolling(3).std()
        df['sales_roll7_std'] = df['sales'].rolling(7).std()
        df['sales_roll3_max'] = df['sales'].rolling(3).max()
        df['sales_roll7_max'] = df['sales'].rolling(7).max()

        # ===== 4. 趋势变化特征 =====
        df['sales_trend_3_7'] = df['sales_roll3_mean'] / (df['sales_roll7_mean'] + 1)
        df['sales_trend_7_14'] = df['sales_roll7_mean'] / (df['sales_roll14_mean'] + 1)
        df['sales_lag1_vs_roll7'] = df['sales_lag_1'] / (df['sales_roll7_mean'] + 1)

        # ===== 5. 事件时间特征 =====
        # 距上次直播日天数
        df['days_since_last_live'] = (
            df['is_live_day'].cumsum()
            .where(df['is_live_day'] == 0)
            .ffill()
            .fillna(0)
            .pipe(lambda s: df['is_live_day'].cumsum() - s)
        )
        # 标记为 after-live 时期需要的是距离，用 groupby + cumcount 更准确
        live_groups = df['is_live_day'].cumsum()
        df['days_since_last_live'] = df.groupby(live_groups).cumcount()
        df.loc[df['is_live_day'] == 1, 'days_since_last_live'] = 0

        # 近7天内有几天是直播日
        df['days_live_in_7d'] = df['is_live_day'].rolling(7).sum()

        # 昨天是不是直播日（shift 后）
        df['is_post_live_day'] = df['is_live_day'].shift(1).fillna(0).astype(int)

        # ===== 6. 日期特征 =====
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        # is_holiday, is_weekend 已在原始数据中

        # ===== 7. 价格与折扣特征 =====
        df['price_change'] = df['price'].diff().fillna(0)
        df['discount_change'] = df['discount'].diff().fillna(0)

        # ===== 8. 交叉特征 =====
        df['discount_x_anchor'] = df['discount'] * df['anchor_fans']
        df['discount_x_traffic'] = df['discount'] * df['platform_traffic']
        df['price_vs_competitor'] = df['competitor_discount'] / (df['discount'] + 0.01)
        df['anchor_x_traffic'] = df['anchor_fans'] / 1000 * df['platform_traffic']
        df['discount_x_sales_momentum'] = df['discount'] * df['sales_roll7_mean'].fillna(0)

        # ===== 9. 丢弃 NaN 行 =====
        # 前14行无滚动窗口 + 最后1行无 y
        df = df.dropna(subset=['y', 'sales_roll14_mean']).reset_index(drop=True)

        # ===== 10. 特征列定义 =====
        feature_cols = [
            # 时序滞后
            'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
            'sales_lag_5', 'sales_lag_7', 'sales_lag_14',
            # 滚动统计
            'sales_roll3_mean', 'sales_roll7_mean', 'sales_roll14_mean',
            'sales_roll3_std', 'sales_roll7_std',
            'sales_roll3_max', 'sales_roll7_max',
            # 趋势变化
            'sales_trend_3_7', 'sales_trend_7_14', 'sales_lag1_vs_roll7',
            # 事件时间
            'days_since_last_live', 'days_live_in_7d', 'is_post_live_day',
            # 日期
            'day_of_week', 'month', 'is_holiday', 'is_weekend',
            'is_month_start', 'is_month_end',
            # 价格折扣
            'price', 'discount', 'price_change', 'discount_change',
            # 交叉特征
            'discount_x_anchor', 'discount_x_traffic', 'price_vs_competitor',
            'anchor_x_traffic', 'discount_x_sales_momentum',
            # 原始
            'anchor_fans', 'anchor_gmv_7d',
            'platform_traffic', 'competitor_discount',
        ]

        # 类别特征（LightGBM 原生支持）
        self.categorical_features_ = [
            'day_of_week', 'month', 'is_holiday', 'is_weekend',
            'is_month_start', 'is_month_end', 'is_post_live_day',
        ]

        self.feature_names_ = feature_cols

        X = df[feature_cols]
        y = df['y']

        print(f"特征工程完成：{X.shape[0]} 行 × {X.shape[1]} 个特征")
        print(f"类别特征 ({len(self.categorical_features_)} 个): {self.categorical_features_}")

        return X, y


if __name__ == '__main__':
    # 快速验证
    import sys
    sys.path.insert(0, '..')
    df = pd.read_csv('data/synthetic_sales_data.csv', parse_dates=['date'])

    engineer = LiveStreamFeatureEngineer()
    X, y = engineer.fit_transform(df)

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Feature columns:")
    for i, col in enumerate(X.columns, 1):
        print(f"  {i:2d}. {col}")
    print(f"\ny 统计: mean={y.mean():.0f}, std={y.std():.0f}, min={y.min():.0f}, max={y.max():.0f}")
    print(f"\nX 预览 (前3行):\n{X.head(3)}")
