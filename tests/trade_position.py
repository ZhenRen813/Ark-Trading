import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Tuple

class PairsTradingPosition:
    def __init__(self, available_margin,total_capital: float = 1000, risk_ratio: float = 0.1, 
                 leverage: float = 1000, min_trade_size: float = 0.01):
        self.total_capital = total_capital  # 账户总资金
        self.risk_ratio = risk_ratio  # 风险敞口比例
        self.leverage = leverage  # 杠杆比例
        self.min_trade_size = min_trade_size  # 最小交易手数
        self.hedge_ratio = None  # 对冲比率
        self.current_position = {'X': 0, 'Y': 0}  # 当前持仓
        self.available_margin = available_margin


    def calculate_position_size(self, signal, available_margin):
        # 提取参数并校验
        spread = abs(signal.get('spread', 0))
        zscore = signal.get('zscore', 0)
        self.available_margin = available_margin
        hedge_ratio = signal.get('hedge_ratio', 1.0)
        
        # === 交易信号判断 ===
        position_direction = 0
        if abs(zscore) > self.z_entry:
            position_direction = -np.sign(zscore)  # Z>0时做空价差，Z<0时做多
        elif abs(zscore) < self.z_exit:
            return {'X': 0.0, 'Y': 0.0}  # 平仓
        
        # === 头寸计算 ===
        # 风险金额计算
        risk_amount = available_margin * self.leverage * self.risk_ratio
        
        # 波动率计算（分母保护）
        denominator = max(abs(zscore), 0.1)  # 防止分母过小
        spread_volatility = spread / denominator
        
        # 基础手数计算
        if spread_volatility < 1e-6:
            base_units = 0.0
        else:
            base_units = (risk_amount / spread_volatility) * position_direction
        
        # 应用最小手数限制（保留方向）
        adjusted_units = np.sign(base_units) * max(
            abs(base_units), 
            self.min_trade_size
        )
        
        # === 头寸分配 ===
        return {
            'X': adjusted_units * abs(hedge_ratio),  # 幅度由对冲比率绝对值决定
            'Y': -adjusted_units * np.sign(hedge_ratio)  # 方向由对冲比率符号控制
        }



    # def calculate_position_size(self, signal: Dict,available_margin) -> Dict[str, float]:
    #     self.available_margin = available_margin
    #     if available_margin <= 0:
    #         return {'X': 0, 'Y': 0}
    #     if signal['signal'] is None:
    #         return {'X': 0, 'Y': 0}
    #     if signal['signal'] == 'close':
    #         x_units = -self.current_position['X']
    #         y_units = -self.current_position['Y']
    #         # 更新持仓状态
    #         self.current_position['X'] += x_units
    #         self.current_position['Y'] += y_units
    #         # 返回计算结果
    #         return {
    #             'X': x_units,
    #             'Y': y_units,
    #             'hedge_ratio': self.hedge_ratio,
    #             'current_spread': signal['spread']
    #         }

    #     self.hedge_ratio = signal['hedge_ratio']

    #     # 计算基础头寸单位（考虑杠杆）
    #     notional_value = self.available_margin * self.leverage  # 可用的名义价值
    #     risk_amount = notional_value * self.risk_ratio  # 愿意承担的风险金额

    #     # 计算价差波动率（与原始代码相同）
    #     # spread_volatility = abs(signal['spread'] / signal['zscore']) if signal['zscore'] != 0 else 1
    #     # 修改后
    #     spread_volatility = max(abs(signal['spread'] / signal['zscore']), 0.01)  # 设置最小波动率

    #     # 基础单位计算（保留符号）
    #     base_units = risk_amount / spread_volatility
        
    #     # 应用最小交易手数限制‌:ml-citation{ref="1,3" data="citationList"}
    #     adjusted_base_units = np.sign(base_units) * max(
    #         abs(base_units), 
    #         self.min_trade_size
    #     )

    #     # 确定交易方向并计算交易手数
    #     if signal['signal'] == 'long':
    #         y_units = adjusted_base_units
    #         x_units = -adjusted_base_units * self.hedge_ratio
    #     elif signal['signal'] == 'short':
    #         y_units = -adjusted_base_units
    #         x_units = adjusted_base_units * self.hedge_ratio
    #     else:  # close信号
    #         x_units = -self.current_position['X']
    #         y_units = -self.current_position['Y']

    #     # 更新持仓状态
    #     self.current_position['X'] += x_units
    #     self.current_position['Y'] += y_units

    #     # 注意：这里需要外部传入available_margin，因为Position类本身不知道保证金情况
    #     # 为了示例完整性，这里假设有一个外部传入的available_margin变量（实际应用中需要调整）
    #     # 但在本示例中，我们仅返回计算结果，不实际使用这个变量进行计算（因为get_signal暂时没传）
    #     # 实际应用中，您需要在get_signal中调用此方法，并传入正确的available_margin值
    #     # 例如：position.calculate_position_size(signal_with_margin)
    #     # 并且在Position类中添加一个方法或参数来接收和使用这个值

    #     # 返回计算结果
    #     return {
    #         'X': x_units,
    #         'Y': y_units,
    #         'hedge_ratio': self.hedge_ratio,
    #         'current_spread': signal['spread']
    #     }

    # # 注意：由于Position类本身不知道当前的保证金情况，因此这里假设了一个外部传入的机制
    # # 在实际应用中，您可能需要调整类的设计或添加额外的参数/方法来处理这种情况



# # 使用示例
# if __name__ == "__main__":
#     # 生成测试数据
#     np.random.seed(42)
#     price_x = np.random.normal(0, 1, 100).cumsum() + 50
#     price_y = price_x * 1.2 + np.random.normal(0, 0.5, 100)
    
#     pt = PairsTradingPosition(total_capital=1e6)
    
#     # 模拟交易信号
#     test_signal = {
#         'signal': 'long',
#         'zscore': -1.8,
#         'hedge_ratio': float(0.9999999999999893),
#         'spread': -3.2
#     }

#     print(test_signal)
    
#     position = pt.calculate_position_size(test_signal)
#     print(f"头寸分配: {position}")
