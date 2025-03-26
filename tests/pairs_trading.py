import numpy as np
import pandas as pd
from paris_trading_2 import TradeSignal

class PairsTrading:
    def __init__(self, 保证金, 杠杆, 最小手数, X数据, Y数据, entry_z, exit_z):
        # 初始化参数
        self.可用保证金 = 保证金
        self.杠杆 = 杠杆
        self.最小手数 = 最小手数
        self.X = pd.Series(X数据['close'], index=X数据['date'])
        self.Y = pd.Series(Y数据['close'], index=Y数据['date'])
        self.entry_z = entry_z
        self.exit_z = exit_z
        
        # 策略参数
        self.头寸策略 = 'Single'  # 默认固定策略
        self.风险敞口 = 0.2  # 独立策略时单次交易最大资金占比
        
        # 状态变量
        self.价差 = self.Y - self.X
        self.mean_spread = self.价差.mean()
        self.std_spread = self.价差.std()
        self.position = {'X':0, 'Y':0}  # 当前持仓
        self.active_positions = []  # 独立策略的持仓记录

    def 更新数据(self, X价格, Y价格, 新保证金):
        # 更新最新价格和保证金
        self.X.iloc[-1] = X价格
        self.Y.iloc[-1] = Y价格
        self.可用保证金 = 新保证金
        
        # 计算新价差
        current_spread = self.Y.iloc[-1] - self.X.iloc[-1]
        z_score = (current_spread - self.mean_spread) / self.std_spread
        
        return self.生成信号(z_score)

    def 生成信号(self, z_score):
        if self.头寸策略 == 'fixed':
            return self.固定策略(z_score)
        else:
            return self.独立策略(z_score)

    def 固定策略(self, z_score):
        # 当前无持仓
        if sum(self.position.values()) == 0:  
            if z_score > self.entry_z:
                vol = self.计算手数(做空=True)
                self.position = {'X': vol, 'Y': -vol}
                return {'标的':('X','Y'), '手数':vol, '方向':'做空价差'}
            elif z_score < -self.entry_z:
                vol = self.计算手数()
                self.position = {'X': -vol, 'Y': vol}
                return {'标的':('X','Y'), '手数':vol, '方向':'做多价差'}
        
        # 已有持仓处理
        elif (self.position['X'] > 0 and z_score < self.exit_z) or \
             (self.position['X'] < 0 and z_score > -self.exit_z):
            vol = abs(self.position['X'])
            self.position = {'X':0, 'Y':0}
            return {'标的':('X','Y'), '手数':vol, '方向':'平仓'}
        
        return {'标的':(), '手数':0, '方向':'无操作'}

    def 独立策略(self, z_score):
        # 计算可用资金限额
        max_risk = self.可用保证金 * self.风险敞口 * self.杠杆
        signals = []
        
        # 生成新信号
        if z_score > self.entry_z:
            vol = min(max_risk/(self.X.iloc[-1]+self.Y.iloc[-1]), 
                     self.计算手数())
            signals.append({'标的':('X','Y'), '手数':vol, '方向':'做空价差'})
            self.active_positions.append({'vol':vol, 'entry_z':z_score})
        elif z_score < -self.entry_z:
            vol = min(max_risk/(self.X.iloc[-1]+self.Y.iloc[-1]),
                     self.计算手数())
            signals.append({'标的':('X','Y'), '手数':vol, '方向':'做多价差'})
            self.active_positions.append({'vol':vol, 'entry_z':z_score})
        
        # 检查平仓条件
        for pos in self.active_positions:
            if abs(z_score) < self.exit_z:
                signals.append({'标的':('X','Y'), '手数':pos['vol'], '方向':'平仓'})
                self.active_positions.remove(pos)
        
        return signals if signals else {'方向':'无操作'}

    def 计算手数(self, 做空=False):
        base = self.可用保证金 * self.杠杆
        price = self.X.iloc[-1] + self.Y.iloc[-1]
        vol = max(round(base / price / 100, 2), self.最小手数)
        return vol if not 做空 else -vol

    def 切换策略(self, 策略名称):
        self.头寸策略 = 策略名称
        self.active_positions = []  # 清空独立策略持仓记录

# 数据回测示例
if __name__ == "__main__":
    # 生成测试数据
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    np.random.seed(42)
    X_prices = np.cumsum(np.random.normal(0, 1, 100)) + 50
    Y_prices = X_prices * 1.2 + np.random.normal(0, 0.5, 100)
    
    # 初始化交易类
    trader = PairsTrading(
        保证金=100000,
        杠杆=5,
        最小手数=0.01,
        X数据={'date':dates, 'close':X_prices},
        Y数据={'date':dates, 'close':Y_prices},
        entry_z=1.5,
        exit_z=0.5
    )
    
    # 模拟实时数据更新
    results = []
    for i in range(len(dates)):
        signal = trader.更新数据(X_prices[i], Y_prices[i], 100000)
        results.append({
            'date': dates[i],
            'X_close': X_prices[i],
            'Y_close': Y_prices[i],
            '信号': signal
        })
    
    # 展示回测结果
    backtest_df = pd.DataFrame(results)
    print(backtest_df.tail())
