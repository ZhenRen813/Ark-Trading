import numpy as np
import pandas as pd
from data_wapper import TradeDirection, TradePositionSignal,TradingAlgorithm

class MeanReversionAlgo(TradingAlgorithm):
    def __init__(self, init_margin, leverage, min_lots, 
                 risk_exposure=0.3, stop_loss=0.3, algo_settings=None):
        super().__init__(init_margin, leverage, min_lots, 
                        risk_exposure, stop_loss, algo_settings)
        
        # 均值回归特有参数
        self.lookback_period = 20    # 均值计算周期
        self.entry_z = 1.5          # 入场阈值（标准差倍数）
        self.exit_z = 0.5           # 离场阈值
        self.dynamic_position = True  # 是否动态调整头寸
        self.spread_mean = None
        self.spread_std = None
        self.current_spread = None

    def load_data(self, datas):
        """加载并预处理数据，计算价差"""
        super().load_data(datas)
        
        # 计算价差序列（示例：假设双标的价差）
        df = pd.DataFrame(datas, columns=['date', 'X', 'Y'])
        df['spread'] = df['X'] - df['Y']
        
        # 计算移动均值和标准差
        df['spread_mean'] = df['spread'].rolling(self.lookback_period).mean()
        df['spread_std'] = df['spread'].rolling(self.lookback_period).std()
        
        self.datas = df
        self._update_spread_stats(df.iloc[-1])

    def _update_spread_stats(self, latest_data):
        """更新价差统计值"""
        self.spread_mean = latest_data['spread_mean']
        self.spread_std = latest_data['spread_std']
        self.current_spread = latest_data['spread']

    def check_trade_signal(self, prices):
        """生成均值回归交易信号"""
        current_price = prices[-1]  # 获取最新价格数据
        self._update_spread_stats(current_price)
        
        signals = []
        z_score = (self.current_spread - self.spread_mean) / self.spread_std

        # 生成开仓信号
        if abs(z_score) > self.entry_z:
            direction = TradeDirection.SHORT if z_score > 0 else TradeDirection.LONG
            symbol = 'X' if direction == TradeDirection.LONG else 'Y'
            lots = self._calculate_position_size()
            signals.append(TradePositionSignal(symbol, lots, direction))
        
        # 生成平仓信号
        elif abs(z_score) < self.exit_z and (self.current_position['X'] !=0 or self.current_position['Y'] !=0):
            self.close_position(signals)
        
        return signals

    def _calculate_position_size(self):
        """计算头寸规模（基于风险敞口）"""
        # 可用保证金计算
        max_risk = self.available_margin * self.risk_exposure
        
        # 波动性调整（ATR或价差标准差）
        volatility = self.spread_std if self.dynamic_position else 1
        contract_value = self.current_spread  # 假设标的合约价值等于价差
        
        # 计算手数
        raw_lots = (max_risk * self.leverage) / (volatility * contract_value)
        adjusted_lots = max(round(raw_lots), self.min_lots)
        
        return min(adjusted_lots, self.available_margin // contract_value)

    def open_position(self, signals):
        """执行开仓操作"""
        for signal in signals:
            # 更新保证金和头寸
            margin_required = signal.lots * self.current_spread / self.leverage
            self.available_margin -= margin_required
            self.current_position[signal.symbol] += signal.lots * signal.direction
            
        # 记录交易日志
        self._log_trade('OPEN', signals)

    def close_position(self, signals):
        """执行平仓操作"""
        # 计算平仓收益
        for symbol in ['X', 'Y']:
            if self.current_position[symbol] != 0:
                position_value = self.current_position[symbol] * self.current_spread
                profit = position_value * (1 - 1/self.leverage)
                self.available_margin += profit
                self.current_position[symbol] = 0
        
        # 记录交易日志
        self._log_trade('CLOSE', signals)

    def _log_trade(self, action, signals):
        """交易日志记录（示例）"""
        print(f"{action} Position at {pd.Timestamp.now()}")
        for s in signals:
            print(f" - {s.symbol}: {s.lots} lots @ {self.current_spread}")
