import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS


class TradeSignal:
    LONG = 'long'
    SHORT = 'short'
    CLOSE = 'close'
    NoAction = None


class PairsTradingAlgorithm:
    def __init__(self, init_margin, leverage, min_lots, 
                 X_data, Y_data, entry_z, exit_z,
                 risk_exposure=0.3, stop_loss=0.3):
        """
        初始化配对交易算法类
        :param init_margin: 初始保证金
        :param leverage: 杠杆倍数
        :param min_lots: 最小交易手数
        :param X_data: X标的数组 [(timestamp, price)...]
        :param Y_data: Y标的数组 [(timestamp, price)...]
        :param entry_z: 入场阈值
        :param exit_z: 平仓阈值
        :param risk_exposure: 风险敞口比例
        :param stop_loss: 强制平仓线
        """
        # 基本参数初始化
        self.available_margin = init_margin
        self.leverage = leverage
        self.min_lots = min_lots
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss = stop_loss
        self.hedge_ratio = None
        self.current_position = {'X': 0, 'Y': 0}
        self.position_strategy = 'fixed'  # 默认固定头寸策略
        self.risk_exposure = risk_exposure

        # self.load_data(X_data, Y_data)
        
    def load_data(self, X_data, Y_data):
        # 数据处理
        self.X = pd.DataFrame(X_data, columns=['time', 'price'])
        self.Y = pd.DataFrame(Y_data, columns=['time', 'price'])

        self.entry_price = {'X': None, 'Y': None}  # 记录建仓价格
        
        # 协整关系检验
        self._cointegration_test()
        
        # 价差计算
        self._calculate_spread()
        
        # 风险控制参数
        self.max_position = self._calculate_max_position()

    def _cointegration_test(self):
        """协整检验与对冲比率计算"""
        X_prices = self.X['price'].values
        Y_prices = self.Y['price'].values
        
        # 协整检验
        _, pvalue, _ = coint(Y_prices, X_prices)
        if pvalue > 0.05:
            print(f"标的资产未通过协整检验(p={pvalue})")
            
        # 计算对冲比率
        model = OLS(Y_prices, X_prices)
        results = model.fit()
        self.hedge_ratio = results.params

    def _calculate_spread(self):
        """计算价差序列"""
        spread = self.Y['price'] - self.hedge_ratio * self.X['price']
        self.spread_mean = np.mean(spread)
        self.spread_std = np.std(spread)
        
    def _calculate_zscore(self, new_spread):
        """计算标准化价差"""
        return (new_spread - self.spread_mean) / self.spread_std

    def _calculate_max_position(self):
        """计算最大可持仓手数"""
        margin_per_lot = (self.X['price'].iloc[-1] + 
                         self.Y['price'].iloc[-1]) / self.leverage
        return self.available_margin * self.risk_exposure / margin_per_lot
    
    def _calculate_unrealized_pnl(self):

        if self.entry_price['X'] is None or self.entry_price['Y'] is None:
            return 0

        """计算当前持仓未实现盈亏"""
        current_X = self.X['price'].iloc[-1]
        current_Y = self.Y['price'].iloc[-1]
        
        # 获取当前持仓信息
        x_pos = self.current_position['X']
        y_pos = self.current_position['Y']
        
        # 计算标的X的盈亏
        x_pnl = (current_X - self.entry_price['X']) * abs(x_pos) * (-1 if x_pos < 0 else 1)
        
        # 计算标的Y的盈亏
        y_pnl = (current_Y - self.entry_price['Y']) * abs(y_pos) * (-1 if y_pos < 0 else 1)
        
        return x_pnl + y_pnl


    def update_data(self, new_X_price, new_Y_price, new_margin):
        """
        更新市场数据并返回交易指令
        :param new_X_price: X标的最新价格
        :param new_Y_price: Y标的最新价格
        :param new_margin: 最新可用保证金
        """
        self.available_margin = new_margin
        new_spread = new_Y_price - self.hedge_ratio * new_X_price
        zscore = self._calculate_zscore(new_spread)
        
        # 风险检查
        if self._check_stop_loss():
            return self._close_all_positions()
            
        # 生成交易信号
        signal = self._generate_signal(zscore)
        # 在生成交易指令后添加：
        if signal in ['short_spread', 'long_spread']:
            self.entry_price['X'] = new_X_price
            self.entry_price['Y'] = new_Y_price
        elif signal == TradeSignal.CLOSE:
            self.entry_price = {'X': None, 'Y': None}
        return self._execute_trade(signal, new_X_price, new_Y_price)

    def _generate_signal(self, zscore):
        """生成交易信号逻辑"""
        if zscore > self.entry_z:
            return 'short_spread'
        elif zscore < -self.entry_z:
            return 'long_spread'
        elif abs(zscore) < self.exit_z and self.current_position != (0,0):
            return TradeSignal.CLOSE
        else:
            return 'no_action'

    def _execute_trade(self, signal, X_price, Y_price):
        """执行交易指令"""
        # 固定头寸策略逻辑
        if self.position_strategy == 'fixed':
            if self.current_position != (0,0) and signal != TradeSignal.CLOSE:
                return {'action': TradeSignal.NoAction}
                
        # 计算头寸规模
        position_size = self._calculate_position_size(X_price, Y_price)
        
        # 生成交易指令
        if signal == 'short_spread':
            return {
                'X': {'lots': position_size, 'direction': TradeSignal.LONG},
                'Y': {'lots': position_size, 'direction': TradeSignal.SHORT},
                'action': TradeSignal.SHORT
            }
        elif signal == 'long_spread':
            return {
                'X': {'lots': position_size, 'direction': TradeSignal.SHORT},
                'Y': {'lots': position_size, 'direction': TradeSignal.LONG},
                'action': TradeSignal.LONG
            }
        elif signal == TradeSignal.CLOSE:
            return self._close_all_positions()
        else:
            return {'action': TradeSignal.NoAction}

    def _calculate_position_size(self, X_price, Y_price):
        """计算头寸规模"""
        margin_required = (X_price + Y_price) / self.leverage
        max_lots = self.available_margin / margin_required
        return max(self.min_lots, round(min(max_lots, self.max_position), 2))

    def _close_all_positions(self):
        """平仓所有头寸"""
        close_order = {
            'X': {'lots': abs(self.current_position['X']), 
                 'direction': TradeSignal.SHORT if self.current_position['X']>0 else TradeSignal.LONG},
            'Y': {'lots': abs(self.current_position['Y']), 
                 'direction': TradeSignal.LONG if self.current_position['Y']>0 else TradeSignal.SHORT},
            'action': TradeSignal.CLOSE
        }
        self.current_position = {'X':0, 'Y':0}
        return close_order

    def _check_stop_loss(self):
        """检查强制平仓条件"""
        unrealized_pnl = self._calculate_unrealized_pnl()
        total_equity = self.available_margin + unrealized_pnl
        return (total_equity / self.available_margin) < (1 - self.stop_loss)

    def switch_strategy(self, strategy_name):
        """切换头寸策略"""
        valid_strategies = ['fixed', 'independent']
        if strategy_name in valid_strategies:
            self.position_strategy = strategy_name
        else:
            raise ValueError("无效策略名称")

# 数据回测示例
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    X_prices = np.cumsum(np.random.normal(0.1, 0.5, 100)) + 50
    Y_prices = X_prices * 1.2 + np.random.normal(0, 0.5, 100)
    
    xdata = list(zip(dates, X_prices))
    ydata = list(zip(dates, Y_prices))
    

    # 初始化策略
    algo = PairsTradingAlgorithm(
        init_margin=100000,
        leverage=5,
        min_lots=0.01,
        X_data=xdata,
        Y_data=ydata,
        entry_z=1.5,
        exit_z=0.5
    )
    
    algo.load_data(xdata, ydata)

    # 执行回测
    history = []
    for i in range(len(X_prices)):
        trade_signal = algo.update_data(
            new_X_price=X_prices[i],
            new_Y_price=Y_prices[i],
            new_margin=100000  # 假设保证金不变
        )
        # if trade_signal['action'] is not None:
        #     print(trade_signal)
        history.append({
            'date': dates[i],
            'X_price': X_prices[i],
            'Y_price': Y_prices[i],
            'signal': trade_signal
        })
    
    # 输出回测结果
    print(pd.DataFrame(history).tail())
