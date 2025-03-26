import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ReversionAlgo import MeanReversionAlgo, TradeDirection

class BacktestEngine:
    def __init__(self, data_path):
        # 读取历史数据
        self.data = pd.read_csv(data_path)
        self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]
        self.data['spread'] = self.data['BTC'] - self.data['ETH']
        
        # 初始化交易算法
        self.algo = MeanReversionAlgo(
            init_margin=1000000,  # 初始保证金100万
            leverage=5,
            min_lots=10,
            risk_exposure=0.3,
            algo_settings={'lookback_period': 20}
        )
        
        # 回测记录
        self.trade_log = []
        self.equity_curve = []
        self.position_history = []

    def run_backtest(self):
        # 滚动窗口回测
        for i in range(len(self.data)):
            if i < self.algo.lookback_period: 
                continue  # 跳过初始化期
                
            # 获取历史数据窗口
            window_data = self.data.iloc[:i+1]
            
            # 更新算法数据
            self.algo.load_data(window_data[['BTC', 'ETH']].values)
            
            # 获取最新价格
            current_price = window_data.iloc[-1][['BTC', 'ETH']].values
            
            # 检查交易信号
            signals = self.algo.check_trade_signal([current_price])
            
            # 执行交易
            if signals:
                if any([s.direction == TradeDirection.LONG for s in signals]):
                    self.algo.open_position(signals)
                else:
                    self.algo.close_position(signals)
                    
            # 记录每日数据
            self._record_daily_data(i)
            
        return self._generate_report()

    def _record_daily_data(self, idx):
        # 记录当日账户信息
        daily_info = {
            'equity': self.algo.available_margin,
            'position_X': self.algo.current_position['X'],
            'position_Y': self.algo.current_position['Y'],
            'spread': self.data.iloc[idx]['spread']
        }
        self.equity_curve.append(daily_info)

    def _generate_report(self):
        # 生成绩效报告
        df = pd.DataFrame(self.equity_curve)
        df['returns'] = df['equity'].pct_change()
        df['cum_returns'] = (1 + df['returns']).cumprod()
        
        # 关键指标计算
        total_return = df['cum_returns'].iloc[-1] - 1
        annualized_return = (df['cum_returns'].iloc[-1] ** (252/len(df))) - 1
        max_drawdown = (df['cum_returns'] / df['cum_returns'].cummax() - 1).min()
        sharpe_ratio = np.sqrt(252) * df['returns'].mean() / df['returns'].std()
        
        return {
            "总收益率": total_return,
            "年化收益率": annualized_return,
            "夏普比率": sharpe_ratio,
            "最大回撤": max_drawdown,
            "交易次数": len(self.trade_log),
            "持仓天数占比": (df[['position_X', 'position_Y']].abs().sum(axis=1) > 0).mean()
        }

# 使用示例
if __name__ == "__main__":
    backtester = BacktestEngine("temp.csv")
    results = backtester.run_backtest()
    
    print("\n回测结果摘要：")
    for k, v in results.items():
        print(f"{k}: {v if isinstance(v, int) else round(v, 4)}")
