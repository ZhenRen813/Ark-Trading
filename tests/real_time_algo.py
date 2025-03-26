import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import sys
sys.path.append('/Users/aaron/Development/trade/waterstart')
from waterstart.client import OpenApiClient

class PairsTradingRealtime:

    window: int
    initial_window: int
    calibrate_counter: int
    reset_size: int
    in_position: bool = False
    entry_z: float
    exit_z: float
    client: OpenApiClient

    def __init__(self, client = None,initial_window=1000, window=1500, reset_size=30, 
                 entry_z=2.0, exit_z=0.5):
        # ...原有初始化代码...
        
        # 新增历史数据初始化标志
        self.is_initialized = False
        self.window = window
        self.initial_window = initial_window
        self.calibrate_counter = 0
        self.reset_size = reset_size
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.client = client
        self.hedge_ratio = None
        self.data = None
        self.allData = None

    def initialize_history(self, historical_data: pd.DataFrame):
        """
        初始化历史数据（必须包含BTC和ETH价格）
        
        参数:
        historical_data - DataFrame需包含:
            index: 时间戳 (datetime类型)
            columns: ['BTC', 'ETH'] 价格列
        """
        # 数据校验
        # if not isinstance(historical_data.index, pd.DatetimeIndex):
        #     raise ValueError("索引必须是时间戳类型")
        
        if not {'BTC', 'ETH'}.issubset(historical_data.columns):
            raise ValueError("数据必须包含BTC和ETH价格列")
        
        # 按时间排序并去重
        clean_data = historical_data[['BTC', 'ETH']].sort_index().dropna()
        
        # 保留最近的window条数据
        self.data = clean_data.iloc[-self.window:]
        self.allData = clean_data
        print(f"历史数据初始化完成，当前数据量: {len(self.data)}")
        
        # 立即进行首次校准
        self._calibrate_model()
        self.is_initialized = True

    # 修改update_data方法
    def update_data(self, btc_price: float, eth_price: float, timestamp: pd.Timestamp):
        """更新实时数据"""
        print(f"更新数据: BTC={btc_price}, ETH={eth_price}")
        # 转换为正确的时间类型
        # if not isinstance(timestamp, pd.Timestamp):
        #     timestamp = pd.to_datetime(timestamp).minute
            
        new_row = pd.DataFrame({'BTC': [btc_price], 'ETH': [eth_price]}, 
                              index=[timestamp])
        self.data = pd.concat([self.data, new_row]).tail(self.window)
        self.allData = pd.concat([self.allData, new_row])
        
        # 首次校准检查
        if not self.is_initialized and len(self.data) >= self.initial_window:
            self._calibrate_model()
            self.is_initialized = True
        
        self.calibrate_counter += 1

        # 定期重新校准模型
        if self.calibrate_counter >= self.reset_size:
            self._calibrate_model()
            self.calibrate_counter = 0

    def _write_all_data_to_file(self):
        self.allData.to_csv('trader_allData.csv')

    def _calibrate_model(self):
        """
        执行模型校准（协整检验和对冲比率计算）
        """
        if len(self.data) < self.initial_window:
            self._reset_model()
            return

        # 协整检验
        try:
            score, pvalue, _ = coint(self.data['BTC'], self.data['ETH'])
        except:
            self._reset_model()
            return

        if pvalue < 0.05:
            print(f"协整检验通过，p值: {pvalue:.4f}")
            # 计算对冲比率
            model = sm.OLS(self.data['ETH'], sm.add_constant(self.data['BTC'])).fit()
            self.hedge_ratio = model.params['BTC']
            
            # 计算价差统计量
            spread = self.data['ETH'] - self.hedge_ratio * self.data['BTC']
            self.zscore_params = {
                'mean': spread.mean(),
                'std': max(spread.std(), 1e-8)  # 防止零标准差
            }
        else:
            print(f"协整检验未通过，p值: {pvalue:.4f}")
            self._reset_model(close_position=True)

    def _reset_model(self, close_position=False):
        """
        重置模型参数（协整关系失效时调用）
        """
        self.hedge_ratio = None
        self.zscore_params = {}
        if close_position and self.in_position:
            # self._generate_signal(force_close=True)
            self.force_close()

    def get_signal(self):
        """
        生成当前交易信号
        {
            'signal': 'long'/'short'/'close'/None,
            'zscore': float,
            'hedge_ratio': float,
            'spread': float
        }
        """
        # 检查模型有效性
        if self.hedge_ratio is None or not self.zscore_params:
            return {'signal': None, 'zscore': None, 'hedge_ratio': None, 'spread': None}

        # 计算当前价差
        current_btc = self.data['BTC'].iloc[-1]
        current_eth = self.data['ETH'].iloc[-1]
        spread = current_eth - self.hedge_ratio * current_btc
        # 打印spread计算公式
        print(f"spread: {current_eth} - {self.hedge_ratio} * {current_btc} = {spread}")
        zscore = (spread - self.zscore_params['mean']) / self.zscore_params['std']

        signal = None

        # 平仓逻辑
        if self.in_position:
            if (self.position_type == 'long' and zscore >= -self.exit_z) or \
               (self.position_type == 'short' and zscore <= self.exit_z):
                signal = 'close'
                self.in_position = False
                self.position_type = None
        # 开仓逻辑
        else:
            if zscore > self.entry_z:
                signal = 'short'
                self.in_position = True
                self.position_type = 'short'
            elif zscore < -self.entry_z:
                signal = 'long'
                self.in_position = True
                self.position_type = 'long'

        return {
            'signal': signal,
            'zscore': zscore,
            'hedge_ratio': self.hedge_ratio,
            'spread': spread,
            'timestamp': self.data.index[-1]
        }

    def check_signal(self, btc_price: float, eth_price: float, timestamp: pd.Timestamp,need_update=False):
        """
        完整处理流程：更新数据 + 生成信号
        """
        print(f"检查信号: BTC={btc_price}, ETH={eth_price}")
        if need_update:
            self.update_data(btc_price, eth_price, timestamp)
        signal = self.get_signal()
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(signal)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return signal
    
    # 强制平仓
    def force_close(self):
        print("强制平仓")
        if self.in_position:
            self.in_position = False
            self.position_type = None

# # 使用示例
# if __name__ == "__main__":
#     # 初始化实时交易判断器
#     trader = PairsTradingRealtime(
#         initial_window=1000,
#         window=1500,
#         reset_size=30,
#         entry_z=2.0,
#         exit_z=0.5
#     )

#     # 模拟实时数据流
#     for i in range(2000):
#         # 获取实时价格（这里用随机数据演示）
#         btc_price = np.random.normal(50000, 1000)
#         eth_price = np.random.normal(3000, 100)
#         timestamp = pd.Timestamp.now() + pd.Timedelta(minutes=i)
        
#         # 获取交易信号
#         signal = trader.check_signal(btc_price, eth_price, timestamp)
        
#         # 处理交易信号
#         if signal['signal'] == 'long':
#             print(f"{timestamp} [做多信号] Z-Score: {signal['zscore']:.2f}")
#         elif signal['signal'] == 'short':
#             print(f"{timestamp} [做空信号] Z-Score: {signal['zscore']:.2f}")
#         elif signal['signal'] == 'close':
#             print(f"{timestamp} [平仓信号] Z-Score: {signal['zscore']:.2f}")