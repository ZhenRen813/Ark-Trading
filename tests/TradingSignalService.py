import pandas as pd
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
import numpy as np

class ETHSymbolInfo:
    SYMBOL_NAME = "US 500"
    SYMBOL_LEVERAGE = 20

class BTCSymbolInfo:
    SYMBOL_NAME = "US TECH 100"
    SYMBOL_LEVERAGE = 20

class SymbolInfo:
    def __init__(self, symbolId, symbolName, leverage):
        self.symbolId = symbolId
        self.leverage = leverage
        self.symbolName = symbolName

class TickData:
    def __init__(self, symbol, time, price):
        self.symbol = symbol
        self.time = time
        self.price = price

class BaseTradeSignalService:
    def __init__(self, data, symbols:dict[str,SymbolInfo], window=30, settings=None):
        """初始化配对交易算法类"""
        self.data = data
        self.ticks = {}
        self.symbols = symbols
        self.window = window
        self.settings = settings

        self.zsocres = []

    def update_data(self, ticks:dict[str,TickData]):
        if self.data is None:
            self.data = pd.DataFrame(columns=self.symbols.keys())
        """更新数据"""
        # print(f"更新数据:BTC:{btc_price},ETH:{eth_price},datalen:{len(self.current_data)}")
        if len(self.data) >= self.window:
            self.data = self.data.iloc[1:].reset_index(drop=True)

        # use self.symbols to get symbolName and use the symbolName to update self.data
        new_data = { symbolName: ticks[symbolName].price for symbolName in self.symbols.keys() }
        self.data.loc[len(self.data)] = new_data
        # print(f"更新数据: {self.data.loc[len(self.data)-1]}")
        
    def generate_signal(self,ticks:dict[str,TickData]):
        self.ticks = ticks
        return 0


class PairsTradeSignalService(BaseTradeSignalService):
    def __init__(self, data, symbols:dict[str,SymbolInfo], window=30, settings=None):
        """初始化配对交易算法类"""
        # self.data = data
        # self.ticks = {}
        # self.symbols = symbols
        # self.window = window
        # self.settings = settings
        super().__init__(data, symbols, window, settings)
        self.hedge_ratio = None
        self.zscore_params = {}
        self.entry_z = settings['entry_z']
        self.exit_z = settings['exit_z']
        self.symbol_names = [ symbolName for symbolName in self.symbols.keys() ]

    def update_data(self, ticks:dict[str,TickData]):
        print('suck')
        """更新数据"""
        # print(f"更新数据:BTC:{btc_price},ETH:{eth_price},datalen:{len(self.current_data)}")
        if ticks is not None:
            super().update_data(ticks)

        self._calibrate_model()
    
    def _smooth_spread(self,spread_series, alpha=0.1):
        """指数加权平滑价差序列"""
        ewma = pd.Series(spread_series).ewm(alpha=alpha).mean()
        return ewma


    def _calibrate_model(self):
        """校准协整模型（增加空值保护）"""
        # 保证有足够数据计算
        if len(self.data) < 30:  # 最小数据量要求
            self.hedge_ratio = None
            self.zscore_params = {}
            return
        
        # 打印当前数据量
        print(f"当前数据量: {len(self.data)}")

        handle_data = self.data.copy()
        alpha = 0.5

        if len(self.symbol_names) != 2:
            ValueError("Only two symbols are supported")

        print(f"sysmbol_names: {self.symbol_names}")
        print(handle_data.head())
        # 平滑价差序列
        x_smooth = self._smooth_spread(handle_data[self.symbol_names[0]],alpha)
        y_smooth = self._smooth_spread(handle_data[self.symbol_names[1]],alpha)
        
        # 协整检验
        score, pvalue, _ = coint(x_smooth,y_smooth)
        # print(f"协整检验结果: p={pvalue:.4f}")
        # print(f"协整检验结果: score={score:.4f}")
        if pvalue < 0.05:
            # print(">>>>协整关系有效")
            # 计算对冲比率
            model = sm.OLS(y_smooth, sm.add_constant(x_smooth)).fit()
            self.hedge_ratio = model.params[self.symbol_names[0]]
            
            # 计算价差统计量
            spread = y_smooth - self.hedge_ratio * x_smooth
            self.zscore_params = {
                'mean': spread.mean(),
                'std': spread.std()
            }
            # print(self.zscore_params)
            # log(f"协整关系有效 pvalue:{pvalue},参数:{self.zscore_params}")
            return True
        else:
            print("协整关系无效")
            # 协整关系失效时清除参数
            # self.hedge_ratio = None
            # self.zscore_params = {}
            # 强制平仓
            return False
        
    def generate_signal(self,ticks:dict[str,TickData],hedge_ratio=None,zscore_params=None):
        super().generate_signal(ticks)
        # print(f"data_len:{len(self.data)}")
        """生成交易信号（增加空值检查）"""
        # 检查模型是否有效
        if not hedge_ratio or not zscore_params:
            # print(f"无有效模型,self.hedge_ratio:{self.hedge_ratio},self.zscore_params:{self.zscore_params}")
            return 0  # 无有效模型时返回中性信号
        
        # 检查参数有效性
        if np.isnan(hedge_ratio) or zscore_params['std'] == 0:
            print("参数无效")
            return None

        # 计算当前价差
        x_price = ticks[self.symbol_names[0]].price
        y_price = ticks[self.symbol_names[1]].price
        # print(f"x_price:{x_price},y_price:{y_price}")
        spread = y_price - hedge_ratio * x_price
        # print(f"spread:{spread}")
        
        # 计算Z-Score
        zscore = (spread - zscore_params['mean']) / zscore_params['std']
        self.zsocres.append(zscore)
        # print(f"++++++++++++++++++++++++++++++zscore:{zscore},hedgeratio:{self.hedge_ratio},spread:{spread}++++++++++++++++++++++++++++++++")
        # 生成信号（添加边界检查）
        try:
            if zscore > self.entry_z:
                print(f"zscore:({zscore}) > self.entry_z:{self.entry_z}")
                return -1
            elif zscore < -self.entry_z:
                print(f"zscore:({zscore}) < -self.entry_z:{-self.entry_z}")
                return 1
            elif abs(zscore) < self.exit_z:
                return 0
            else:
                return None
        except:
            return 0