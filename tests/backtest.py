import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import datetime
import gc
from mockTrading import MockTradingSingle,MockTrading
import sys
sys.path.append('../src')
from TradingSignalService import SymbolInfo,TickData, PairsTradeSignalService, XSymbolInfo, YSymbolInfo
from MarginService import Position,MarginManager

def log(msg):
    print(msg)
    # write to log file
    with open('tradinglogP0.log', 'a') as f:
        f.write(f"{datetime.datetime.now()}: {msg}\n")

class MarginParams:
    MAX_POSITION_Value = 5000


market_prices = {XSymbolInfo.SYMBOL_NAME:0,YSymbolInfo.SYMBOL_NAME:0}

def closeOrder():
    # self.print("平仓")
    MockTradingSingle.shared._closeOredr(market_prices)

def createOrder(X_size,Y_size,X_price,Y_price):
    # self.print(f"开仓:X:{X_size} Y:{Y_size} X价格:{X_price} Y价格:{Y_price}")
    MockTradingSingle.shared._createOrder(X_size,Y_size,X_price,Y_price)

# def closeOrder():
#     self.print("平仓")

# def createOrder(X_size,Y_size,X_price,Y_price):
#     self.print(f"开仓:X:{X_size} Y:{Y_size} X价格:{X_price} Y价格:{Y_price}")


class PairsTradingBacktester:


    def print(self, msg):
        """打印日志"""
        print(msg)

    createOrder = 0
    closeOrder = 0

    def __init__(self,entry_z=1.5, exit_z=0.5, window = 180, initial_cash=1500, closeOrderExcute = None ,createOrderExcute = None):
        """
        参数:
        Y_data, X_data - 包含时间戳和收盘价的数据框
        initial_window - 初始训练窗口大小
        initial_cash - 初始资金
        """
        
        self.current_data = None
        self.hedge_ratio = None
        self.zscore_params = {}
        
        # 投资组合状态
        self.portfolio = {
            'cash': initial_cash,
            'Y_position': 0,
            'X_position': 0,
            'total_value': initial_cash,
            'history': []
        }
        self.trade_log = []
        self.entry_z = entry_z  # 保存参数
        self.exit_z = exit_z
        self.closeOrderExcute = closeOrderExcute
        self.createOrderExcute = createOrderExcute
        self.Y_price = None
        self.X_price = None
        self.window = window
        self.alreadyOpen = False
        self.used_margin = 0
        self.leverage = 100
        self.positions = []
        self.balance = initial_cash
        self.report = None
        self.service = PairsTradeSignalService(data=None,symbols={YSymbolInfo.SYMBOL_NAME:SymbolInfo(YSymbolInfo.SYMBOL_NAME,YSymbolInfo.SYMBOL_NAME,YSymbolInfo.SYMBOL_LEVERAGE),
                                                                  XSymbolInfo.SYMBOL_NAME:SymbolInfo(XSymbolInfo.SYMBOL_NAME,XSymbolInfo.SYMBOL_NAME,XSymbolInfo.SYMBOL_LEVERAGE)},window=window,settings={'entry_z':self.entry_z,'exit_z':self.exit_z})
        self.margin_manager = MarginManager(total_balance=initial_cash)

    def run_backtest_2(self,df, window=1500):
        # self.service.data = df.rename(columns={YSymbolInfo.SYMBOL_NAME:YSymbolInfo.SYMBOL_NAME,XSymbolInfo.SYMBOL_NAME:XSymbolInfo.SYMBOL_NAME})
        self._backtest_with_df(df, window)

    def run_backtest(self,Y_data, X_data, window=1500):

        # 合并数据
        self.data = pd.DataFrame({
            YSymbolInfo.SYMBOL_NAME: Y_data,
            XSymbolInfo.SYMBOL_NAME: X_data
        }).dropna()
        self._backtest_with_df(self.data, window)
    
    def _backtest_with_df(self, data, window=1500):
        """主回测逻辑"""
        self.data = data
        self.window = window
        for i in range(self.window, len(self.data)):
            # 获取当前数据窗口（滚动扩展）,窗口最大为1500条记录
            # 修改获取 current_data 的逻辑
            start_idx = max(0, i - self.window)
            if self.current_data is None or len(self.current_data) < self.window:
                # self.print("初始化数据")
                self.current_data = self.data.iloc[start_idx:i+1]
                continue
            else:
                if i+1 < len(self.data):
                    Y_price = self.data[YSymbolInfo.SYMBOL_NAME].iloc[i+1]
                    etc_price = self.data[XSymbolInfo.SYMBOL_NAME].iloc[i+1]
                    self._update_data(self.data[YSymbolInfo.SYMBOL_NAME].iloc[i+1], self.data[XSymbolInfo.SYMBOL_NAME].iloc[i+1])
                    self._checkUpdateSignal(reset_model=True,Y_price=Y_price,X_price=etc_price)
                else:
                    break
        self._generate_report()

    def _update_balance(self, cash):
                # 投资组合状态
        if cash < 0:
            exit(0)
        self.portfolio['cash'] = cash/100.0 * 100
        self.balance = cash/100.0 * 100
        self.margin_manager.total_balance = cash/100.0 * 100

    def _reload_data(self, Y_data, X_data):
        self.print("重新加载数据")
        """重新加载数据"""
        self.data = pd.DataFrame({
            YSymbolInfo.SYMBOL_NAME: Y_data['close'],
            XSymbolInfo.SYMBOL_NAME: X_data['close']
        }).dropna()
        self.current_data = self.data
        self.print(f"重新加载数据:Y:{Y_data['close']},X:{X_data['close']},datalen:{len(self.current_data)}")
        self.service.data = pd.DataFrame({
            YSymbolInfo.SYMBOL_NAME: Y_data['close'],
            XSymbolInfo.SYMBOL_NAME: X_data['close']
        }).dropna()
        self.service.update_data(None)

    def _update_used_margin(self, margin):
        self.used_margin = margin
        # self.portfolio['cash'] = self.balance - self.used_margin

    def _update_data(self, Y_price, X_price):
        """更新数据"""
        # self.print(f"更新数据:Y:{Y_price},X:{X_price},datalen:{len(self.current_data)}")
        if len(self.current_data) >= self.window:
            # self.current_data = self.current_data.drop(index=self.current_data.index[0])
            self.current_data = self.current_data.iloc[1:].reset_index(drop=True)
        self.current_data.loc[len(self.current_data)] = {
            YSymbolInfo.SYMBOL_NAME: Y_price,
            XSymbolInfo.SYMBOL_NAME: X_price,
        }
        self.data.loc[len(self.data)] = {
            YSymbolInfo.SYMBOL_NAME: Y_price,
            XSymbolInfo.SYMBOL_NAME: X_price,
        }
        self.Y_price = Y_price
        self.X_price = X_price
        market_prices[XSymbolInfo.SYMBOL_NAME] = self.X_price
        market_prices[YSymbolInfo.SYMBOL_NAME] = self.Y_price

        self.service.update_data({YSymbolInfo.SYMBOL_NAME:TickData(YSymbolInfo.SYMBOL_NAME,datetime.datetime.now().timestamp(),Y_price),XSymbolInfo.SYMBOL_NAME:TickData(XSymbolInfo.SYMBOL_NAME,datetime.datetime.now().timestamp(),X_price)})
    
    def _write_all_data_to_file(self):
        self.data.to_csv('algo_allData.csv')

    def _checkUpdateSignal(self,reset_model = False,Y_price = None, X_price = None):
        self.margin_manager.update_price(YSymbolInfo.SYMBOL_NAME, Y_price, Y_price)
        self.margin_manager.update_price(XSymbolInfo.SYMBOL_NAME, X_price, X_price)
        # TODO: check why 'cash' < 0
        if len(self.current_data) < 30 or self.margin_manager.get_available_margin() <= 0 or Y_price is None or X_price is None:  # 最小数据量要求
            self.print(f"数据不足,当前数据量:{len(self.current_data)},cash:{self.portfolio['cash']},Y_price:{Y_price},X_price:{X_price}")
            return None

        # self.print(f">>>>>>>>>>>>>>>>>>>>>>>>>cash:{self.portfolio['cash']},balance:{self.balance},used_margin:{self.used_margin},hedge_ratio:{self.hedge_ratio},zsocre_params:{self.zscore_params}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        # 生成交易信号
        signal = self._generate_signal(Y_price, X_price)
        
        self._execute_trade(signal,Y_price, X_price)
        
        # 更新投资组合价值
        self._update_portfolio(Y_price, X_price)
        
        # 记录每日状态
        self._record_state(Y_price, X_price)

        return signal
    
    def smooth_spread(self,spread_series, alpha=0.1):
        """指数加权平滑价差序列"""
        ewma = pd.Series(spread_series).ewm(alpha=alpha).mean()
        return ewma

    def _generate_signal(self,Y_price = None, X_price = None):
        time = datetime.datetime.now().timestamp()
        ticks:dict[str,TickData] = {
            YSymbolInfo.SYMBOL_NAME:TickData(YSymbolInfo.SYMBOL_NAME,time,Y_price),
            XSymbolInfo.SYMBOL_NAME:TickData(XSymbolInfo.SYMBOL_NAME,time,X_price)
        }
        return self.service.generate_signal(ticks,self.service.hedge_ratio,self.service.zscore_params)
    
    def _last_Date(self):
        return self.current_data.index[-1]

    def _execute_trade(self, signal,_Y_price = None, _X_price = None):
        """执行交易（增加空值保护）"""
        # 检查价格有效性
        try:
            Y_price = float(_Y_price)
            X_price = float(_X_price)
        except (TypeError, ValueError):
            return
        
        # 跳过无效价格
        if np.isnan(_Y_price) or np.isnan(_X_price):
            return
        """执行交易"""
        Y_price = _Y_price
        X_price = _X_price
        self.margin_manager.update_price(YSymbolInfo.SYMBOL_NAME, Y_price, Y_price)
        self.margin_manager.update_price(XSymbolInfo.SYMBOL_NAME, X_price, X_price)
        
        # 平仓逻辑
        if signal == 0 and (self.portfolio['Y_position'] != 0 or self.portfolio['X_position'] != 0):
            self._close_positions(Y_price, X_price)
        
        # 开仓逻辑
        if signal in [-1, 1]:
            if self.alreadyOpen:
                return
            if self.margin_manager.get_available_margin() < MarginParams.MAX_POSITION_Value:
                self.print(f"保证金不足==================================,used_margin:{self.used_margin},balance:{self.balance}")
                return
            # 计算头寸规模（保持美元中性）
            position_value = MarginParams.MAX_POSITION_Value#self.portfolio['cash'] / 2 #* 36
            self.print(f"position_value:{position_value}")
            # 仓位精度为两个小数点，价格精度为三个小数点
            # Y_size = position_value / Y_price
            # X_size = position_value / X_price
            if abs(self.service.hedge_ratio) < 1:
                Y_size = YSymbolInfo.MIN_LOT_SIZE
                X_size = Y_size / self.service.hedge_ratio #self.hedge_ratio
            else:
                X_size = XSymbolInfo.MIN_LOT_SIZE
                Y_size = X_size * self.service.hedge_ratio #self.hedge_ratio
            es_margin1 = self.margin_manager.calculate_required_margin(YSymbolInfo.SYMBOL_NAME, Y_size, 1 if signal == 1 else -1, YSymbolInfo.SYMBOL_LEVERAGE)
            es_margin2 = self.margin_manager.calculate_required_margin(XSymbolInfo.SYMBOL_NAME, X_size, 1 if signal == 1 else -1, XSymbolInfo.SYMBOL_LEVERAGE)
            self.print(f"es_margin{es_margin1 + es_margin2}")
            times = round(position_value / (es_margin1 + es_margin2),0)
            Y_size = Y_size * times
            X_size = X_size * times
            # self.print("对冲比率",self.service.hedge_ratio,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",Y_size,X_size)
            # times = 0

            # while (abs(Y_size) * abs(Y_price)) + (abs(X_size) * abs(X_price)) * times < position_value:
            #     times += 1
            # if times == 0:
            #     self.print("不够钱==================================",Y_price,X_price)
            #     return
            # Y_size = Y_size * times
            # X_size = X_size * times

            if Y_size < YSymbolInfo.MIN_LOT_SIZE or X_size < XSymbolInfo.MIN_LOT_SIZE:
                log(f"头寸过小{Y_size},{X_size},取消交易,position_value:{position_value},time:{datetime.datetime.now()},Y_price:{Y_price},X_price:{X_price}")
                return
            
            Y_size = round(Y_size, 2)
            X_size = round(X_size, 2)

            self.createOrder += 1

            margin1 = self.margin_manager.calculate_required_margin(YSymbolInfo.SYMBOL_NAME, Y_size, 1 if signal == 1 else -1, YSymbolInfo.SYMBOL_LEVERAGE)
            margin2 = self.margin_manager.calculate_required_margin(XSymbolInfo.SYMBOL_NAME, X_size, 1 if signal == 1 else -1, XSymbolInfo.SYMBOL_LEVERAGE)
            if margin1 + margin2 > MarginParams.MAX_POSITION_Value:
                log(f"保证金不足==================================,margin1:{margin1},margin2:{margin2}used_margin:{self.used_margin},balance:{self.balance}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                return
            
            if signal == 1:  # 做多价差
                self._open_position(X_size, -Y_size, X_price, Y_price)
            else:  # 做空价差
                self._open_position(-X_size, Y_size, X_price, Y_price)

            
            self._update_used_margin(margin1 + margin2)
            self.alreadyOpen = True

    def _open_position(self, X_size, Y_size, X_price, Y_price):
        """开仓操作"""
        # X交易
        X_cost = X_size * X_price
        self.portfolio['cash'] -= X_cost
        # self.portfolio['cash'] -= MockTradingSingle.shared.calculate_reqired_margin(1, X_size, X_price)
        self.portfolio['X_position'] += X_size
        
        # Y交易
        Y_cost = Y_size * Y_price
        self.portfolio['cash'] -= Y_cost
        # self.portfolio['cash'] -= MockTradingSingle.shared.calculate_reqired_margin(1, Y_size, Y_price)
        self.portfolio['Y_position'] += Y_size

        self.margin_manager.add_position(XSymbolInfo.SYMBOL_NAME, abs(X_size), 1 if X_size > 0 else -1, X_price, XSymbolInfo.SYMBOL_LEVERAGE)
        self.margin_manager.add_position(YSymbolInfo.SYMBOL_NAME, abs(Y_size), 1 if Y_size > 0 else -1, Y_price, YSymbolInfo.SYMBOL_LEVERAGE)
        
        self.createOrderExcute(X_size,Y_size,X_price,Y_price)

        # 记录交易日志到本地
        if X_size != 0 and Y_size != 0:
            log(f"time:{datetime.datetime.now} OPEN: X_Size:{X_size} Y_Size:{Y_size} X价格:{X_price} Y价格:{Y_price}")

        self.trade_log.append({
            'date': self._last_Date(),
            'action': 'OPEN',
            'X_size': X_size,
            'Y_size': Y_size,
            'X_price': X_price,
            'Y_price': Y_price
        })

    def _close_positions(self, Y_price, X_price):
        if Y_price is None or X_price is None:
            log(f"价格为空,取消平仓")
            return
        """平仓操作"""
        # 平仓X
        X_size = self.portfolio['X_position']
        X_value = X_size * X_price
        # self.portfolio['cash'] += X_value
        # self.portfolio['cash'] += MockTradingSingle.shared.calculate_released_margin(self.leverage, self.portfolio['X_position'], X_price)
        self.portfolio['X_position'] = 0
        
        # 平仓Y
        Y_size = self.portfolio['Y_position']
        Y_value = Y_size * Y_price
        # self.portfolio['cash'] += Y_value
        self.portfolio['Y_position'] = 0
        
        self.trade_log.append({
            'date': self._last_Date(),
            'action': 'CLOSE',
            'X_price': X_price,
            'Y_price': Y_price
        })

        # 记录交易日志到本地
        if X_value != 0 and Y_value != 0:
            log(f"{datetime.datetime.now()}: CLOSE X_Size:{X_size} Y_Size:{Y_size} X价格:{X_price} Y价格:{Y_price}\n")

        self.closeOrder += 1
        self.closeOrderExcute()
        self.margin_manager.close_all_positions()
        self._update_used_margin(0)
        self.portfolio['cash'] = self.margin_manager.get_available_margin()
        if self.margin_manager.get_available_margin() < 0:
            self.print(f"保证金不足==================================,used_margin:{self.used_margin},balance:{self.balance}")
            self.print(self.margin_manager.get_all_current_margin())
            exit(0)
        self.alreadyOpen = False

    def _update_portfolio(self,Y_price = None,X_price = None):
        """更新投资组合价值"""
        Y_value = self.portfolio['Y_position'] * Y_price
        X_value = self.portfolio['X_position'] * X_price
        self.portfolio['total_value'] = self.portfolio['cash']+ Y_value + X_value
        # self.print(f"更新投资组合价值: {self.portfolio['total_value']:.2f}")
        if Y_value != 0 or X_value != 0:
            self.print(f"更新投资组合价值: {self.portfolio['total_value']:.2f},cash:{self.portfolio['cash']:.2f},Y_value:{Y_value:.2f},X_value:{X_value:.2f}")

    def _record_state(self,Y_price = None,X_price = None):
        """记录每日状态"""
        self.portfolio['history'].append({
            'date': self._last_Date(),
            'total_value': self.portfolio['total_value'],
            'Y_price': Y_price,
            'X_price': X_price,
            'Y_position': self.portfolio['Y_position'],
            'X_position': self.portfolio['X_position']
        })
        # self.print(f"记录每日状态: {self._last_Date()},total_value:{self.portfolio['total_value']:.2f},cash:{self.portfolio['cash']:.2f},Y_price:{Y_price:.2f},X_price:{X_price:.2f},Y_position:{self.portfolio['Y_position']},X_position:{self.portfolio['X_position']}")

    def _generate_report(self):

        if self.createOrder == 0 or self.closeOrder == 0:
            self.print("无交易记录")
            return

        """生成分析报告"""
        history_df = pd.DataFrame(self.portfolio['history'])
        trade_df = pd.DataFrame(self.trade_log)
        
        # 计算收益
        self.print("计算收益")
        history_df['returns'] = history_df['total_value'].pct_change()
        history_df['cum_returns'] = (1 + history_df['returns']).cumprod()
        
        # 价差和仓位
        self.print("价差和仓位")
        # 移除掉history_df的X_price和Y_price的空值
        history_df = history_df.dropna(subset=['X_price', 'Y_price'])
        # 裁剪数据将history_df的X_price和Y_price长度对齐
        history_df = history_df.iloc[-len(self.current_data):]

        # 计算关键指标
        self.print("计算关键指标")
        total_return = history_df['cum_returns'].iloc[-1] - 1
        sharpe = self._calculate_sharpe(history_df['returns'])
        max_drawdown = self._calculate_max_drawdown(history_df['total_value'])
        
        self.print(f"策略总收益: {total_return:.2%}")
        self.print(f"夏普比率: {sharpe:.2f}")
        self.print(f"最大回撤: {max_drawdown:.2%}")
        self.print(f"总交易次数: {len(trade_df)}")
        self.report = {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(trade_df)
        }

        # 将 self.service.zsocres 绘制成折线图
        self.print(f"绘制折线图{len(self.service.zsocres)}")
        # plt.plot(self.service.zsocres)
        # plt.show()
        
    def _calculate_sharpe(self, returns, risk_free=0):
        """计算年化夏普比率"""
        excess_returns = returns - risk_free/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_max_drawdown(self, values):
        """计算最大回撤"""
        peak = values.cummax()
        drawdown = (values - peak) / peak
        return drawdown.min() 


# 示例使用
if __name__ == "__main__":

    # 初始化回测器
    
    # read trader_allData.csv
    # all_data = pd.read_csv('data0410.csv')
    all_data = pd.read_csv('data0411.csv')
    # data3 = pd.read_csv('data0412.csv')
    # all_data = data1 + data2 + data3
    # all_data = pd.concat([data1, data2, data3], ignore_index=False)
    all_data[[YSymbolInfo.SYMBOL_NAME]] = all_data[[YSymbolInfo.SYMBOL_NAME]]
    all_data[[XSymbolInfo.SYMBOL_NAME]] = all_data[[XSymbolInfo.SYMBOL_NAME]]

    # use first 1000 records
# 策略总收益: 0.32%
# 夏普比率: 2.26
# 最大回撤: 0.00%
# 总交易次数: 18
# 建仓次数:12
# 平仓次数:6
    #all_data = all_data.tail(1000)

    entry_z_arr = [1.7,1.8,1.9,2.0]
    windows = [160,170,180,190,200,300]

    for entry_z in entry_z_arr:
        for window in windows:
            # MockTradingSingle.shared = MockTrading(balance=1000000,leverage=100)
            print(f"window:{window}  entry_z={entry_z}")
            # backtester = PairsTradingBacktester(entry_z=entry_z, exit_z=0.5,window=window,closeOrderExcute=closeOrder,createOrderExcute=createOrder)
            # backtester.run_backtest_2(all_data, window=window)
            # self.print(f"建仓次数:{backtester.createOrder}")
            # self.print(f"平仓次数:{backtester.closeOrder}")
            # MockTradingSingle.shared._generate_report()
            # self.print("=========================================================")
            # # 测试结果写入日志
            # with open('tradinglogPxxx.log', 'a') as f:
            #     f.write(f"window:{window}  entry_z={entry_z}\n")
            #     f.write(f"建仓次数:{backtester.createOrder}\n")
            #     f.write(f"平仓次数:{backtester.closeOrder}\n")
            #     f.write(f"策略总收益: {backtester.report['total_return']:.2%}\n")
            #     f.write(f"report: {backtester.report}\n")
            #     f.write("=========================================================\n")
            # del backtester
            # del MockTradingSingle.shared
            # gc.collect()
    MockTradingSingle.shared = MockTrading(balance=5000,leverage=100)
    backtester = PairsTradingBacktester(entry_z=1.8, exit_z=0.7, initial_cash= 5000,closeOrderExcute=closeOrder,createOrderExcute=createOrder)
    backtester.run_backtest_2(all_data, window=180)

    
    print(f"建仓次数:{backtester.createOrder}")
    print(f"平仓次数:{backtester.closeOrder}")

    MockTradingSingle.shared._generate_report()

    # 35天数据动态协整
    # window:160  entry_z=1.5 exit_z=0.5 策略总收益: 2.17% 夏普比率: -0.82 最大回撤: -0.13%
    # window:180  entry_z=1.5 exit_z=0.5 策略总收益: 6.44% 夏普比率: -0.68 最大回撤: -0.18%
    # window:180  entry_z=1.5 exit_z=0.6 策略总收益: 6.65% 夏普比率: -0.68 最大回撤: -0.18%
    # window:180  entry_z=1.5 exit_z=0.7 策略总收益: 7.53% 夏普比率: -0.67 最大回撤: -0.18%
    # window:180  entry_z=1.5 exit_z=0.8 策略总收益: 6.51% 夏普比率: -0.68 最大回撤: -0.18%
    # window:180  entry_z=2.0 exit_z=0.5 策略总收益: 5.96% 夏普比率: -0.51 最大回撤: -0.14%
    # window:200  entry_z:1.5 exit_z:0.5 策略总收益: 5.13% 夏普比率: -0.48 最大回撤: -0.10%
    # window:400  entry_z:1.5 exit_z:0.5 策略总收益: 5.04% 夏普比率: 0.14 最大回撤: -0.08%
    # window:500  entry_z=2.0 exit_z=0.5 策略总收益: -3.52% 夏普比率: -0.46 最大回撤: -1.46%
    # window:600  entry_z=2.0 exit_z=0.5 策略总收益: -33.36% 夏普比率: -0.16 最大回撤: -3.43%
    # window:700  entry_z=2.0 exit_z=0.5 策略总收益: -21.95% 夏普比率: -0.10 最大回撤: -1.14%
    # window:800  entry_z=2.0 exit_z=0.5 策略总收益: -0.44% 夏普比率: -0.65 最大回撤: -0.31%
    # window:900  entry_z=2.0 exit_z=0.5 策略总收益: -7.77% 夏普比率: -0.76 最大回撤: -1.23%
    # window:1000 entry_z=2.0 exit_z=0.5 策略总收益: -12.86% 夏普比率: -0.65 最大回撤: -1.32%
    # window:1200 entry_z=2.0 exit_z=0.5 策略总收益: -20.89% 夏普比率: -0.15 最大回撤: -1.37%
    # window:1400 entry_z=2.0 exit_z=0.5 策略总收益: -1.45% 夏普比率: nan 最大回撤: 0.00%
    # window:1600 entry_z=2.0 exit_z=0.5 策略总收益: -28.18% 夏普比率: nan 最大回撤: 0.00%

