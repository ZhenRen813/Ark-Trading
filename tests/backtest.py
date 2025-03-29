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
from TradingSignalService import SymbolInfo,TickData,PairsTradeSignalService
from MarginService import Position,MarginManager

def log(msg):
    print(msg)
    # write to log file
    with open('tradinglogP0.log', 'a') as f:
        f.write(f"{datetime.datetime.now()}: {msg}\n")

def load_trendbars_data(symbol = "ETH"):
    # 读取{symbol}_trendbars.csv文件
    filename = f"{symbol}_trendbars.csv"
    data = pd.read_csv(filename)
    print(f"读取 {len(data)} 条记录 <- ETH_trendbars.csv")
    return data


market_prices = {'ETHUSD':0,'BTCUSD':0}

def closeOrder():
    # print("平仓")
    MockTradingSingle.shared._closeOredr(market_prices)

def createOrder(eth_size,btc_size,eth_price,btc_price):
    # print(f"开仓:ETH:{eth_size} BTC:{btc_size} ETH价格:{eth_price} BTC价格:{btc_price}")
    MockTradingSingle.shared._createOrder(eth_size,btc_size,eth_price,btc_price)

# def closeOrder():
#     print("平仓")

# def createOrder(eth_size,btc_size,eth_price,btc_price):
#     print(f"开仓:ETH:{eth_size} BTC:{btc_size} ETH价格:{eth_price} BTC价格:{btc_price}")


class PairsTradingBacktester:

    createOrder = 0
    closeOrder = 0

    def __init__(self,entry_z=1.5, exit_z=0.5, window = 180, initial_cash=1500, closeOrderExcute = None ,createOrderExcute = None):
        """
        参数:
        btc_data, eth_data - 包含时间戳和收盘价的数据框
        initial_window - 初始训练窗口大小
        initial_cash - 初始资金
        """
        
        self.current_data = None
        self.hedge_ratio = None
        self.zscore_params = {}
        
        # 投资组合状态
        self.portfolio = {
            'cash': initial_cash,
            'BTC_position': 0,
            'ETH_position': 0,
            'total_value': initial_cash,
            'history': []
        }
        self.trade_log = []
        self.entry_z = entry_z  # 保存参数
        self.exit_z = exit_z
        self.closeOrderExcute = closeOrderExcute
        self.createOrderExcute = createOrderExcute
        self.btc_price = None
        self.eth_price = None
        self.window = window
        self.alreadyOpen = False
        self.used_margin = 0
        self.leverage = 100
        self.positions = []
        self.balance = initial_cash
        self.report = None
        self.service = PairsTradeSignalService(data=None,symbols={'BTCUSD':SymbolInfo('BTCUSD','BTCUSD',25),'ETHUSD':SymbolInfo('ETHUSD','ETHUSD',3)},window=window,settings={'entry_z':self.entry_z,'exit_z':self.exit_z})
        self.margin_manager = MarginManager(total_balance=initial_cash)

    def run_backtest_2(self,df, window=1500):
        # self.service.data = df.rename(columns={'BTC':'BTCUSD','ETH':'ETHUSD'})
        self._backtest_with_df(df, window)

    def run_backtest(self,btc_data, eth_data, window=1500):

        # 合并数据
        self.data = pd.DataFrame({
            'BTC': btc_data,
            'ETH': eth_data
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
                # print("初始化数据")
                self.current_data = self.data.iloc[start_idx:i+1]
                continue
            else:
                if i+1 < len(self.data):
                    btc_price = self.data['BTC'].iloc[i+1]
                    etc_price = self.data['ETH'].iloc[i+1]
                    self._update_data(self.data['BTC'].iloc[i+1], self.data['ETH'].iloc[i+1])
                    self._checkUpdateSignal(reset_model=True,btc_price=btc_price,eth_price=etc_price)
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

    def _reload_data(self, btc_data, eth_data):
        print("重新加载数据")
        """重新加载数据"""
        self.data = pd.DataFrame({
            'BTC': btc_data['close'],
            'ETH': eth_data['close']
        }).dropna()
        self.current_data = self.data
        print(f"重新加载数据:BTC:{btc_data['close']},ETH:{eth_data['close']},datalen:{len(self.current_data)}")
        self.service.data = pd.DataFrame({
            'BTCUSD': btc_data['close'],
            'ETHUSD': eth_data['close']
        }).dropna()
        self.service.update_data(None)

    def _update_used_margin(self, margin):
        self.used_margin = margin
        # self.portfolio['cash'] = self.balance - self.used_margin

    def _update_data(self, btc_price, eth_price):
        """更新数据"""
        # print(f"更新数据:BTC:{btc_price},ETH:{eth_price},datalen:{len(self.current_data)}")
        if len(self.current_data) >= self.window:
            # self.current_data = self.current_data.drop(index=self.current_data.index[0])
            self.current_data = self.current_data.iloc[1:].reset_index(drop=True)
        self.current_data.loc[len(self.current_data)] = {
            'BTC': btc_price,
            'ETH': eth_price,
        }
        self.data.loc[len(self.data)] = {
            'BTC': btc_price,
            'ETH': eth_price,
        }
        self.btc_price = btc_price
        self.eth_price = eth_price
        market_prices['ETHUSD'] = self.eth_price
        market_prices['BTCUSD'] = self.btc_price

        self.service.update_data({'BTCUSD':TickData('BTCUSD',datetime.datetime.now().timestamp(),btc_price),'ETHUSD':TickData('ETHUSD',datetime.datetime.now().timestamp(),eth_price)})
    
    def _write_all_data_to_file(self):
        self.data.to_csv('algo_allData.csv')

    def _checkUpdateSignal(self,reset_model = False,btc_price = None, eth_price = None):
        self.margin_manager.update_price('BTCUSD', btc_price, btc_price)
        self.margin_manager.update_price('ETHUSD', eth_price, eth_price)
        # TODO: check why 'cash' < 0
        if len(self.current_data) < 30 or self.margin_manager.get_available_margin() <= 0 or btc_price is None or eth_price is None:  # 最小数据量要求
            print(f"数据不足,当前数据量:{len(self.current_data)},cash:{self.portfolio['cash']},btc_price:{btc_price},eth_price:{eth_price}")
            return None

        # print(f">>>>>>>>>>>>>>>>>>>>>>>>>cash:{self.portfolio['cash']},balance:{self.balance},used_margin:{self.used_margin},hedge_ratio:{self.hedge_ratio},zsocre_params:{self.zscore_params}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

        # 生成交易信号
        signal = self._generate_signal(btc_price, eth_price)
        
        self._execute_trade(signal,btc_price, eth_price)
        
        # 更新投资组合价值
        self._update_portfolio(btc_price, eth_price)
        
        # 记录每日状态
        self._record_state(btc_price, eth_price)

        return signal
    
    def smooth_spread(self,spread_series, alpha=0.1):
        """指数加权平滑价差序列"""
        ewma = pd.Series(spread_series).ewm(alpha=alpha).mean()
        return ewma

    def _generate_signal(self,btc_price = None, eth_price = None):
        time = datetime.datetime.now().timestamp()
        ticks:dict[str,TickData] = {
            'BTCUSD':TickData('BTCUSD',time,btc_price),
            'ETHUSD':TickData('ETHUSD',time,eth_price)
        }
        return self.service.generate_signal(ticks,self.service.hedge_ratio,self.service.zscore_params)
    
    def _last_Date(self):
        return self.current_data.index[-1]

    def _execute_trade(self, signal,_btc_price = None, _eth_price = None):
        """执行交易（增加空值保护）"""
        # 检查价格有效性
        try:
            btc_price = float(_btc_price)
            eth_price = float(_eth_price)
        except (TypeError, ValueError):
            return
        
        # 跳过无效价格
        if np.isnan(_btc_price) or np.isnan(_eth_price):
            return
        """执行交易"""
        btc_price = _btc_price
        eth_price = _eth_price
        self.margin_manager.update_price('BTCUSD', btc_price, btc_price)
        self.margin_manager.update_price('ETHUSD', eth_price, eth_price)
        
        # 平仓逻辑
        if signal == 0 and (self.portfolio['BTC_position'] != 0 or self.portfolio['ETH_position'] != 0):
            self._close_positions(btc_price, eth_price)
        
        # 开仓逻辑
        if signal in [-1, 1]:
            if self.alreadyOpen:
                return
            if self.margin_manager.get_available_margin() < 700:
                print(f"保证金不足==================================,used_margin:{self.used_margin},balance:{self.balance}")
                return
            # 计算头寸规模（保持美元中性）
            position_value = 700#self.portfolio['cash'] / 2 #* 36
            print(f"position_value:{position_value}")
            # 仓位精度为两个小数点，价格精度为三个小数点
            # btc_size = position_value / btc_price
            # eth_size = position_value / eth_price
            btc_size = 0.01
            eth_size = btc_size / self.service.hedge_ratio #self.hedge_ratio
            es_margin1 = self.margin_manager.calculate_required_margin('BTCUSD', btc_size, 1 if signal == 1 else -1, 25)
            es_margin2 = self.margin_manager.calculate_required_margin('ETHUSD', eth_size, 1 if signal == 1 else -1, 3)
            times = round(position_value / (es_margin1 + es_margin2),0)
            btc_size = btc_size * times
            eth_size = eth_size * times
            # print("对冲比率",self.service.hedge_ratio,">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",btc_size,eth_size)
            # times = 0

            # while (abs(btc_size) * abs(btc_price)) + (abs(eth_size) * abs(eth_price)) * times < position_value:
            #     times += 1
            # if times == 0:
            #     print("不够钱==================================",btc_price,eth_price)
            #     return
            # btc_size = btc_size * times
            # eth_size = eth_size * times

            if btc_size < 0.01 or eth_size < 0.01:
                log(f"头寸过小{btc_size},{eth_size},取消交易,position_value:{position_value},time:{datetime.datetime.now()},btc_price:{btc_price},eth_price:{eth_price}")
                return
            
            btc_size = round(btc_size, 2)
            eth_size = round(eth_size, 2)

            self.createOrder += 1

            margin1 = self.margin_manager.calculate_required_margin('BTCUSD', btc_size, 1 if signal == 1 else -1, 25)
            margin2 = self.margin_manager.calculate_required_margin('ETHUSD', eth_size, 1 if signal == 1 else -1, 3)
            # if margin1 + margin2 + self.used_margin > (self.margin_manager.get_available_margin()):
            #     print(f"保证金不足==================================,margin1:{margin1},margin2:{margin2}used_margin:{self.used_margin},balance:{self.balance}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            #     return
            
            if signal == 1:  # 做多价差
                self._open_position(eth_size, -btc_size, eth_price, btc_price)
            else:  # 做空价差
                self._open_position(-eth_size, btc_size, eth_price, btc_price)

            
            self._update_used_margin(margin1 + margin2)
            self.alreadyOpen = True

    def _open_position(self, eth_size, btc_size, eth_price, btc_price):
        """开仓操作"""
        # ETH交易
        eth_cost = eth_size * eth_price
        self.portfolio['cash'] -= eth_cost
        # self.portfolio['cash'] -= MockTradingSingle.shared.calculate_reqired_margin(1, eth_size, eth_price)
        self.portfolio['ETH_position'] += eth_size
        
        # BTC交易
        btc_cost = btc_size * btc_price
        self.portfolio['cash'] -= btc_cost
        # self.portfolio['cash'] -= MockTradingSingle.shared.calculate_reqired_margin(1, btc_size, btc_price)
        self.portfolio['BTC_position'] += btc_size

        self.margin_manager.add_position('ETHUSD', abs(eth_size), 1 if eth_size > 0 else -1, eth_price, 3)
        self.margin_manager.add_position('BTCUSD', abs(btc_size), 1 if btc_size > 0 else -1, btc_price, 25)
        
        self.createOrderExcute(eth_size,btc_size,eth_price,btc_price)

        # 记录交易日志到本地
        if eth_size != 0 and btc_size != 0:
            log(f"time:{datetime.datetime.now} OPEN: ETH_Size:{eth_size} BTC_Size:{btc_size} ETH价格:{eth_price} BTC价格:{btc_price}")

        self.trade_log.append({
            'date': self._last_Date(),
            'action': 'OPEN',
            'ETH_size': eth_size,
            'BTC_size': btc_size,
            'ETH_price': eth_price,
            'BTC_price': btc_price
        })

    def _close_positions(self, btc_price, eth_price):
        if btc_price is None or eth_price is None:
            log(f"价格为空,取消平仓")
            return
        """平仓操作"""
        # 平仓ETH
        eth_size = self.portfolio['ETH_position']
        eth_value = eth_size * eth_price
        # self.portfolio['cash'] += eth_value
        # self.portfolio['cash'] += MockTradingSingle.shared.calculate_released_margin(self.leverage, self.portfolio['ETH_position'], eth_price)
        self.portfolio['ETH_position'] = 0
        
        # 平仓BTC
        btc_size = self.portfolio['BTC_position']
        btc_value = btc_size * btc_price
        # self.portfolio['cash'] += btc_value
        self.portfolio['BTC_position'] = 0
        
        self.trade_log.append({
            'date': self._last_Date(),
            'action': 'CLOSE',
            'ETH_price': eth_price,
            'BTC_price': btc_price
        })

        # 记录交易日志到本地
        if eth_value != 0 and btc_value != 0:
            log(f"{datetime.datetime.now()}: CLOSE ETH_Size:{eth_size} BTC_Size:{btc_size} ETH价格:{eth_price} BTC价格:{btc_price}\n")

        self.closeOrder += 1
        self.closeOrderExcute()
        self.margin_manager.close_all_positions()
        self._update_used_margin(0)
        self.portfolio['cash'] = self.margin_manager.get_available_margin()
        if self.margin_manager.get_available_margin() < 0:
            print(f"保证金不足==================================,used_margin:{self.used_margin},balance:{self.balance}")
            print(self.margin_manager.get_all_current_margin())
            exit(0)
        self.alreadyOpen = False

    def _update_portfolio(self,btc_price = None,eth_price = None):
        """更新投资组合价值"""
        btc_value = self.portfolio['BTC_position'] * btc_price
        eth_value = self.portfolio['ETH_position'] * eth_price
        self.portfolio['total_value'] = self.portfolio['cash']+ btc_value + eth_value
        # print(f"更新投资组合价值: {self.portfolio['total_value']:.2f}")
        if btc_value != 0 or eth_value != 0:
            print(f"更新投资组合价值: {self.portfolio['total_value']:.2f},cash:{self.portfolio['cash']:.2f},btc_value:{btc_value:.2f},eth_value:{eth_value:.2f}")

    def _record_state(self,btc_price = None,eth_price = None):
        """记录每日状态"""
        self.portfolio['history'].append({
            'date': self._last_Date(),
            'total_value': self.portfolio['total_value'],
            'BTC_price': btc_price,
            'ETH_price': eth_price,
            'BTC_position': self.portfolio['BTC_position'],
            'ETH_position': self.portfolio['ETH_position']
        })
        # print(f"记录每日状态: {self._last_Date()},total_value:{self.portfolio['total_value']:.2f},cash:{self.portfolio['cash']:.2f},BTC_price:{btc_price:.2f},ETH_price:{eth_price:.2f},BTC_position:{self.portfolio['BTC_position']},ETH_position:{self.portfolio['ETH_position']}")

    def _generate_report(self):

        if self.createOrder == 0 or self.closeOrder == 0:
            print("无交易记录")
            return

        """生成分析报告"""
        history_df = pd.DataFrame(self.portfolio['history'])
        trade_df = pd.DataFrame(self.trade_log)
        
        # 计算收益
        print("计算收益")
        history_df['returns'] = history_df['total_value'].pct_change()
        history_df['cum_returns'] = (1 + history_df['returns']).cumprod()
        
        # 价差和仓位
        print("价差和仓位")
        # 移除掉history_df的ETH_price和BTC_price的空值
        history_df = history_df.dropna(subset=['ETH_price', 'BTC_price'])
        # 裁剪数据将history_df的ETH_price和BTC_price长度对齐
        history_df = history_df.iloc[-len(self.current_data):]

        # 计算关键指标
        print("计算关键指标")
        total_return = history_df['cum_returns'].iloc[-1] - 1
        sharpe = self._calculate_sharpe(history_df['returns'])
        max_drawdown = self._calculate_max_drawdown(history_df['total_value'])
        
        print(f"策略总收益: {total_return:.2%}")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"总交易次数: {len(trade_df)}")
        self.report = {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(trade_df)
        }

        # 将 self.service.zsocres 绘制成折线图
        print("绘制折线图",len(self.service.zsocres))
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
    all_data = pd.read_csv('data0329.csv')
    all_data[['BTC']] = all_data[['BTC']] / 100000
    all_data[['ETH']] = all_data[['ETH']] / 100000

    # use first 1000 records
# 策略总收益: 0.32%
# 夏普比率: 2.26
# 最大回撤: 0.00%
# 总交易次数: 18
# 建仓次数:12
# 平仓次数:6
    all_data = all_data.tail(1000)

    entry_z_arr = [1.7,1.8,1.9,2.0]
    windows = [160,170,180,190,200,300]

    for entry_z in entry_z_arr:
        for window in windows:
            # MockTradingSingle.shared = MockTrading(balance=1000000,leverage=100)
            print(f"window:{window}  entry_z={entry_z}")
            # backtester = PairsTradingBacktester(entry_z=entry_z, exit_z=0.5,window=window,closeOrderExcute=closeOrder,createOrderExcute=createOrder)
            # backtester.run_backtest_2(all_data, window=window)
            # print(f"建仓次数:{backtester.createOrder}")
            # print(f"平仓次数:{backtester.closeOrder}")
            # MockTradingSingle.shared._generate_report()
            # print("=========================================================")
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
    MockTradingSingle.shared = MockTrading(balance=1500,leverage=100)
    backtester = PairsTradingBacktester(entry_z=1.5, exit_z=0.8,closeOrderExcute=closeOrder,createOrderExcute=createOrder)
    backtester.run_backtest_2(all_data, window=180)

    # 加载数据（示例数据）
    # btc_prices = load_trendbars_data(symbol="BTC")
    # eth_prices = load_trendbars_data(symbol="ETH")
    # df1_aligned,df2_aligned = _cleanData_(btc_prices,eth_prices)
    # backtester.run_backtest(df1_aligned, df2_aligned, window=1500)
    
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

