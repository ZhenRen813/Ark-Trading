import real_time_algo as rt


class TradeDirection:
    LONG = 1
    SHORT = -1

class TradePositionSignal:
    def __init__(self, symbol, lots, direction):
        self.symbol = symbol
        self.lots = lots
        self.direction = direction



class TradingAlgorithm:
    def __init__(self, init_margin, leverage, min_lots, risk_exposure=0.3, stop_loss=0.3, algo_settings=None):
        """
        初始化配对交易算法类
        :param init_margin: 初始保证金
        :param leverage: 杠杆倍数
        :param min_lots: 最小交易手数
        :param risk_exposure: 风险敞口比例
        :param stop_loss: 强制平仓线
        :param algo_settings: 算法参数, 用于传递算法特有参数
        """
        # 基本参数初始化
        self.available_margin = init_margin
        self.leverage = leverage
        self.min_lots = min_lots
        self.stop_loss = stop_loss
        self.hedge_ratio = None
        self.current_position = {'X': 0, 'Y': 0}
        self.position_strategy = 'fixed'  # 默认固定头寸策略
        self.risk_exposure = risk_exposure
        self.algo_settings = algo_settings
        
    # :param datas: 所有标的dataFrame 【[date, X, Y, ...], ...] ,X, Y为标的价格
    def load_data(self, datas):
        print('load_data')
        # 数据处理
        self.datas = datas

    # :param balance: 最新保证金
    def update_balance(self, balance):
        self.available_margin = balance
    
    # :param prices 为所有标的的最新价格，格式为[[date, X, Y, ...], ...]
    def check_trade_signal(self, prices):
        print('check_trade_signal')

    # :param signal: 开仓信号，包括标的，手数，方向
    def open_position(self, signals: [TradePositionSignal]):
        print('open_position')

    # :param signal: 平仓信号
    def close_position(self, signal):
        print('close_position')
    
    def generate_report(self):
        print('generate_report')


    