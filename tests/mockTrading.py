import sys
sys.path.append('../src')
from MarginService import MarginManager
from TradingSignalService import SymbolInfo,TickData, PairsTradeSignalService, XSymbolInfo, BTCSymbolInfo

class Position:
    def __init__(self, symbol, size, direction, price, used_margin, leverage = 1):
        self.symbol = symbol      # 标的名称
        self.size = size          # 持仓数量（可正可负）
        self.direction = direction  # 持仓方向（'long'/'short'）
        self.price = price        # 开仓均价
        self.margin = used_margin # 已用保证金
        self.leverage = leverage  # 杠杆倍数


class MockTrading:
    def __init__(self, balance, leverage):
        self.init_balance = balance
        self.balance = balance
        self.leverage = leverage
        self.position = []

        self.avaliable_margin = balance
        self.used_margin = 0
        self.max_redraw = 0
        self.max_redraw_rate = 0
        self.closeCount = 0
        self.winCount = 0
        self.openCount = 0
        self.margin_manager = MarginManager(balance)

    def _closeOredr(self,market_price):
        if len(self.position) == 0:
            return
        print("平仓，market_price:",market_price,"diff:",self.avaliable_margin + self.used_margin - self.balance)
        self.margin_manager.update_price(XSymbolInfo.SYMBOL_NAME,market_price[XSymbolInfo.SYMBOL_NAME],market_price[XSymbolInfo.SYMBOL_NAME])
        self.margin_manager.update_price(BTCSymbolInfo.SYMBOL_NAME,market_price[BTCSymbolInfo.SYMBOL_NAME],market_price[BTCSymbolInfo.SYMBOL_NAME])
        all_margin = 0
        # for pos in self.position:
        #     print(f"平仓:{pos.symbol} 数量:{pos.size} 价格:{market_price[pos.symbol]}")
        #     margin = self.calculate_remaining_margin_after_closing(pos,market_price[pos.symbol])
        #     all_margin += margin
        #     if margin < 0:
        #         self.max_redraw = max(self.max_redraw,abs(margin))
        #         self.max_redraw_rate = self.max_redraw / self.avaliable_margin * 100
        #         print(f"position:{pos.symbol},size:{pos.size},open_price:{pos.price},used_margin:{pos.margin} 平仓后亏损:{margin},market_price:{market_price}")
        #         self.stopRun()
        #     self.avaliable_margin += margin
        #     if self.avaliable_margin < 0:
        #         print(f"平仓后保证金不足:{self.avaliable_margin}")
        #         self.stopRun()
        self.position = []
        self.used_margin = 0
        res = self.margin_manager.close_all_positions()
        profit = res - self.balance
        if profit < 0:
            print(f"平仓后亏损:{profit}")
        else:
            print(f"平仓后盈利:{profit})")
            self.winCount += 1
        self.closeCount += 1
        self.balance = res

    def _createOrder(self,eth_size,btc_size,eth_price,btc_price):
        # if self.avaliable_margin < self.balance / 2.0:
        #     return
        self.openCount += 1
        # print(f"开仓:ETH:{eth_size} BTC:{btc_size} ETH价格:{eth_price} BTC价格:{btc_price}")
        self.margin_manager.update_price(XSymbolInfo.SYMBOL_NAME,eth_price,eth_price)
        self.margin_manager.update_price(BTCSymbolInfo.SYMBOL_NAME,btc_price,btc_price)

        eth_required_margin = self.margin_manager.calculate_required_margin(XSymbolInfo.SYMBOL_NAME, eth_size, 1 if eth_size > 0 else -1, XSymbolInfo.SYMBOL_LEVERAGE)
        self.position.append(Position(XSymbolInfo.SYMBOL_NAME,eth_size,1 if eth_size > 0 else -1,eth_price,eth_required_margin, XSymbolInfo.SYMBOL_LEVERAGE))
        self.used_margin += eth_required_margin
        self.avaliable_margin -= eth_required_margin

        btc_required_margin = self.margin_manager.calculate_required_margin(BTCSymbolInfo.SYMBOL_NAME, btc_size, 1 if btc_size > 0 else -1, BTCSymbolInfo.SYMBOL_LEVERAGE)
        self.position.append(Position(BTCSymbolInfo.SYMBOL_NAME,btc_size,1 if btc_size > 0 else -1,btc_price,btc_required_margin, BTCSymbolInfo.SYMBOL_LEVERAGE))
        self.used_margin += btc_required_margin
        self.avaliable_margin -= btc_required_margin

        self.margin_manager.add_position(XSymbolInfo.SYMBOL_NAME,eth_size,1 if eth_size > 0 else -1,eth_price,XSymbolInfo.SYMBOL_LEVERAGE)
        self.margin_manager.add_position(BTCSymbolInfo.SYMBOL_NAME,btc_size,1 if btc_size > 0 else -1,btc_price,BTCSymbolInfo.SYMBOL_LEVERAGE)
        # self.symbol = symbol
        # self.size = size
        # self.direction = direction
        # self.price = price
        # self.margin = used_margin
    
    def _generate_report(self):
        self.margin_manager.close_all_positions()
        print("生成报告")
        profit = self.margin_manager.total_balance - self.init_balance #self.avaliable_margin - self.balance
        print(f"总盈亏:{profit}")
        # 盈利情况
        percent = profit / self.balance * 100
        print(f"盈利率:{percent}%")
        # 最大回撤
        print(f"最大回撤:{self.max_redraw},回撤率:{self.max_redraw_rate}%")
        # 胜率
        print(f"胜率:{self.winCount / self.closeCount * 100}%")
        print(f"建仓次数:{self.openCount}")
        print(f"平仓次数:{self.closeCount}")
        print(f"胜利次数:{self.winCount}")

        for pos in self.position:
            print(f"未平仓:{pos.symbol} 数量:{pos.size} 价格:{pos.price}")
    
    def stopRun(self):
        print("停止运行")
        # self._generate_report()
        # exit(0)

class MockTradingSingle:
    shared = None


