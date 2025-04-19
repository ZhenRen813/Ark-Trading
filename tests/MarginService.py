class Position:
    def __init__(self, symbol, size, direction, price, used_margin, leverage):
        self.symbol = symbol      # 标的名称
        self.size = size          # 持仓数量（可正可负）
        self.direction = direction  # 持仓方向（'多':1/'空':-1）
        self.price = price        # 开仓均价
        self.margin = used_margin # 已用保证金
        self.leverage = leverage  # 杠杆倍数

class MarginManager:
    def __init__(self, total_balance):
        self.total_balance = total_balance  # 总资金
        self.positions = []       # 当前持仓 {symbol: Position}
        self.symbol_prices = {}   # 标的最新价格 {symbol: (bid, ask)}

    # ‌Bid‌（买价）和‌Ask‌（卖价）
    def update_price(self, symbol, bid, ask):
        """更新标的的最新买卖价格"""
        self.symbol_prices[symbol] = (bid, ask)

    def calculate_required_margin(self, symbol, volume, direction, leverage):
        """计算开仓所需保证金"""
        if symbol not in self.symbol_prices:
            raise ValueError(f"Price for {symbol} not available: {self.symbol_prices}")
        
        # 根据方向选择价格
        bid, ask = self.symbol_prices[symbol]
        price = bid if direction == 1 else ask
        return abs(volume) * price / leverage

    def add_position(self, symbol, size, direction, price, leverage):
        """添加/合并仓位（自动计算保证金）"""
        # 计算新仓位的保证金
        new_margin = abs(size) * price / leverage
        
        self.positions.append(Position(
            symbol, abs(size), direction, price, new_margin, leverage
        ))

    def _get_current_margin(self, position):
        """动态计算单个仓位的当前保证金"""
        if position.symbol not in self.symbol_prices:
            return position.margin  # 价格未更新时返回原值
        
        bid, ask = self.symbol_prices[position.symbol]
        current_price = bid if position.direction == 1 else ask
        print("get current profit = ",current_price - position.price)
        return (abs(position.size) * current_price) / position.leverage

    # TODO: 相反方向的仓位保证金应该合并，同样数量合并后按最大的来算
    def get_used_margin(self):
        """获取当前总占用保证金（动态计算）"""
        return sum(pos.margin for pos in self.positions)
    
    def get_all_current_margin(self):
        """获取当前总保证金（动态计算）"""
        return sum(self._get_current_margin(pos) for pos in self.positions)

    def get_available_margin(self):
        """获取实时可用保证金"""
        return self.total_balance - self.get_all_current_margin()
    
    def close_all_positions(self):
        """清空所有仓位"""
        profit = 0
        releaded_margin = 0
        copy = self.positions.copy()
        for pos in copy:
            res = self.close_position(pos, pos.size)
            profit += res["realized_pnl"]
            print(profit)
        del copy
        return self.total_balance

    def close_position(self, position, size):
        """平仓操作（支持部分平仓）"""
        
        if size <= 0 or size > position.size:
            raise ValueError(f"Invalid close size: {size}")

        # 获取最新价格
        bid, ask = self.symbol_prices.get(position.symbol, (0, 0))
        if bid == 0 or ask == 0:
            raise ValueError(f"Missing price for {position.symbol}")

        # 计算平仓价格和盈亏
        if position.direction == 1:
            close_price = ask  # 平多用卖价
            profit = size * (close_price - position.price)
        else:
            close_price = bid  # 平空用买价
            profit = size * (position.price - close_price)

        # 更新账户余额
        self.total_balance += profit
        print("balance",self.total_balance,"profit:",profit)

        # 计算释放的保证金
        released_ratio = size / position.size
        released_margin = position.margin * released_ratio

        # 更新仓位状态
        position.size -= size
        position.margin -= released_margin

        # 删除空仓位
        if position.size == 0:
            del self.positions[self.positions.index(position)]

        return {
            "realized_pnl": profit,
            "released_margin": released_margin,
            "remaining_size": position.size if position.size > 0 else 0
        }
    



def test():
    # ‌Bid‌（买价）和‌Ask‌（卖价）
    m_m = MarginManager(1000)
    Y_price = {'bid':84437.960,'ask':84417.540}
    m_m.update_price('YUSD', bid=Y_price['bid'], ask=Y_price['ask'])
    # m_m.update_price('XUSD', bid=300, ask=301)
    m_m.add_position('YUSD', 0.01, 1, Y_price["bid"], 25)
    m_m.add_position('YUSD', 0.01,-1, Y_price["ask"], 25)
    # m_m.add_position('XUSD', 1, -1, 301, 3)
    print(m_m.get_used_margin())
    print(m_m.get_available_margin())
    m_m.update_price('YUSD', bid=84670.360, ask=84649.940)
    # print(m_m.get_available_margin())
    # # m_m.update_price('XUSD', 301, 302)
    # print(m_m.get_used_margin())
    print('availiable margin',m_m.get_available_margin())
    m_m.close_all_positions()
    print(m_m.total_balance)

test()