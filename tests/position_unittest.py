import unittest
from trade_position import PairsTradingPosition

class TestPairsTradingPosition(unittest.TestCase):
    def setUp(self):
        self.pt = PairsTradingPosition(
            available_margin=1000,
            total_capital=1000,
            risk_ratio=0.1,
            leverage=1000,
            min_trade_size=0.01
        )

    
    def test_long_signal_normal(self):
        """Z-score为负时做多价差（买X卖Y）"""
        signal = {
            'zscore': -2.5,  # 触发做多
            'spread': 10,
            'available_margin': 100,
            'leverage': 1000,
            'hedge_ratio': 1.0
        }
        result = self.pt.calculate_position_size(signal,available_margin=100)
        # 预期：X多头，Y空头
        self.assertGreater(result['X'], 0)  
        self.assertLess(result['Y'], 0)
    
    def test_min_trade_size(self):
        """极小金额时强制满足最小手数"""
        signal = {
            'zscore': 3.0,  # 触发做空
            'spread': 0.01,
            'available_margin': 0.0001,
            'leverage': 1000,
            'hedge_ratio': 1.0
        }
        result = self.pt.calculate_position_size(signal,available_margin=0.0001)
        # 验证方向正确且手数达标
        self.assertAlmostEqual(abs(result['X']), 0.01, delta=1e-6)
        self.assertEqual(np.sign(result['X']), -1)  # 做空价差，X应为负
    
    def test_negative_hedge_ratio(self):
        """对冲比率为负时方向反转"""
        signal = {
            'zscore': 2.0,
            'spread': 10,
            'available_margin': 1,
            'leverage': 1000,
            'hedge_ratio': -2.0  # 对冲比率符号影响Y方向
        }
        result = self.pt.calculate_position_size(signal,available_margin=1)
        # 预期：X = -2*units, Y = +units
        self.assertLess(result['X'], 0)  # 对冲比率为负，X方向反转
        self.assertGreater(result['Y'], 0)  # Y方向与X相反

    
    # # 测试场景1：正常做多信号（可用保证金充足）
    # def test_long_signal_normal(self):
    #     signal = {
    #         'signal': 'long',
    #         'spread': 2,
    #         'zscore': 2.0,
    #         'hedge_ratio': 0.5,
    #         'available_margin': 1000,
    #         'leverage': 1000
    #     }

    #     result = self.position.calculate_position_size(signal, available_margin=1000)
    #     # 验证对冲比率应用‌:ml-citation{ref="4" data="citationList"}
    #     self.assertAlmostEqual(result['X'], -50_000)
    #     self.assertAlmostEqual(result['Y'], 100_000)
    #     self.assertTrue(abs(result['Y']) <= 100_000)  # 100,000 <= 100,000

    # # 测试场景2：保证金不足时返回0手数
    # def test_insufficient_margin(self):
    #     signal = {
    #         'signal': 'long',
    #         'spread': 100,
    #         'zscore': 2.0,
    #         'hedge_ratio': 0.5,
    #         'available_margin': 0  # 保证金不足
    #     }
    #     result = self.position.calculate_position_size(signal,available_margin=0)
    #     self.assertEqual(result['X'], 0)
    #     self.assertEqual(result['Y'], 0)  # ‌:ml-citation{ref="2" data="citationList"}

    # # 测试场景3：最小手数限制验证
    # def test_min_trade_size(self):
    #     signal = {
    #         'signal': 'short',
    #         'spread': 1000,
    #         'zscore': 2,  # 高波动率场景
    #         'hedge_ratio': 0.5,
    #         'available_margin': 10
    #     }
    #     result = self.position.calculate_position_size(signal, available_margin=10)
    #     self.assertAlmostEqual(abs(result['Y']), 0.01, delta=1e-8)


    # # 测试场景4：对冲比率为负数的处理
    # def test_negative_hedge_ratio(self):
    #     signal = {
    #         'signal': 'long',
    #         'spread': 100,
    #         'zscore': 2.0,
    #         'hedge_ratio': -0.5,  # 负对冲比率
    #         'available_margin': 1000
    #     }
    #     result = self.position.calculate_position_size(signal,available_margin=1000)
    #     self.assertAlmostEqual(result['X'], 1000)  # 负号反转方向‌:ml-citation{ref="4" data="citationList"}

    # # 测试场景5：平仓信号验证
    # def test_close_signal(self):
    #     # 先建立持仓
    #     self.position.current_position = {'X': 100, 'Y': -200}
    #     signal = {
    #         'signal': 'close',
    #         'spread': 0,
    #         'zscore': 2.0,
    #         'hedge_ratio': 0,
    #         'available_margin': 1000
    #     }
    #     result = self.position.calculate_position_size(signal, available_margin=1000)
    #     self.assertEqual(result['X'], -100)  # 完全平仓‌:ml-citation{ref="2" data="citationList"}
    #     self.assertEqual(result['Y'], 200)

if __name__ == '__main__':
    unittest.main()
