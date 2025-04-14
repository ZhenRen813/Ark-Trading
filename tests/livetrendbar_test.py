import asyncio
import time
import datetime
from typing import Optional, TypeVar
import calendar
import sys
sys.path.append('..')
import pandas as pd
from backtest import PairsTradingBacktester
from TradingSignalService import SymbolInfo, PairsTradeSignalService, ETHSymbolInfo, BTCSymbolInfo


import numpy as np
from google.protobuf.message import Message
from waterstart.client import OpenApiClient
from waterstart.openapi import (
    M1,
    M30,
    H1,
    D1,
    H4,
    H12,
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOAErrorRes,
    ProtoOASpotEvent,
    ProtoOAGetTrendbarsReq,
    ProtoOAGetTrendbarsRes,
    ProtoOASubscribeLiveTrendbarReq,
    ProtoOASubscribeLiveTrendbarRes,
    ProtoOASubscribeSpotsReq,
    ProtoOASubscribeSpotsRes,
    ProtoOASymbolByIdReq,
    ProtoOASymbolByIdRes,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
    ProtoOAMarginChangedEvent,
    ProtoOAMarginCall,
    ProtoOAMarginCallUpdateReq,
    ProtoOAMarginCallUpdateRes,
    ProtoOAMarginCallUpdateEvent,
    ProtoOAExpectedMarginReq,
    ProtoOAExpectedMarginRes,
    ProtoOANewOrderReq,
    ProtoOAOrderType,
    ProtoOATradeSide,
    ProtoOATraderReq,
    ProtoOATraderRes,
    ProtoOAClosePositionReq,
    ProtoOAReconcileReq,
    ProtoOAReconcileRes,
)

class EndPoints:
    AUTH_URI = "https://openapi.ctrader.com/apps/auth"
    TOKEN_URI = "https://openapi.ctrader.com/apps/token"
    PROTOBUF_DEMO_HOST = "demo.ctraderapi.com"
    PROTOBUF_LIVE_HOST = "live.ctraderapi.com"
    PROTOBUF_PORT = 5035

class ACCOUNT:
    USD = 42750992
    EUD = 42563188

class AccountInfo:
    ACCOUNT_ID = ACCOUNT.USD
    CLIENT_ID = "13628_lbJ2ix1H7UFg5W8ar2eeoFOL0xJUR88G2BLhdvJqnCWtytEzSn"
    CLIENT_SECRET = "hpGJohkYLBBDuzd1nYkb6YPuZD74hE45yGTu9U0nNwJxwlurQu"
    ACCESS = "Xjfns1lfehTNSh3YYIAEapmO_P9r2FREBFsDhQUdDQ4"

class RunParams:
    Enter_Z = 1.8
    Exit_Z = 0.7
    Window = 180

HOST = EndPoints.PROTOBUF_DEMO_HOST
PORT = EndPoints.PROTOBUF_PORT
ACCOUNT_ID = AccountInfo.ACCOUNT_ID

T = TypeVar("T", bound=Message)

Price_Digits = 100000

# TODO: 排查卡死的问题

def request_trendbars(symbol_id: int, period: int, count = 60 * 24 * 1,day = 1) -> ProtoOAGetTrendbarsReq:
    print("requesting trendbars")
    current_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    from_timestamp = (current_time - 60 * (RunParams.Window)) * 1000
    print(symbol_id, period, from_timestamp)
    print("from")
    print(datetime.datetime.utcfromtimestamp(from_timestamp / 1000))
    # 打印当前from_timestamp时间日期和symbol_id
    # int(calendar.timegm((datetime.datetime.utcnow() - datetime.timedelta(weeks=int(1))).utctimetuple())) * 1000
    # get current timestamp
    exday = day - 7
    to_timestamp = current_time * 1000
    print("to")
    print(datetime.datetime.utcfromtimestamp(to_timestamp / 1000))
    # int(calendar.timegm(datetime.datetime.utcnow().utctimetuple())) * 1000
    return ProtoOAGetTrendbarsReq(
        ctidTraderAccountId=ACCOUNT_ID,
        symbolId=symbol_id,
        period=period,
        fromTimestamp= from_timestamp,
        toTimestamp= to_timestamp,
    )


async def createNewOrderRequest(client: OpenApiClient,account_id: int, symbol_id: int, volume: float, side: ProtoOATradeSide, order_type: ProtoOAOrderType, stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
    print("create new order")
    # print volume and side and order_type and save log to file
    print(f"volume: {volume}, side: {side}, order_type: {order_type}")
    with open('order.log', 'a') as f:
        f.write(f"volume: {volume}, side: {side}, order_type: {order_type}\n")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    new_order_req = ProtoOANewOrderReq(
        ctidTraderAccountId=account_id,
        symbolId=symbol_id,
        volume=int(volume*100),
        tradeSide=side,
        orderType=order_type,
        stopLoss=stop_loss,
        takeProfit=take_profit,
    )
    await client.send_message(new_order_req)

async def closeAllPostion(client: OpenApiClient,account_id: int):
    print(f"close all position.")
    reconcile_req = ProtoOAReconcileReq(ctidTraderAccountId=account_id)
    reconcile_res = await client.send_and_wait_response(
        reconcile_req, ProtoOAReconcileRes
    )
    print(f"position count: {len(reconcile_res.position)}")
    for pos in reconcile_res.position:
        close_req = ProtoOAClosePositionReq(
            ctidTraderAccountId=account_id,
            positionId=pos.positionId,
            volume=pos.tradeData.volume,
        )
        print(f"pre close position: {pos.positionId}, volume: {pos.tradeData.volume}")
        await client.send_message(close_req)
        print(f"close position: {pos.positionId}, volume: {pos.tradeData.volume}")



btcData = None
ethData = None

class TrendbarData:
    btcData: ProtoOAGetTrendbarsRes = None
    ethData: ProtoOAGetTrendbarsRes = None
    processData = None

class RealTimePrice:
    eth: int = 0
    btc: int = 0
    timestampInMin: int = 0

class Holdings:
    def __init__(self):
        self.X = 0
        self.Y = 0

current_refresh_time = datetime.datetime.now()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
trendbarData = TrendbarData()
result_queue = asyncio.Queue()  # 创建异步队列存储结果
realTime = RealTimePrice()
balance = 0

local_data = pd.DataFrame({
            ETHSymbolInfo.SYMBOL_NAME: [],
            BTCSymbolInfo.SYMBOL_NAME: []
        }).dropna()
# trader = rt.PairsTradingRealtime()
# position = tp.PairsTradingPosition(available_margin=1000,total_capital=1000) # TODO: 传入账户总资金和当前仓位

day = 35
# 定义定时任务（每分钟执行）
async def scheduled_task(client,realtime, trendbarDa,result_queue, symbol_id1, symbol_id2,algo):
        global day
        history_trend_req1 = request_trendbars(symbol_id1, M1,day=day)  
        history_trend_res1 = await client.send_and_wait_response(  
            history_trend_req1, ProtoOAGetTrendbarsRes  
        )
        print(f"获取{symbol_id1}数据成功, 共{len(history_trend_res1.trendbar)}条数据")

        history_trend_req2 = request_trendbars(symbol_id2, M1,day=day)  
        history_trend_res2 = await client.send_and_wait_response(  
            history_trend_req2, ProtoOAGetTrendbarsRes  
        )
        print(f"获取{symbol_id2}数据成功, 共{len(history_trend_res2.trendbar)}条数据")
        day -= 7
        trendbarDa.ethData = history_trend_res1
        trendbarDa.btcData = history_trend_res2

        # last timeStamp in minutes
        current_timeStamp = history_trend_res1.trendbar[-1].utcTimestampInMinutes
        if realtime.btc == 0 or realtime.timestampInMin < current_timeStamp:
            realtime.btc = history_trend_res1.trendbar[-1].low + history_trend_res1.trendbar[-1].deltaClose
            realtime.timestampInMin = history_trend_res1.trendbar[-1].utcTimestampInMinutes
        if realtime.eth == 0 or realtime.timestampInMin < current_timeStamp:
            realtime.eth = history_trend_res2.trendbar[-1].low + history_trend_res2.trendbar[-1].deltaClose
            realtime.timestampInMin = history_trend_res2.trendbar[-1].utcTimestampInMinutes
        
        df1,df2 = processData(trendbarData.btcData, trendbarData.ethData)
        dataFrame = pd.DataFrame({
            ETHSymbolInfo.SYMBOL_NAME: df1['close']/Price_Digits,
            BTCSymbolInfo.SYMBOL_NAME: df2['close']/Price_Digits
        }).dropna()
        global local_data
        local_data = pd.concat([local_data,dataFrame])
        local_data.to_csv('data.csv')
        # btc、eth
        algo._reload_data(df2,df1)
        hanleHistoryData(trendbarData.btcData, trendbarData.ethData)
        # await asyncio.sleep(30)

def hanleHistoryData(btc_data, eth_data):
# 数据处理
    df1, df2 = processData(btc_data, eth_data)
    print(len(df1))

    # 读取数据
    historical =  pd.DataFrame({
            ETHSymbolInfo.SYMBOL_NAME: df2['close'],
            BTCSymbolInfo.SYMBOL_NAME: df1['close']
        }).dropna()
    
    # trader.initialize_history(historical)
    print("初始化历史数据完成")

    return historical


def processData(btc_data, eth_data):

    # 将数组对象转换为DataFrame
    btc_prices = pd.DataFrame(columns=['utcTimestampInMinutes', 'close'])
    btc_list = []
    for trendbar in btc_data.trendbar:
        new_row = {'utcTimestampInMinutes':trendbar.utcTimestampInMinutes,'close': trendbar.low + trendbar.deltaClose/Price_Digits}  
        btc_list.append(new_row)
    btc_prices = pd.DataFrame(btc_list)

    eth_prices = pd.DataFrame(columns=['utcTimestampInMinutes', 'close'])
    eth_list = []
    for trendbar in eth_data.trendbar:
        new_row = {'utcTimestampInMinutes':trendbar.utcTimestampInMinutes,'close': trendbar.low + trendbar.deltaClose/Price_Digits}  
        eth_list.append(new_row)
    eth_prices = pd.DataFrame(eth_list)

    print(btc_prices.head())
    print(eth_prices.head())

    df1 = btc_prices
    df2 = eth_prices

    # print(">>>>1 " + str(len(df1)))

    # 确保时间戳列类型正确（假设存储为整数分钟）
    df1['utcTimestampInMinutes'] = df1['utcTimestampInMinutes'].astype(int)
    df2['utcTimestampInMinutes'] = df2['utcTimestampInMinutes'].astype(int)

    # print(">>>>2 " + str(len(df1)))

    # 通过高效数组操作获取时间戳交集
    common_ts = np.intersect1d(
        df1['utcTimestampInMinutes'], 
        df2['utcTimestampInMinutes']
    )

    # print(">>>>3 " + str(len(df1)))

    # 筛选交集数据
    df1_aligned = df1[df1['utcTimestampInMinutes'].isin(common_ts)]
    df2_aligned = df2[df2['utcTimestampInMinutes'].isin(common_ts)]

    # print(">>>>4 " + str(len(df1_aligned)))

    # print(">>>>5 " + str(len(df1_aligned)))

    # 按时间戳升序排序
    df1_sorted = df1_aligned.sort_values('utcTimestampInMinutes', ascending=True)
    df2_sorted = df2_aligned.sort_values('utcTimestampInMinutes', ascending=True)

    # print(">>>>6 " + str(len(df1_sorted)))

    # 使用时间戳作为索引
    df1_sorted.set_index('utcTimestampInMinutes', inplace=True)
    df2_sorted.set_index('utcTimestampInMinutes', inplace=True)

    return df1_sorted, df2_sorted

holding = Holdings()

global_clent = None

async def main() -> None:
    client = await OpenApiClient.create(HOST, PORT)
    global global_clent
    global_clent = client


    XSymbolId = 0
    YSymbolId = 0

    def resetHoldingSize():
        holding.X = 0
        holding.Y = 0

    def closeOrder():
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(async_closeOrder(), loop)

    async def async_closeOrder():
        print("平仓")
        print(f"平仓:ETH:{holding.X} BTC:{holding.Y}")
        await closeAllPostion(client,ACCOUNT_ID)
        resetHoldingSize()
        print("平仓完成")
        await async_checkUsedMargin()
        await async_updateBalance()

    def createOrder(eth_size,btc_size,eth_price,btc_price):
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(async_createOrder(eth_size,btc_size,eth_price,btc_price), loop)

    async def async_createOrder(eth_size,btc_size,eth_price,btc_price):
        print(f"开仓:ETH:{eth_size} BTC:{btc_size} ETH价格:{eth_price} BTC价格:{btc_price}")
        holding.X += eth_size
        holding.Y += btc_size
        await createNewOrderRequest(client, ACCOUNT_ID, XSymbolId, abs(eth_size), ProtoOATradeSide.BUY if eth_size >0 else ProtoOATradeSide.SELL, ProtoOAOrderType.MARKET)
        await createNewOrderRequest(client, ACCOUNT_ID, YSymbolId, abs(btc_size), ProtoOATradeSide.BUY if btc_size >0 else ProtoOATradeSide.SELL, ProtoOAOrderType.MARKET)
        print("开仓完成")
        await async_checkUsedMargin()
        # await async_updateBalance()


    algo = PairsTradingBacktester(entry_z=RunParams.Enter_Z, exit_z=RunParams.Exit_Z,initial_cash=0,closeOrderExcute=closeOrder,createOrderExcute=createOrder)
    def updateBalance():
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(async_updateBalance(), loop)

    async def async_updateBalance():
        print("更新账户余额")
        trade_req = ProtoOATraderReq(ctidTraderAccountId=ACCOUNT_ID)
        trade_res = await client.send_and_wait_response(trade_req, ProtoOATraderRes)
        balance = trade_res.trader.balance
        algo._update_balance(balance)
        print(f"账户余额:{balance}")
        print("更新账户余额完成")

    def checkMargin():
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(async_checkUsedMargin(), loop)

    async def async_checkUsedMargin():
        reconcile_req = ProtoOAReconcileReq(ctidTraderAccountId=ACCOUNT_ID)
        reconcile_res = await client.send_and_wait_response(
            reconcile_req, ProtoOAReconcileRes
        )
        usedMargin = 0
        for pos in reconcile_res.position:
            usedMargin += pos.usedMargin 
            print(f"position: {pos.positionId}, usedMargin: {pos.usedMargin}, margin_rate: {pos.marginRate}")
        algo._update_used_margin(usedMargin)

    try:
        print("authenticating")
        app_auth_req = ProtoOAApplicationAuthReq(
            clientId=AccountInfo.CLIENT_ID,
            clientSecret=AccountInfo.CLIENT_SECRET,
        )
        app_auth_res = await client.send_and_wait_response(
            app_auth_req, ProtoOAApplicationAuthRes
        )

        acc_auth_req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=ACCOUNT_ID,
            accessToken=AccountInfo.ACCESS,
        )
        acc_auth_res = await client.send_and_wait_response(
            acc_auth_req, ProtoOAAccountAuthRes
        )
        print(f"res: {acc_auth_res.ctidTraderAccountId}")

        symbol_list_req = ProtoOASymbolsListReq(ctidTraderAccountId=ACCOUNT_ID)
        symbol_list_res = await client.send_and_wait_response(
            symbol_list_req, ProtoOASymbolsListRes
        )

        symbol_ids =  []
        symbol_infos = {}

        for symbol in symbol_list_res.symbol:
            if symbol.symbolName == ETHSymbolInfo.SYMBOL_NAME:
                print("find eth symbol")
                print(symbol.symbolId)
                XSymbolId = symbol.symbolId
                symbol_ids.append(symbol.symbolId)
                symbol_infos[symbol.symbolName] = SymbolInfo(symbol.symbolId,symbol.symbolName,ETHSymbolInfo.SYMBOL_LEVERAGE)
            elif symbol.symbolName == BTCSymbolInfo.SYMBOL_NAME:
                print("find btc symbol")
                print(symbol.symbolId)
                YSymbolId = symbol.symbolId
                symbol_ids.append(symbol.symbolId)
                symbol_infos[symbol.symbolName] = SymbolInfo(symbol.symbolId,symbol.symbolName,BTCSymbolInfo.SYMBOL_LEVERAGE)

        print(symbol_ids)
        algo.service = PairsTradeSignalService(data=None,symbols=symbol_infos,window=RunParams.Window,settings={'entry_z':algo.entry_z,'exit_z':algo.exit_z})

        trade_req = ProtoOATraderReq(ctidTraderAccountId=ACCOUNT_ID)
        trade_res = await client.send_and_wait_response(trade_req, ProtoOATraderRes)
        balance = trade_res.trader.balance
        print(f"balance:{trade_res.trader.balance}")

        algo._update_balance(balance)
        await async_checkUsedMargin()

        # 立即执行首次任务
        # task1 = asyncio.create_task(scheduled_task(client,trendbarData,result_queue, symbol_ids[0], symbol_ids[1]))
        await scheduled_task(client,realTime,trendbarData,result_queue, XSymbolId, YSymbolId,algo)
        # createOrder(0.44,0.01,realTime.eth,realTime.btc)

        global current_refresh_time
        current_refresh_time = datetime.datetime.now()

        tasks = []
        if True:
            sub_spot_req = ProtoOASubscribeSpotsReq(
                ctidTraderAccountId=ACCOUNT_ID, symbolId=symbol_ids
            )
            task = asyncio.create_task(
                client.send_and_wait_response(sub_spot_req, ProtoOASubscribeSpotsRes)
            )
            tasks.append(task)
        else:
            return

        print("subscribed to spots")

        await asyncio.wait(tasks)

        async with client.register_types(ProtoOASpotEvent) as gen:
            print("listening for spot events")
            async for spot_event in gen:
                print(f"got spot event {spot_event}")
                # print current date
                print(datetime.datetime.now())
                if spot_event.symbolId == XSymbolId:
                    if spot_event.ask == 0 or spot_event.bid == 0:
                        print(">>>>>>>>>>>>>>>>ask is 0")
                        continue
                    realTime.eth = (spot_event.ask + spot_event.bid) / 2 / Price_Digits
                if spot_event.symbolId == YSymbolId:
                    if spot_event.ask == 0 or spot_event.bid == 0:
                        print(">>>>>>>>>>>>>>>>ask is 0")
                        continue
                    realTime.btc = (spot_event.ask + spot_event.bid) / 2 / Price_Digits
                if realTime.eth == 0 or realTime.btc == 0:
                    print(">>>>>>>>>>>>>>>>ask is 0")
                    continue
                if trendbarData.btcData is not None:
                    current_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
                    """
                    {
                        'signal': 'long'/'short'/'close'/None,
                        'zscore': float,
                        'hedge_ratio': float,
                        'spread': float
                    }
                    """
                    
                    if True:
                        if XSymbolId == 0 or YSymbolId == 0:
                            continue
                        signal = algo._checkUpdateSignal(reset_model=False, btc_price=realTime.btc, eth_price=realTime.eth)
                        if signal is not None:
                            print(f"signal: {signal}")
                            # trade_req = ProtoOATraderReq(ctidTraderAccountId=ACCOUNT_ID)
                            # trade_res = await asyncio.wait_for(client.send_and_wait_response(trade_req,ProtoOATraderRes), timeout=5)
                            # balance = trade_res.trader.balance
                            # algo._update_balance(balance)
                            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                            # print(f"BTC:{realTime.btc} ETH:{realTime.eth}")
                            # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


                    now = datetime.datetime.now()
                    if (now - current_refresh_time).seconds > 60:
                        # await scheduled_task(client,realTime,trendbarData,result_queue, symbol_ids[0], symbol_ids[1],algo)
                        algo._update_data(btc_price=realTime.btc, eth_price=realTime.eth)
                        realTime.timestampInMin += 1
                        print("Update data every minute")
                        algo._write_all_data_to_file()
                        current_refresh_time = now
                    
                # print(spot_event)
                if isinstance(spot_event, ProtoOAErrorRes):
                    # print("error")
                    raise RuntimeError(spot_event.description)
        while True:
            print("waiting for events")
            await asyncio.sleep(1)



    except Exception as ex:
        print(ex)
        await client.close()
        raise ex
    finally:
        print("closing connection")
        await client.close()

async def run_forever():
    while True:
        try:
            await main()
        except Exception:
            # 可选：添加重启前的延迟（避免高频崩溃）
            await asyncio.sleep(5)
            print("重启 main()...")
            continue
        else:
            # 如果 main() 正常退出（非异常），则终止循环
            break

asyncio.run(run_forever(), debug=True)
print("done")
