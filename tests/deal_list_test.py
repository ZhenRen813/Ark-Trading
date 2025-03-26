import asyncio
from datetime import timedelta
from typing import TypeVar

from livetrendbar_test import AccountInfo
from aioitertools import next
from google.protobuf.message import Message
from waterstart.client import OpenApiClient
from waterstart.openapi import (
    ProtoOAAccountAuthReq,
    ProtoOAAccountAuthRes,
    ProtoOAApplicationAuthReq,
    ProtoOAApplicationAuthRes,
    ProtoOADealListReq,
    ProtoOADealListRes,
    ProtoOAReconcileReq,
    ProtoOAReconcileRes,
)

HOST = "demo.ctraderapi.com"
PORT = 5035
ACCOUNT_ID = AccountInfo.ACCOUNT_ID


T = TypeVar("T", bound=Message)


async def main() -> None:
    client = await OpenApiClient.create(HOST, PORT)

    # refresh_token => wwJXiEBoC-7Uu4NzyNf90iWIZRlFCUdW4jUWBYoDOYs

    try:
        app_auth_req = ProtoOAApplicationAuthReq(
            clientId= AccountInfo.CLIENT_ID,
            clientSecret= AccountInfo.CLIENT_SECRET,
        )
        app_auth_res = await client.send_and_wait_response(
            app_auth_req, ProtoOAApplicationAuthRes
        )

        acc_auth_req = ProtoOAAccountAuthReq(
            ctidTraderAccountId=ACCOUNT_ID,
            accessToken= AccountInfo.ACCESS,
        )
        acc_auth_res = await client.send_and_wait_response(
            acc_auth_req, ProtoOAAccountAuthRes
        )

        reconcile_req = ProtoOAReconcileReq(ctidTraderAccountId=ACCOUNT_ID)
        reconcile_res = await client.send_and_wait_response(
            reconcile_req, ProtoOAReconcileRes
        )

        position = reconcile_res.position[0]
        for pos in reconcile_res.position:
            print(pos)

        from_timestamp = position.tradeData.openTimestamp
        to_timestamp = position.utcLastUpdateTimestamp

        assert to_timestamp - from_timestamp < timedelta(weeks=1).total_seconds()

        deal_list_req = ProtoOADealListReq(
            ctidTraderAccountId=ACCOUNT_ID,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
        )

        deal_list_res = await client.send_and_wait_response(
            deal_list_req, ProtoOADealListRes
        )

        pos_deals = [
            deal
            for deal in deal_list_res.deal
            if deal.positionId == position.positionId
        ]

        for deal in pos_deals:
            print(deal)

    finally:
        await client.close()


asyncio.run(main(), debug=True)
