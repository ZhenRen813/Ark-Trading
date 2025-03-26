
from waterstart.client import OpenApiClient


class RZDataService:
    def __init__(self, client: OpenApiClient, account_id: int):
        self.client = client
        self.account_id = account_id

    def get_data(self):
        return self.data

    def get_data_by_id(self, id):
        return self.data[id]

    def get_data_by_name(self, name):
        for item in self.data:
            if item['name'] == name:
                return item
        return None

    def get_data_by_age(self, age):
        result = []
        for item in self.data:
            if item['age'] == age:
                result.append(item)
        return result

    def