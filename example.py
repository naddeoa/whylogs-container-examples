import pandas as pd
import asyncio
import typing as t
import time
import numpy as np
from tqdm import tqdm
from whylogs_container_client import Configuration, WhylogsApi, ApiClient

config = Configuration()
config.host = 'http://localhost:8080'
client = ApiClient(config)
whylogs_api = WhylogsApi(client)

times: t.Dict[str, t.List[float]] = {}

def timer(name):
    def wrapped(fn):
        def timerfn(*args):
            start= time.perf_counter()
            fn(*args)
            end = time.perf_counter()
            total = end - start 
            # print(f"{name} took {end - start} seconds")
            if name not in times:
              times[name] = []

            times[name].append(total)
        return timerfn
    return  wrapped

def print_times():

  for k,v in times.items():
    print(f'==== {k} times ====')
    print(f'p50 {np.percentile(v, 50):.3f}')
    print(f'p95 {np.percentile(v, 95):.3f}')
    print(f'p99 {np.percentile(v, 99):.3f}')
    print()


@timer('send_to_container_sync')
def send_to_container_sync(data: pd.DataFrame):
  multiple = data.to_dict(orient="split")
  del multiple['index'] # get rid of this to min/max payload size

  payload = {
    'datasetId': 'fake-id',
    'multiple': multiple
  }

  whylogs_api.track(body=payload, x_api_key='password')


@timer('send_to_container_async')
async def send_to_container_async(data: pd.DataFrame):
  multiple = data.to_dict(orient="split")
  del multiple['index'] # get rid of this to min/max payload size

  payload = {
    'datasetId': 'fake-id',
    'multiple': multiple
  }

  whylogs_api.track(body=payload, x_api_key='password')


@timer('inference_sync')
def inference_sync(data: pd.DataFrame):
  print(len(data))
  time.sleep(1)
  send_to_container_sync(data)

@timer('inference_async')
def inference_async(data: pd.DataFrame):
  print(len(data))
  time.sleep(1)
  send_to_container_async(data)


if __name__ == "__main__":

  df = pd.read_csv('./data.csv')
  for i in tqdm(range(3)):
    inference_sync(df)

  print_times()
  times = {}

  for i in tqdm(range(3)):
    inference_async(df)
  print_times()