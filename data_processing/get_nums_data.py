import pandas as pd
import logging
from multiprocessing import Pool
import multiprocessing as mp

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log 출력
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# log를 파일에 출력
file_handler = logging.FileHandler('data_total.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# data = pd.read_csv('/home/ailab/Desktop/NY/2023_ipactory/data/csv/data.csv', chunksize=1000000)
data = pd.read_csv
    
len_data = 0

def count_csv(data) :
    c_proc = mp.current_process()
    new = next(data)
    len_data += len(new)
    logger.info(f'Process : {c_proc.name} {c_proc.pid} | Total Length of CSV : {len_data}')
    new = None

while True :
    p = Pool(10)
    p.map_async(count_csv, data)

    p.close()
    p.join()

# while True :
#     new = next(data)
#     len_data += len(new)
#     logger.info(f'Total Length of CSV : {len_data}')
#     new = None