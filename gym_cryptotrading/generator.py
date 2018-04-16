import os

import time
import datetime

import tempfile
import urllib2
import gzip

import pandas as pd

from gym import logger

from gym_cryptotrading.strings import *

class Generator:
    dataset_path = None
    temp_dir = None

    def __init__(self, history_length, horizon):
        Generator.load_gen()

        self.history_length = history_length
        self.horizon = horizon

        self._load_data()
        
    @property
    def diff_blocks(self):
        return self._diff_blocks

    @property
    def price_blocks(self):
        return self._price_blocks

    @property
    def timestamp_blocks(self):
        return self._timestamp_blocks

    def _preprocess(self):
        data = pd.read_csv(Generator.dataset_path)
        message = 'Columns found in the dataset {}'.format(data.columns)
        logger.info(message)
        data = data.dropna()
        start_time_stamp = data['Timestamp'][0]
        timestamps = data['Timestamp'].apply(lambda x: (x - start_time_stamp) / 60)
        timestamps = timestamps - range(timestamps.shape[0])
        data.insert(0, 'blocks', timestamps)
        blocks = data.groupby('blocks')
        message = 'Number of blocks of continuous prices found are {}'.format(len(blocks))
        logger.info(message)
        
        self._data_blocks = []
        distinct_episodes = 0

        for name, indices in blocks.indices.items():
            ''' 
            Length of the block should exceed the history length and horizon by 1.
            Extra 1 is required to normalize each price block by previos time stamp
            '''
            if len(indices) > (self.history_length + self.horizon + 1):
                
                self._data_blocks.append(blocks.get_group(name))
                # similarly, we subtract an extra 1 to calculate the number of distinct episodes
                distinct_episodes = distinct_episodes + (len(indices) - (self.history_length + self.horizon) + 1 + 1)

        data = None
        message_list = [
            'Number of usable blocks obtained from the dataset are {}'.format(len(self._data_blocks))
        ]
        message_list.append(
            'Number of distinct episodes for the current configuration are {}'.format(distinct_episodes)
        )
        map(logger.info, message_list)

    def _generate_attributes(self):
        self._diff_blocks = []
        self._price_blocks = []
        self._timestamp_blocks = []

        for data_block in self._data_blocks:
            block = data_block[['price_close', 'price_low', 'price_high', 'volume']]
            closing_prices = block['price_close']

            diff_block = closing_prices.shift(-1)[:-1].subtract(closing_prices[:-1])

            # currently normalizing the prices by previous prices of the same category
            normalized_block = block.shift(-1)[:-1].truediv(block[:-1])        
            
            self._diff_blocks.append(diff_block.as_matrix())
            self._price_blocks.append(normalized_block.as_matrix())
            self._timestamp_blocks.append(data_block['DateTime_UTC'].values[1:])
        
        self._data_blocks = None #free memory

    def _load_data(self):
        self._preprocess()
        self._generate_attributes()

    @staticmethod
    def get_transactions():
        if not Generator.dataset_path:
            Generator.set_dataset_path()

        message = 'Getting latest transactions from {}.'.format(URL) + \
                    '\nThis might take a few minutes depending upon your internet speed.'
        logger.info(message)
    
        path = os.path.join(Generator.temp_dir, 'coinbaseUSD.csv.gz')
        f = urllib2.urlopen(URL)
        with open(path, 'w') as buffer:
            buffer.write(f.read())
        message = 'Latest transactions saved to {}'.format(path)
        logger.info(message)

        # Read the transactions into pandas dataframe
        with gzip.open(path, 'r') as f:
            d = pd.read_table(f, sep=',', header=None, index_col=0, names=['price', 'volume'])    
        os.remove(path)

        d.index = d.index.map(lambda ts: datetime.datetime.fromtimestamp(int(ts)))
        d.index.names = ['DateTime_UTC']
        p = pd.DataFrame(d['price'].resample('1Min').ohlc())
        p.columns = ['price_open', 'price_high', 'price_low', 'price_close']
        v = pd.DataFrame(d['volume'].resample('1Min').sum())
        v.columns = ['volume']
        p['volume'] = v['volume']
        unix_timestamps = p.index.map(lambda ts: int(time.mktime(ts.timetuple())))
        p.insert(0, 'Timestamp', unix_timestamps)

        p.to_csv(Generator.dataset_path, sep=',')
        message = 'Dataset sampled and saved to {}'.format(Generator.dataset_path)
        logger.info(message)

    @staticmethod
    def update_gen():
        if not Generator.dataset_path:
            Generator.set_dataset_path()

        if os.path.isfile(Generator.dataset_path):
            os.remove(Generator.dataset_path)
        Generator.get_transactions()

    @staticmethod
    def load_gen():
        if not Generator.dataset_path:
            Generator.set_dataset_path()

        '''
        TODO: Need to do sanity check of the sampled dataset
        '''
        if not os.path.isfile(Generator.dataset_path):
            message = 'Sampled Dataset not found at {}.'.format(Generator.dataset_path) + \
                            '\nSetting up the environment for first use.'
            logger.info(message)
            Generator.get_transactions()

    @staticmethod
    def set_dataset_path():
        if not Generator.temp_dir:
            Generator.set_temp_dir()

        Generator.dataset_path = os.path.join(Generator.temp_dir, 'btc.csv')

    @staticmethod
    def set_temp_dir():
        Generator.temp_dir = tempfile.gettempdir()
