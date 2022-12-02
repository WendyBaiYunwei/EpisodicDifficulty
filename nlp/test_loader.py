import configparser
import pickle
import os

config = configparser.ConfigParser()
config.read("config.ini")
with open(os.path.join(config['data']['path'], config['data']['train_loader']), 'rb') as f:
    loader = pickle.load(f)
loader.get_batch(2)
loader.get_batch(1)