"""
Module responsible to setup configurations

"""

import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

'''Load env variables'''
print('Setting up environment variables!!!')
ENV = os.getenv("ENV", default='config')
load_dotenv(dotenv_path=BASE_DIR + '/configs/' + ENV + '.env')
