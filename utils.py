from enum import Enum
import requests

import os
is_prod = os.environ.get('IS_HEROKU', None)

class Endpoints(Enum):
    local = "http://127.0.0.1:5000"
    remote = "https://hatespeech-api.herokuapp.com"

if is_prod:
    url_root = Endpoints.remote.value
else: 
    url_root = Endpoints.local.value
    

def fetch(URL:str, PARAMS:dict)->dict:
    r = requests.get(url = URL, params = PARAMS)
    return r.json()
    

def get_prediction(text_to_predict: str):
    URL = f"{url_root}/query"
    PARAMS = {'code':text_to_predict}

    data = fetch(URL, PARAMS)
    
    return data['data']
