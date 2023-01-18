import os
import requests
import base64
import numpy as np
import pandas as pd
import json
import torch
import wfdb

if __name__ == '__main__':

    images = []
    for name in os.listdir('./mnist'):
        images.append(np.load('./mnist/' + name))
    images = np.array(images).astype(np.float32)

    url = 'http://0.0.0.0:8080/predictions/mnist_classification'
    X = []
    for row in images[:2]:
        row = row.tobytes()
        X.append(base64.b64encode(row).decode())
    obj = {
        'data': X,
    }
    response = requests.post(url, data=json.dumps(obj), headers={"Content-Type": "application/json"})
    predicts = np.frombuffer(response.content, dtype=np.float32).astype(np.int)
    print(predicts)
