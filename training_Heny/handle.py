import base64
import time
from abc import ABC


import torch
import numpy as np
from ts.torch_handler.base_handler import BaseHandler

from model import Net


class MNISTCLassificationHandler(BaseHandler, ABC):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self):
        super(MNISTCLassificationHandler).__init__()
        self.max_length = None
        self.sample_method = None
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties

        self.model = Net()
        self.model.load_state_dict(torch.load('mnist_cnn.pt'))
        self.model.eval()

        self.initialized = True

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        body = requests[0].get('body')
        data = body.get('data')
        arrays = []
        for item in data:
            bytes_array = base64.b64decode(item)
            array = np.frombuffer(bytes_array, dtype=np.float32)
            array = torch.from_numpy(array).type(torch.FloatTensor)
            array = array.view(28, 28)
            arrays.append(array)
        arrays = torch.stack(arrays, 0)

        return arrays.unsqueeze(1)

    def inference(self, x):
        """
        Given the data from .preprocess, perform inference using the models.
        We return the predicted label for each image.
        """
        output = self.model(x)
        output = output.argmax(dim=1)

        return output

    def postprocess(self, preds):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        preds = preds.detach().numpy().astype(np.float32)
        preds = preds.tobytes()

        return [preds]

    def handle(self, data, context):
        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics
        data_preprocess = self.preprocess(data)
        data_inference = self.inference(data_preprocess)
        data_postprocess = self.postprocess(data_inference)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )

        return data_postprocess
