"""Send JPEG image to tensorflow_model_server loaded with ip5wke model.
"""

import numpy

from grpc.beta import implementations
from time import time
import tensorflow as tf

import predict_pb2
import prediction_service_pb2

tf.app.flags.DEFINE_string("host", "127.0.0.1", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "ip5wke", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 100.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS


def main():
    host = FLAGS.host
    port = FLAGS.port
    model_name = FLAGS.model_name
    model_version = FLAGS.model_version
    request_timeout = FLAGS.request_timeout

    # Create gRPC client and request
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'predict_images'
    if model_version > 0:
        request.model_spec.version.value = model_version
    with open('/media/windows/DEV/IP5/DSC_0241.JPG', 'rb') as f:
        # See prediction_service.proto for gRPC request/response details.
        data = f.read()
        print(time())

        for i in range(100):
            request.inputs['images'].CopyFrom(
                tf.contrib.util.make_tensor_proto(data, shape=[1]))

            # Send request
            result = stub.Predict(request, request_timeout)
            print(result)
        print(time())

if __name__ == '__main__':
    main()