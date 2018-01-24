# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with cat-no-cat model.
The client sends a list of image urls (can be in google storage) to the server,
and the server parses and resizes them into appropriate image tensors for
cat-no-cat prediction.
"""

from __future__ import print_function

import cv2
import numpy as np
import urllib
import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

from step_2b_get_images import resize_and_pad_image

# Command line arguments
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path or url to image')
tf.app.flags.DEFINE_string('model', 'cats', 'name of model to call')
tf.app.flags.DEFINE_string('size', '128', 'size of image to send to model')
FLAGS = tf.app.flags.FLAGS


def _int_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def main(_):

  # Things that you need ot do to send an RPC request to the TF server.
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()

  feature = None
  if 'http' in FLAGS.image:
    resp = urllib.urlopen(FLAGS.image)
    feature = np.asarray(bytearray(resp.read()), dtype="uint8")
    feature = cv2.imdecode(feature, cv2.IMREAD_COLOR)
  else:
    feature = cv2.imread(FLAGS.image)  # Parse the image from your local disk.

  feature = cv2.cvtColor(feature, cv2.COLOR_BGR2RGB)  # Flip the RGB (cv2 issue)

  # Resize and pad the image
  feature = resize_and_pad_image(feature, output_image_dim=int(FLAGS.size))

  # tf.train.Features only takes at most a 1d array. Flatten the image to 1d.
  flat_feature = feature.flatten()
  feature_dict = {'flattened_image': _int_feature(flat_feature)}
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  serialized = example.SerializeToString()  # Serialize the example

  # Call CAT cnn model to make prediction on the image
  request.model_spec.name = FLAGS.model
  # Convert serialized string to protobuf
  request.inputs['examples'].CopyFrom(
    tf.contrib.util.make_tensor_proto(serialized, shape=[1]))

  # Call the server to predict, and return the result
  result = stub.Predict(request, 5.0)  # 5 secs timeout
  print(result)


if __name__ == '__main__':
  tf.app.run()