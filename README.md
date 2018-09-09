# Training ShuffleNets with TPU

## How to use this
I assume that you know how to use Google Cloud Platform.

1. You need to prepare ImageNet data. Create a cheap compute instance and follow steps from [here](https://github.com/TropComplique/shufflenet-v2-tensorflow/tree/master/data).  
You will need to connect a disk with size ~400 GB to the instance.
2. Upload created data shards to a bucket.
3. Create an instance with TPU and run `python train.py`.

## Credit
It is based on:
https://github.com/tensorflow/tpu/tree/master/models/official/resnet
