# TensorFlow-Slim image classification model library

# TensorFlow-Slim图像分类模型库。

[TF-slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
is a new lightweight high-level API of TensorFlow (`tensorflow.contrib.slim`)
for defining, training and evaluating complex
models. This directory contains
code for training and evaluating several widely used Convolutional Neural
Network (CNN) image classification models using TF-slim.
It contains scripts that will allow

是一个新的轻量级的TensorFlow高级API用于定义、培训和评估复杂情况。模型。这个目录包含训练和评估几种广泛使用的卷积神经的代码。网络(CNN)图像分类模型使用TF-slim。

you to train models from scratch or fine-tune them from pre-trained network
weights. It also contains code for downloading standard image datasets,
converting them
to TensorFlow's native TFRecord format and reading them in using TF-Slim's
data reading and queueing utilities. 

您可以从头开始训练模型，或者从预先训练的网络中对模型进行微调。权重。它还包含用于下载标准图像数据集的代码，把他们对TensorFlow的本机TFRecord格式进行阅读，并使用TF-Slim的。数据读取和队列实用程序。

You can easily train any model on any of
these datasets, as we demonstrate below. We've also included a
[jupyter notebook](https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb),

你可以很容易地训练任何模型。这些数据集，如下所示。我们还包括一个

which provides working examples of how to use TF-Slim for image classification.

它提供了如何使用TF-Slim进行图像分类的实例。

For developing or modifying your own models, see also the [main TF-Slim page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

要开发或修改您自己的模型，请参见。

## Contacts

Maintainers of TF-slim:

TF-slim维护人员:

* Nathan Silberman,
  github: [nathansilberman](https://github.com/nathansilberman)
* Sergio Guadarrama, github: [sguada](https://github.com/sguada)

## Table of contents

<a href="#Install">Installation and setup</a><br>
<a href='#Data'>Preparing the datasets</a><br>
<a href='#Pretrained'>Using pre-trained models</a><br>
<a href='#Training'>Training from scratch</a><br>
<a href='#Tuning'>Fine tuning to a new task</a><br>
<a href='#Eval'>Evaluating performance</a><br>
<a href='#Export'>Exporting Inference Graph</a><br>
<a href='#Troubleshooting'>Troubleshooting</a><br>

# Installation
<a id='Install'></a>

In this section, we describe the steps required to install the appropriate
prerequisite packages.

在本节中，我们将描述安装适当的步骤所需的步骤。先决条件包。

## Installing latest version of TF-slim

## 安装最新版本的TF-slim。

TF-Slim is available as `tf.contrib.slim` via TensorFlow 1.0. To test that your
installation is working, execute the following command; it should run without
raising any errors.

TF-Slim是可用的“tf.o.b”。通过1.0 TensorFlow苗条的。测试你的安装工作，执行以下命令;它应该没有运行提高任何错误。

```
python -c "import tensorflow.contrib.slim as slim; eval = slim.evaluation.evaluate_once"
```

## Installing the TF-slim image models library

## 安装TF-slim图像模型库。

To use TF-Slim for image classification, you also have to install

要使用TF-Slim进行图像分类，还需要安装。

the [TF-Slim image models library](https://github.com/tensorflow/models/tree/master/research/slim),
which is not part of the core TF library.
它不是核心TF库的一部分。
To do this, check out the
[tensorflow/models](https://github.com/tensorflow/models/) repository as follows:

```bash
cd $HOME/workspace
git clone https://github.com/tensorflow/models/
```

This will put the TF-Slim image models library in `$HOME/workspace/models/research/slim`.

这将使TF-Slim图像模型库进入。

(It will also create a directory called

它还将创建一个名为的目录。

[models/inception](https://github.com/tensorflow/models/tree/master/research/inception),
which contains an older version of slim; you can safely ignore this.)

其中包含了较老版本的slim;你可以忽略这个。

To verify that this has worked, execute the following commands; it should run
without raising any errors.

为了验证这是否有效，执行以下命令;它应该运行不增加任何错误。

```
cd $HOME/workspace/models/research/slim
python -c "from nets import cifarnet; mynet = cifarnet.cifarnet"
```


# Preparing the datasets
<a id='Data'></a>

As part of this library, we've included scripts to download several popular
image datasets (listed below) and convert them to slim format.

作为这个库的一部分，我们已经包含了下载几个流行的脚本。图像数据集(如下所列)并转换为瘦格式。

Dataset | Training Set Size | Testing Set Size | Number of Classes | Comments
:------:|:---------------:|:---------------------:|:-----------:|:-----------:
Flowers|2500 | 2500 | 5 | Various sizes (source: Flickr)
[Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) | 60k| 10k | 10 |32x32 color
[MNIST](http://yann.lecun.com/exdb/mnist/)| 60k | 10k | 10 | 28x28 gray
[ImageNet](http://www.image-net.org/challenges/LSVRC/2012/)|1.2M| 50k | 1000 | Various sizes

## Downloading and converting to TFRecord format

## 下载并转换为TFRecord格式。

For each dataset, we'll need to download the raw data and convert it to
TensorFlow's native

对于每个数据集，我们需要下载原始数据并将其转换为。 TensorFlow的本机

[TFRecord](https://www.tensorflow.org/versions/r0.10/api_docs/python/python_io.html#tfrecords-format-details)
format. Each TFRecord contains a

格式。每个TFRecord包含一个

[TF-Example](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/core/example/example.proto)
protocol buffer. Below we demonstrate how to do this for the Flowers dataset.

协议缓冲区。下面我们将演示如何为花卉数据集实现这一点。

```shell
$ DATA_DIR=/tmp/data/flowers
$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir="${DATA_DIR}"
```

When the script finishes you will find several TFRecord files created:

当脚本完成时，您将发现创建的几个TFRecord文件:

```shell
$ ls ${DATA_DIR}
flowers_train-00000-of-00005.tfrecord
...
flowers_train-00004-of-00005.tfrecord
flowers_validation-00000-of-00005.tfrecord
...
flowers_validation-00004-of-00005.tfrecord
labels.txt
```

These represent the training and validation data, sharded over 5 files each.
You will also find the `$DATA_DIR/labels.txt` file which contains the mapping
from integer labels to class names.

这些代表了培训和验证数据，每个文件分成5个文件。您还将找到“$DATA_DIR/标签”。包含映射的txt文件。从整数标签到类名。

You can use the same script to create the mnist and cifar10 datasets.
However, for ImageNet, you have to follow the instructions
[here](https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started).
Note that you first have to sign up for an account at image-net.org.
Also, the download can take several hours, and could use up to 500GB.

您可以使用相同的脚本创建mnist和cifar10数据集。但是，对于ImageNet，您必须按照说明进行操作。 [这](https://github.com/tensorflow/models/blob/master/research/inception/README.md #开始)。请注意，您首先必须在image-net.org上注册一个帐户。另外，下载需要几个小时，最多可以使用500GB。

## Creating a TF-Slim Dataset Descriptor.

## 创建一个TF-Slim数据集描述符。

Once the TFRecord files have been created, you can easily define a Slim

一旦创建了TFRecord文件，您就可以轻松定义一个Slim。

[Dataset](https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/contrib/slim/python/slim/data/dataset.py),
which stores pointers to the data file, as well as various other pieces of
metadata, such as the class labels, the train/test split, and how to parse the
TFExample protos. We have included the TF-Slim Dataset descriptors
for

哪个存储指向数据文件的指针，以及其他的其他部分? 元数据，例如类标签、火车/测试分离，以及如何解析。 TFExample原型。我们已经包含了TF-Slim数据集描述符。为

[Cifar10](https://github.com/tensorflow/models/blob/master/research/slim/datasets/cifar10.py),
[ImageNet](https://github.com/tensorflow/models/blob/master/research/slim/datasets/imagenet.py),
[Flowers](https://github.com/tensorflow/models/blob/master/research/slim/datasets/flowers.py),
and
[MNIST](https://github.com/tensorflow/models/blob/master/research/slim/datasets/mnist.py).
An example of how to load data using a TF-Slim dataset descriptor using a
TF-Slim

使用一个TF-Slim数据集描述符加载数据的示例。 TF-Slim

[DatasetDataProvider](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/data/dataset_data_provider.py)
is found below:

```python
import tensorflow as tf
from datasets import flowers

slim = tf.contrib.slim

# Selects the 'validation' dataset.
dataset = flowers.get_split('validation', DATA_DIR)

# Creates a TF-Slim DataProvider which reads the dataset in the background
# during both training and testing.
provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
[image, label] = provider.get(['image', 'label'])
```
## An automated script for processing ImageNet data.

## 用于处理ImageNet数据的自动化脚本。

Training a model with the ImageNet dataset is a common request. To facilitate
working with the ImageNet dataset, we provide an automated script for
downloading and processing the ImageNet dataset into the native TFRecord
format.

使用ImageNet数据集训练模型是一个常见的请求。为了方便使用ImageNet数据集，我们提供了一个自动化的脚本。下载并处理ImageNet数据集到本机TFRecord。格式。

The TFRecord format consists of a set of sharded files where each entry is a serialized `tf.Example` proto. Each `tf.Example` proto contains the ImageNet image (JPEG encoded) as well as metadata such as label and bounding box information.

TFRecord格式由一组分片文件组成，其中每个条目都是一个序列化的“tf”。例子的原型。每个“特遣部队。示例“proto包含ImageNet图像(JPEG编码)和元数据，例如标签和边框信息。

We provide a single [script](datasets/download_and_preprocess_imagenet.sh) for
downloading and converting ImageNet data to TFRecord format. Downloading and
preprocessing the data may take several hours (up to half a day) depending on
your network and computer speed. Please be patient.

我们提供一个单独的[脚本](数据集/download_and_preprocess_imagenet.sh)。下载和转换ImageNet数据到TFRecord格式。下载和根据不同的情况，预处理数据可能需要几个小时(最多半天)。你的网络和电脑速度。请耐心等待。

To begin, you will need to sign up for an account with [ImageNet]
(http://image-net.org) to gain access to the data. Look for the sign up page,
create an account and request an access key to download the data.

首先，您需要注册一个帐户(ImageNet) (http://image.net.org)获取数据。查找注册页面，创建一个帐户并请求一个访问密钥来下载数据。

After you have `USERNAME` and `PASSWORD`, you are ready to run our script. Make
sure that your hard disk has at least 500 GB of free space for downloading and
storing the data. Here we select `DATA_DIR=$HOME/imagenet-data` as such a
location but feel free to edit accordingly.

在您拥有“用户名”和“密码”之后，您就可以运行我们的脚本了。使当然，你的硬盘至少有500gb的免费空间供下载。存储的数据。这里我们选择“DATA_DIR=$HOME/imagenet-data”。位置，但可以随意编辑。

When you run the below script, please enter *USERNAME* and *PASSWORD* when
prompted. This will occur at the very beginning. Once these values are entered,
you will not need to interact with the script again.

当您运行以下脚本时，请输入*用户名*和*密码*。提示。这将在一开始就发生。一旦输入了这些值，您将不需要再次与脚本交互。

```shell
# location of where to place the ImageNet data
DATA_DIR=$HOME/imagenet-data

# build the preprocessing script.
bazel build slim/download_and_preprocess_imagenet

# run it
bazel-bin/slim/download_and_preprocess_imagenet "${DATA_DIR}"
```

The final line of the output script should read:

输出脚本的最后一行应该是:

```shell
2016-02-17 14:30:17.287989: Finished writing all 1281167 images in data set.
```

When the script finishes you will find 1024 and 128 training and validation
files in the `DATA_DIR`. The files will match the patterns `train-????-of-1024`
and `validation-?????-of-00128`, respectively.

当脚本完成后，您将找到1024和128的训练和验证。文件“DATA_DIR”。这些文件将与“1024”的模式匹配。和“验证- ? ? ? ? ?- 00128年的分别。

[Congratulations!](https://www.youtube.com/watch?v=9bZkp7q19f0) You are now
ready to train or evaluate with the ImageNet data set.

你现在准备好使用ImageNet数据集进行培训或评估。

# Pre-trained Models
<a id='Pretrained'></a>

Neural nets work best when they have many parameters, making them powerful
function approximators.

当神经网络有很多参数的时候，它们的工作效率最好。函数近似者。

However, this  means they must be trained on very large datasets. Because
training models from scratch can be a very computationally intensive process
requiring days or even weeks, we provide various pre-trained models,
as listed below. These CNNs have been trained on the
[ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/)
image classification dataset.

然而，这意味着它们必须在非常大的数据集上进行训练。因为从头开始训练模型可以是一个非常计算密集型的过程。需要几天甚至几周的时间，我们提供各种预先训练的模型，如下列出。这些CNNs已经过培训。 (ilsvrc - 2012 cls)(http://www.image-net.org/challenges/LSVRC/2012/) 图像分类的数据集。

In the table below, we list each model, the corresponding
TensorFlow model file, the link to the model checkpoint, and the top 1 and top 5
accuracy (on the imagenet test set).

在下面的表格中，我们列出了每个模型，对应的。 TensorFlow模型文件，连接到模型检查点，以及顶部1和前5。精度(在imagenet测试集上)。

Note that the VGG and ResNet V1 parameters have been converted from their original
caffe formats

注意，VGG和ResNet V1参数已经从它们的原始参数转换了。咖啡的格式

([here](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014)
and
[here](https://github.com/KaimingHe/deep-residual-networks)),
whereas the Inception and ResNet V2 parameters have been trained internally at
Google. Also be aware that these accuracies were computed by evaluating using a
single image crop. Some academic papers report higher accuracy by using multiple
crops at multiple scales.

而初始和ResNet V2参数已经在内部进行了训练。谷歌。还要注意，这些精度是通过使用a来计算的。单一作物图像。一些学术论文使用倍数来提高准确性。在多尺度作物。

Model | TF-Slim File | Checkpoint | Top-1 Accuracy| Top-5 Accuracy |
:----:|:------------:|:----------:|:-------:|:--------:|
[Inception V1](http://arxiv.org/abs/1409.4842v1)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v1.py)|[inception_v1_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)|69.8|89.6|
[Inception V2](http://arxiv.org/abs/1502.03167)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v2.py)|[inception_v2_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz)|73.9|91.8|
[Inception V3](http://arxiv.org/abs/1512.00567)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py)|[inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)|78.0|93.9|
[Inception V4](http://arxiv.org/abs/1602.07261)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py)|[inception_v4_2016_09_09.tar.gz](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz)|80.2|95.2|
[Inception-ResNet-v2](http://arxiv.org/abs/1602.07261)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py)|[inception_resnet_v2_2016_08_30.tar.gz](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz)|80.4|95.3|
[ResNet V1 50](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py)|[resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)|75.2|92.2|
[ResNet V1 101](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py)|[resnet_v1_101_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz)|76.4|92.9|
[ResNet V1 152](https://arxiv.org/abs/1512.03385)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py)|[resnet_v1_152_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz)|76.8|93.2|
[ResNet V2 50](https://arxiv.org/abs/1603.05027)^|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py)|[resnet_v2_50_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)|75.6|92.8|
[ResNet V2 101](https://arxiv.org/abs/1603.05027)^|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py)|[resnet_v2_101_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz)|77.0|93.7|
[ResNet V2 152](https://arxiv.org/abs/1603.05027)^|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py)|[resnet_v2_152_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz)|77.8|94.1|
[ResNet V2 200](https://arxiv.org/abs/1603.05027)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v2.py)|[TBA]()|79.9\*|95.2\*|
[VGG 16](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py)|[vgg_16_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)|71.5|89.8|
[VGG 19](http://arxiv.org/abs/1409.1556.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py)|[vgg_19_2016_08_28.tar.gz](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz)|71.1|89.8|
[MobileNet_v1_1.0_224](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_1.0_224_2017_06_14.tar.gz](http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz)|70.7|89.5|
[MobileNet_v1_0.50_160](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.50_160_2017_06_14.tar.gz](http://download.tensorflow.org/models/mobilenet_v1_0.50_160_2017_06_14.tar.gz)|59.9|82.5|
[MobileNet_v1_0.25_128](https://arxiv.org/pdf/1704.04861.pdf)|[Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py)|[mobilenet_v1_0.25_128_2017_06_14.tar.gz](http://download.tensorflow.org/models/mobilenet_v1_0.25_128_2017_06_14.tar.gz)|41.3|66.2|

^ ResNet V2 models use Inception pre-processing and input image size of 299 (use

ResNet V2模型使用初始化预处理和输入图像大小为299。

`--preprocessing_name inception --eval_image_size 299` when using
`eval_image_classifier.py`). Performance numbers for ResNet V2 models are
reported on the ImageNet validation set.

ResNet V2模型的性能数字是。在ImageNet验证集上报告。

All 16 MobileNet Models reported in the [MobileNet Paper](https://arxiv.org/abs/1704.04861) can be found [here](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet_v1.md).

可以在[MobileNet纸]中发现所有16个MobileNet模型(https://arxiv.org/abs/1704.04861)。

(\*): Results quoted from the [paper](https://arxiv.org/abs/1603.05027).

Here is an example of how to download the Inception V3 checkpoint:

下面是一个如何下载Inception V3检查点的示例:

```shell
$ CHECKPOINT_DIR=/tmp/checkpoints
$ mkdir ${CHECKPOINT_DIR}
$ wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
$ tar -xvf inception_v3_2016_08_28.tar.gz
$ mv inception_v3.ckpt ${CHECKPOINT_DIR}
$ rm inception_v3_2016_08_28.tar.gz
```



# Training a model from scratch.
<a id='Training'></a>

We provide an easy way to train a model from scratch using any TF-Slim dataset.
The following example demonstrates how to train Inception V3 using the default
parameters on the ImageNet dataset.

我们提供了一种简单的方法，可以使用任何TF-Slim数据集从头开始训练模型。下面的示例演示了如何使用缺省值来训练Inception V3。 ImageNet数据集的参数。

```shell
DATASET_DIR=/tmp/imagenet
TRAIN_DIR=/tmp/train_logs
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v3
```

This process may take several days, depending on your hardware setup.
For convenience, we provide a way to train a model on multiple GPUs,
and/or multiple CPUs, either synchrononously or asynchronously.

这个过程可能需要几天，取决于您的硬件设置。为了方便起见，我们提供了在多个gpu上训练模型的方法，和/或多个cpu，无论是同步的还是异步的。

See [model_deploy](https://github.com/tensorflow/models/blob/master/research/slim/deployment/model_deploy.py)
for details.

### TensorBoard

To visualize the losses and other metrics during training, you can use

为了在训练过程中可视化损失和其他指标，您可以使用。

[TensorBoard](https://github.com/tensorflow/tensorboard)
by running the command below.

通过运行下面的命令。

```shell
tensorboard --logdir=${TRAIN_DIR}
```

Once TensorBoard is running, navigate your web browser to http://localhost:6006.

# Fine-tuning a model from an existing checkpoint
<a id='Tuning'></a>

Rather than training from scratch, we'll often want to start from a pre-trained
model and fine-tune it.
To indicate a checkpoint from which to fine-tune, we'll call training with
the `--checkpoint_path` flag and assign it an absolute path to a checkpoint
file.

我们不需要从头开始训练，而是要从预先训练开始。模型和调整。为了指示一个检查点，以便进行调优，我们将调用培训。 “——checkpoint_path”标志，并将其指定为检查点的绝对路径。文件。

When fine-tuning a model, we need to be careful about restoring checkpoint
weights. In particular, when we fine-tune a model on a new task with a different
number of output labels, we wont be able restore the final logits (classifier)
layer. For this, we'll use the `--checkpoint_exclude_scopes` flag. 

当对模型进行微调时，我们需要注意恢复检查点。权重。特别是，当我们对一个新任务的模型进行微调时，会有不同的结果。输出标签数量，我们无法恢复最终的逻辑(分类器) 层。为此，我们将使用“—checkpoint_de_scope”标志。

This flag
hinders certain variables from being loaded. When fine-tuning on a
classification task using a different number of classes than the trained model,
the new model will have a final 'logits' layer whose dimensions differ from the
pre-trained model. 

这个标志阻止某些变量被加载。当微调分类任务使用不同数量的类，而不是训练过的模型，新型号将会有一个最终的“logits”层，其尺寸与之不同。 pre-trained模型。

For example, if fine-tuning an ImageNet-trained model on
Flowers, the pre-trained logits layer will have dimensions `[2048 x 1001]` but
our new logits layer will have dimensions `[2048 x 5]`. Consequently, this
flag indicates to TF-Slim to avoid loading these weights from the checkpoint.

例如，如果对一个imagenet训练的模型进行微调。花，预训练的物流层将有尺寸' [2048 x 1001] '但是。我们的新物流层将有尺寸[2048 x 5]。因此,该标志指示TF-Slim，以避免从检查点装载这些重量。

Keep in mind that warm-starting from a checkpoint affects the model's weights
only during the initialization of the model. Once a model has started training,
a new checkpoint will be created in `${TRAIN_DIR}`. 

请记住，从一个检查点开始的热身影响了模型的权重。只有在模型初始化期间。一旦一个模型开始训练，将在“${TRAIN_DIR}”中创建一个新的检查点。

If the fine-tuning
training is stopped and restarted, this new checkpoint will be the one from
which weights are restored and not the `${checkpoint_path}$`. Consequently,
the flags `--checkpoint_path` and `--checkpoint_exclude_scopes` are only used
during the `0-`th global step (model initialization). 

如果微调培训停止并重新启动，这个新的检查点将会是一个。哪些权重被恢复，而不是“${checkpoint_path}$”。因此, 只使用标记' -checkpoint_path '和' - checkpoint_de_scope '。在“0-”全局步骤(模型初始化)期间。

Typically for fine-tuning
one only want train a sub-set of layers, so the flag `--trainable_scopes` allows
to specify which subsets of layers should trained, the rest would remain frozen.

通常的微调一个只需要训练一个层次的子集合，所以这个标记' -trainable_scope '允许。要指定哪些子集需要训练，其余的将保持冻结。

Below we give an example of
[fine-tuning inception-v3 on flowers](https://github.com/tensorflow/models/blob/master/research/slim/scripts/finetune_inception_v3_on_flowers.sh),
inception_v3  was trained on ImageNet with 1000 class labels, but the flowers
dataset only have 5 classes. Since the dataset is quite small we will only train
the new layers.

下面我们举一个例子。 [微调inception-v3花)(https://github.com/tensorflow/models/blob/master/research/slim/scripts/finetune_inception_v3_on_flowers.sh), inception_v3在ImageNet上接受了1000个类标签的训练，但是这些花。数据集只有5个类。因为数据集很小，所以我们只会训练。新层。

```shell
$ DATASET_DIR=/tmp/flowers
$ TRAIN_DIR=/tmp/flowers-models/inception_v3
$ CHECKPOINT_PATH=/tmp/my_checkpoints/inception_v3.ckpt
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=inception_v3 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
    --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits
```



# Evaluating performance of a model
<a id='Eval'></a>

评估模型的性能

To evaluate the performance of a model (whether pretrained or your own),
you can use the eval_image_classifier.py script, as shown below.

评估一个模型的表现(无论是预先训练的还是你自己的)，您可以使用eval_image_classifier。py脚本，如下所示。

Below we give an example of downloading the pretrained inception model and
evaluating it on the imagenet dataset.

下面我们给出一个下载预先训练的初始模型的例子。在imagenet数据集上进行评估。

```shell
CHECKPOINT_FILE = ${CHECKPOINT_DIR}/inception_v3.ckpt  # Example
$ python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=validation \
    --model_name=inception_v3
```

See the [evaluation module example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim#evaluation-loop) for an example of how to evaluate a model at multiple checkpoints during or after the training.

举例说明如何在培训期间或培训结束后在多个检查点评估模型。

# Exporting the Inference Graph
<a id='Export'></a>

Saves out a GraphDef containing the architecture of the model.

To use it with a model name defined by slim, run:

保存一个包含模型架构的GraphDef。 要使用由slim定义的模型名称，请运行:

```shell
$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/tmp/inception_v3_inf_graph.pb

$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=mobilenet_v1 \
  --image_size=224 \
  --output_file=/tmp/mobilenet_v1_224.pb
```

## Freezing the exported Graph

冻结导出的图


If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

如果你想要用你自己的或预先训练过的模型。检查点作为移动模型的一部分，您可以运行freeze_graph来获得一个图形。定义为常数的变量:

```shell
bazel build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
```

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

输出节点的名称将根据模型的不同而有所不同，但是您可以进行检查。使用summary ze_graph工具估计它们:

```shell
bazel build tensorflow/tools/graph_transforms:summarize_graph

bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=/tmp/inception_v3_inf_graph.pb
```

## Run label image in C++

To run the resulting graph in C++, you can look at the label_image sample code:

要在c++中运行结果图，可以查看label_image示例代码:

```shell
bazel build tensorflow/examples/label_image:label_image

bazel-bin/tensorflow/examples/label_image/label_image \
  --image=${HOME}/Pictures/flowers.jpg \
  --input_layer=input \
  --output_layer=InceptionV3/Predictions/Reshape_1 \
  --graph=/tmp/frozen_inception_v3.pb \
  --labels=/tmp/imagenet_slim_labels.txt \
  --input_mean=0 \
  --input_std=255
```


# Troubleshooting
<a id='Troubleshooting'></a>

#### The model runs out of CPU memory.

See
[Model Runs out of CPU memory](https://github.com/tensorflow/models/tree/master/research/inception#the-model-runs-out-of-cpu-memory).

#### The model runs out of GPU memory.

See
[Adjusting Memory Demands](https://github.com/tensorflow/models/tree/master/research/inception#adjusting-memory-demands).

#### The model training results in NaN's.

See
[Model Resulting in NaNs](https://github.com/tensorflow/models/tree/master/research/inception#the-model-training-results-in-nans).

#### The ResNet and VGG Models have 1000 classes but the ImageNet dataset has 1001

ResNet和VGG模型有1000个类，但是ImageNet数据集有1001个。

The ImageNet dataset provided has an empty background class which can be used
to fine-tune the model to other tasks. If you try training or fine-tuning the
VGG or ResNet models using the ImageNet dataset, you might encounter the
following error:

所提供的ImageNet数据集有一个可用的空背景类。将模型微调到其他任务。如果你尝试训练或微调。使用ImageNet数据集的VGG或ResNet模型，您可能会遇到。以下错误:

```bash
InvalidArgumentError: Assign requires shapes of both tensors to match. lhs shape= [1001] rhs shape= [1000]
```
This is due to the fact that the VGG and ResNet V1 final layers have only 1000
outputs rather than 1001.

这是由于VGG和ResNet V1最终层只有1000个。输出而不是1001。

To fix this issue, you can set the `--labels_offset=1` flag. This results in
the ImageNet labels being shifted down by one:

要解决这个问题，可以设置“-labels_offset=1”标记。这将导致 ImageNet标签被一个:

#### I wish to train a model with a different image size.

我想训练一个不同图像大小的模型。

The preprocessing functions all take `height` and `width` as parameters. You
can change the default values using the following snippet:

预处理函数都以“高度”和“宽度”作为参数。你可以使用以下代码段更改默认值:

```python
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    height=MY_NEW_HEIGHT,
    width=MY_NEW_WIDTH,
    is_training=True)
```

#### What hardware specification are these hyper-parameters targeted for?

See
[Hardware Specifications](https://github.com/tensorflow/models/tree/master/research/inception#what-hardware-specification-are-these-hyper-parameters-targeted-for).
