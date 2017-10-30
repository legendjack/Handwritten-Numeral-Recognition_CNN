## 基于Tensorflow，OpenCV
## 使用MNIST数据集训练卷积神经网络模型，用于手写数字识别

### [ocr.py](https://github.com/legendjack/-_CNN_MNIST/blob/master/ocr.py)
一个单层的神经网络，使用MNIST训练，识别准确率较低

### [cnn_ocr.py](https://github.com/legendjack/-_CNN_MNIST/blob/master/cnn_ocr.py)
两层的卷积神经网络，使用MNIST训练（模型使用MNIST测试集准确率高于99%），识别准确率较高；
但是如果写的较为随意，还是会出现分类错误的情况，可能是图像预处理的问题

### [cnn_ocr_2.py](https://github.com/legendjack/-_CNN_MNIST/blob/master/cnn_ocr_2.py)
直接从cnn_mnist.ckpt.meta文件中加载已经持久化的图（graph），
需要在训练的时候为tensor指定名称（[cnn_mnist_train.py](https://github.com/legendjack/-_CNN_MNIST/blob/master/cnn_mnist_train.py) line82）: 
```
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
```
The name 'x_input' refers to an Operation, not a Tensor.
Tensor names must be of the form "<op_name>:<output_index>".
[cnn_ocr_2.py](https://github.com/legendjack/-_CNN_MNIST/blob/master/cnn_ocr_2.py) line33
```
keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
```

### [cnn_mnist_train.py](https://github.com/legendjack/-_CNN_MNIST/blob/master/cnn_mnist_train.py)
训练模型的程序

### [模型文件](https://github.com/legendjack/Handwritten-Numeral-Recognition_CNN/tree/master/model/cnn_mnist_model)
- *checkpoint*是一个文本文件，保存了一个目录下所有的模型文件列表，这个文件是tf.train.Saver类自动生成且自动维护的。
在checkpoint文件中维护了由一个tf.train.Saver类持久化的所有TensorFlow模型文件的文件名。
当某个保存的TensorFlow模型文件被删除时，这个模型所对应的文件名也会从checkpoint文件中删除。
checkpoint中内容的格式为CheckpointState Protocol Buffer。

- *cnn_mnist.ckpt.meta*文件保存了TensorFlow计算图的结构，可以理解为神经网络的网络结构
TensorFlow通过元图（MetaGraph）来记录计算图中节点的信息以及运行计算图中节点所需要的元数据。
TensorFlow中元图是由MetaGraphDef Protocol Buffer定义的。
MetaGraphDef中的内容构成了TensorFlow持久化时的第一个文件。
保存MetaGraphDef信息的文件默认以.meta为后缀名，文件model.ckpt.meta中存储的就是元图数据。

- *cnn_mnist.ckpt.index*文件保存了TensorFlow程序中每一个变量的取值，这个文件是通过SSTable格式存储的，可以大致理解为就是一个（key，value）列表。
model.ckpt文件中列表的第一行描述了文件的元信息，比如在这个文件中存储的变量列表。
列表剩下的每一行保存了一个变量的片段，变量片段的信息是通过SavedSlice Protocol Buffer定义的。
SavedSlice类型中保存了变量的名称、当前片段的信息以及变量取值。
TensorFlow提供了tf.train.NewCheckpointReader类来查看model.ckpt文件中保存的变量信息。
如何使用tf.train.NewCheckpointReader类这里不做说明，自查。

![image](https://github.com/legendjack/-_CNN_MNIST/blob/master/0.jpg?raw=true)
