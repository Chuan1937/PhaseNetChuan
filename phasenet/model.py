import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ModelConfig:

  batch_size = 20
  depths = 5
  filters_root = 8
  kernel_size = [7, 1]
  pool_size = [4, 1]
  dilation_rate = [1, 1]
  class_weights = [1.0, 1.0, 1.0]
  loss_type = "cross_entropy"
  weight_decay = 0.0
  optimizer = "adam"
  momentum = 0.9
  learning_rate = 0.001
  decay_step = 1e9
  decay_rate = 0.9
  drop_rate = 0.0
  summary = True
  
  X_shape = [3000, 1, 3]
  n_channel = X_shape[-1]
  Y_shape = [3000, 1, 3]
  n_class = Y_shape[-1]

  def __init__(self, **kwargs):
    for k,v in kwargs.items():
      setattr(self, k, v)

  def update_args(self, args):
    for k,v in vars(args).items():
      setattr(self, k, v)


def crop_and_concat(net1, net2):
  """
  the size(net1) <= size(net2)
  """
  ## dynamic shape
  chn1 = net1.get_shape().as_list()[-1]
  chn2 = net2.get_shape().as_list()[-1]
  net1_shape = tf.shape(net1)
  net2_shape = tf.shape(net2)
  # print(net1_shape)
  # print(net2_shape)
  # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
  size = [-1, net1_shape[1], net1_shape[2], -1]
  net2_resize = tf.slice(net2, offsets, size)

  out = tf.concat([net1, net2_resize], 3)
  out.set_shape([None, None, None, chn1+chn2])

  return out 

  # else:
  #     offsets = [0, (net1_shape[1] - net2_shape[1]) // 2, (net1_shape[2] - net2_shape[2]) // 2, 0]
  #     size = [-1, net2_shape[1], net2_shape[2], -1]
  #     net1_resize = tf.slice(net1, offsets, size)
  #     return tf.concat([net1_resize, net2], 3)


def crop_only(net1, net2):
  """
  the size(net1) <= size(net2)
  """
  net1_shape = net1.get_shape().as_list()
  net2_shape = net2.get_shape().as_list()
  # print(net1_shape)
  # print(net2_shape)
  # if net2_shape[1] >= net1_shape[1] and net2_shape[2] >= net1_shape[2]:
  offsets = [0, (net2_shape[1] - net1_shape[1]) // 2, (net2_shape[2] - net1_shape[2]) // 2, 0]
  size = [-1, net1_shape[1], net1_shape[2], -1]
  net2_resize = tf.slice(net2, offsets, size)
  #return tf.concat([net1, net2_resize], 3)
  return net2_resize

class UNet:
  def __init__(self, config=ModelConfig(), input_batch=None, mode='train'):
    self.depths = config.depths
    self.filters_root = config.filters_root
    self.kernel_size = config.kernel_size
    self.dilation_rate = config.dilation_rate
    self.pool_size = config.pool_size
    self.X_shape = config.X_shape
    self.Y_shape = config.Y_shape
    self.n_channel = config.n_channel
    self.n_class = config.n_class
    self.class_weights = config.class_weights
    self.batch_size = config.batch_size
    self.loss_type = config.loss_type
    self.weight_decay = config.weight_decay
    self.optimizer = config.optimizer
    self.learning_rate = config.learning_rate
    self.decay_step = config.decay_step
    self.decay_rate = config.decay_rate
    self.momentum = config.momentum
    self.global_step = tf.compat.v1.get_variable(name="global_step", initializer=0, dtype=tf.int32)
    self.summary_train = []
    self.summary_valid = []

    self.build(input_batch, mode=mode)

  def add_placeholders(self, input_batch=None, mode="train"):
    if input_batch is None:
      # self.X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.X_shape[-3], self.X_shape[-2], self.X_shape[-1]], name='X')
      # self.Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.Y_shape[-3], self.Y_shape[-2], self.n_class], name='y')
      self.X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, None, self.X_shape[-1]], name='X')
      self.Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, None, None, self.n_class], name='y')
    else:
      self.X = input_batch[0]
      if mode in ["train", "valid", "test"]:
        self.Y = input_batch[1]
      self.input_batch = input_batch

    self.is_training = tf.compat.v1.placeholder(dtype=tf.bool, name="is_training")
    # self.keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, name="keep_prob")
    self.drop_rate = tf.compat.v1.placeholder(dtype=tf.float32, name="drop_rate")



  def residual_block(self,input_tensor, filters, kernel_size=(7, 1), stride=1, dilation_rate=(1, 1), use_batchnorm=True, name="res_block"):
    """
    残差块的实现
    :param input_tensor: 输入的张量
    :param filters: 卷积核的数量
    :param kernel_size: 卷积核的大小
    :param stride: 步长
    :param dilation_rate: 空洞卷积率
    :param use_batchnorm: 是否使用批量归一化
    :param name: 残差块的名字
    :return: 残差块的输出
    """
    x = input_tensor
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, 
                               padding='same', dilation_rate=dilation_rate, 
                               kernel_initializer='he_normal', use_bias=False, name=f'{name}_conv2')(x)

    # 使用 1x1 卷积调整输入张量的形状，以匹配输出张量的通道数
    if input_tensor.shape[-1] != filters:
        input_tensor = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), 
                                              strides=stride, padding='same', 
                                              kernel_initializer='he_normal', use_bias=False, 
                                              name=f'{name}_conv_adjust')(input_tensor)

    # 跳跃连接
    x = tf.keras.layers.Add(name=f'{name}_add')([x, input_tensor])

    x = tf.keras.layers.ReLU(name=f'{name}_out_relu')(x)

    return x



  def add_prediction_op(self):
    logging.info("Model: depths {depths}, filters {filters}, "
                 "filter size {kernel_size[0]}x{kernel_size[1]}, "
                 "pool size: {pool_size[0]}x{pool_size[1]}, "
                 "dilation rate: {dilation_rate[0]}x{dilation_rate[1]}".format(
                  depths=self.depths,
                  filters=self.filters_root,
                  kernel_size=self.kernel_size,
                  dilation_rate=self.dilation_rate,
                  pool_size=self.pool_size))

    # 如果 weight_decay 大于 0，设置 L2 正则化以防止过拟合
    self.regularizer = tf.keras.regularizers.l2(0.5 * self.weight_decay) if self.weight_decay > 0 else None

    # 初始化器
    self.initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")

    # 输入层（例如输入大小是 3x3001）
    convs = [None] * self.depths  # 存储每个深度的输出

    with tf.name_scope("Input"):
        net = self.X  # 输入张量

        net = tf.keras.layers.Conv2D(
            filters=self.filters_root, kernel_size=self.kernel_size, padding='same',
            dilation_rate=self.dilation_rate, kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer, name="input_conv")(net)
        net = tf.keras.layers.BatchNormalization(name="input_bn")(net, training=self.is_training)
        net = tf.keras.layers.ReLU(name="input_relu")(net)
        net = tf.keras.layers.Dropout(rate=self.drop_rate)(net, training=self.is_training)

    # 下采样部分
    for depth in range(0, self.depths):
        with tf.name_scope(f"DownConv_{depth}"):
            filters = int(2**depth * self.filters_root)  # 滤波器个数
            net = tf.keras.layers.Conv2D(filters=filters, 
                                         kernel_size=self.kernel_size, 
                                         padding='same', 
                                         dilation_rate=self.dilation_rate, 
                                         use_bias=False, 
                                         kernel_initializer=self.initializer, 
                                         kernel_regularizer=self.regularizer, 
                                         name=f"down_conv_{depth}_1")(net)
            net = tf.keras.layers.BatchNormalization(name=f"down_bn_{depth}_1")(net, training=self.is_training)
            net = tf.keras.layers.ReLU(name=f"down_relu_{depth}_1")(net)
            
            # 插入残差块
            net = self.residual_block(net, filters=filters, name=f"res_block_{depth}")

            net = tf.keras.layers.Dropout(rate=self.drop_rate, name=f"down_dropout_{depth}_1")(net, training=self.is_training)
            convs[depth] = net  # 存储跳跃连接块

            # 下采样卷积（调整尺寸）
            if depth < self.depths - 1:
                net = tf.keras.layers.Conv2D(filters=filters, 
                                             kernel_size=self.kernel_size, 
                                             strides=self.pool_size,  # Convolution+Stride
                                             padding='same', 
                                             dilation_rate=self.dilation_rate, 
                                             use_bias=False, 
                                             kernel_initializer=self.initializer, 
                                             kernel_regularizer=self.regularizer, 
                                             name=f"down_conv_{depth}_2")(net)
                net = tf.keras.layers.BatchNormalization(name=f"down_bn_{depth}_2")(net, training=self.is_training)
                net = tf.keras.layers.ReLU(name=f"down_relu_{depth}_2")(net)
                net = tf.keras.layers.Dropout(rate=self.drop_rate, name=f"down_dropout_{depth}_2")(net, training=self.is_training)

    # 上采样部分
    for depth in range(self.depths - 2, -1, -1):
        with tf.name_scope(f"UpConv_{depth}"):
            filters = int(2**depth * self.filters_root)  # 滤波器个数
            net = tf.keras.layers.Conv2DTranspose(filters=filters, 
                                                  kernel_size=self.kernel_size, 
                                                  strides=self.pool_size, 
                                                  padding="same", 
                                                  use_bias=False, 
                                                  kernel_initializer=self.initializer, 
                                                  kernel_regularizer=self.regularizer, 
                                                  name=f"up_conv0_{depth}")(net)
            net = tf.keras.layers.BatchNormalization(name=f"up_bn0_{depth}")(net, training=self.is_training)
            net = tf.keras.layers.ReLU(name=f"up_relu0_{depth}")(net)
            net = tf.keras.layers.Dropout(rate=self.drop_rate, name=f"up_dropout0_{depth}")(net, training=self.is_training)

            # 跳跃连接
            net = crop_and_concat(convs[depth], net)

            # 插入残差块
            net = self.residual_block(net, filters=filters, name=f"res_block_up_{depth}")

            # 卷积操作
            net = tf.keras.layers.Conv2D(filters=filters, 
                                         kernel_size=self.kernel_size, 
                                         padding='same', 
                                         use_bias=False, 
                                         dilation_rate=self.dilation_rate, 
                                         kernel_initializer=self.initializer, 
                                         kernel_regularizer=self.regularizer, 
                                         name=f"up_conv1_{depth}")(net)
            net = tf.keras.layers.BatchNormalization(name=f"up_bn1_{depth}")(net, training=self.is_training)
            net = tf.keras.layers.ReLU(name=f"up_relu1_{depth}")(net)
            net = tf.keras.layers.Dropout(rate=self.drop_rate, name=f"up_dropout1_{depth}")(net, training=self.is_training)

    # 输出层
    with tf.name_scope("Output"):
        net = tf.keras.layers.Conv2D(filters=self.n_class,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     activation=None,
                                     kernel_initializer=self.initializer,
                                     kernel_regularizer=self.regularizer,
                                     name="output_conv")(net)

    self.logits = net

    # 计算预测结果，使用 softmax 转换为概率分布
    with tf.name_scope("preds"):
        self.preds = tf.nn.softmax(self.logits)

    return self.preds


  # def conv_block(self, input_tensor, filters, n=2, name='conv'):
  #   x = input_tensor
  #   for i in range(n):
  #       x = tf.keras.layers.Conv2D(
  #           filters=filters,
  #           kernel_size=(3, 1),
  #           padding='same',
  #           use_bias=False,
  #           kernel_initializer=self.initializer,
  #           kernel_regularizer=self.regularizer,
  #           name=f"{name}_conv_{i}"
  #       )(x)
  #       x = tf.keras.layers.BatchNormalization(name=f"{name}_bn_{i}")(x, training=self.is_training)
  #       x = tf.keras.layers.ReLU(name=f"{name}_relu_{i}")(x)

  #   # **Residual Connection**
  #   # Check if the input and output channels match
  #   if input_tensor.shape[-1] != filters:
  #       # Adjust the input tensor with a 1x1 convolution
  #       shortcut = tf.keras.layers.Conv2D(
  #           filters=filters,
  #           kernel_size=(1, 1),
  #           padding='same',
  #           use_bias=False,
  #           kernel_initializer=self.initializer,
  #           kernel_regularizer=self.regularizer,
  #           name=f"{name}_shortcut_conv"
  #       )(input_tensor)
  #       shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut, training=self.is_training)
  #   else:
  #       shortcut = input_tensor

  #   x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])
  #   x = tf.keras.layers.ReLU(name=f"{name}_out_relu")(x)
  #   return x



  # def add_prediction_op(self):
  #   logging.info(
  #       "Model: depths {depths}, filters {filters}, "
  #       "filter size {kernel_size[0]}x{kernel_size[1]}, "
  #       "pool size: {pool_size[0]}x{pool_size[1]}, "
  #       "dilation rate: {dilation_rate[0]}x{dilation_rate[1]}".format(
  #           depths=self.depths,
  #           filters=self.filters_root,
  #           kernel_size=self.kernel_size,
  #           dilation_rate=self.dilation_rate,
  #           pool_size=self.pool_size
  #       )
  #   )

  #   # 正则化和初始化器
  #   if self.weight_decay > 0:
  #       weight_decay = tf.constant(self.weight_decay, dtype=tf.float32, name="weight_constant")
  #       self.regularizer = tf.keras.regularizers.l2(l=0.5 * weight_decay)
  #   else:
  #       self.regularizer = None

  #   self.initializer = tf.keras.initializers.VarianceScaling(
  #       scale=1.0, mode="fan_avg", distribution="uniform"
  #   )

  #   filters_root = [8, 16, 32, 64, 128]

  #   # 输入层
  #   with tf.name_scope("Input"):
  #       net = self.X  # 输入张量

  #   """ 编码器部分 """

  #   # block 1
  #   e1 = self.conv_block(net, filters_root[0], name='conv_block_e1')

  #   # block 2
  #   e2 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(e1)
  #   e2 = self.conv_block(e2, filters_root[1], name='conv_block_e2')

  #   # block 3
  #   e3 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(e2)
  #   e3 = self.conv_block(e3, filters_root[2], name='conv_block_e3')

  #   # block 4
  #   e4 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(e3)
  #   e4 = self.conv_block(e4, filters_root[3], name='conv_block_e4')

  #   # block 5 (Bottleneck)
  #   e5 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(e4)
  #   e5 = self.conv_block(e5, filters_root[4], name='conv_block_e5')

  #   """ 解码器部分 """

  #   cat_channels = filters_root[0]  # 8
  #   cat_blocks = len(filters_root)  # 5
  #   upsample_channels = cat_blocks * cat_channels  # 40

  #   # Decoder Block d4
  #   e1_d4 = tf.keras.layers.MaxPool2D(pool_size=(8, 1), padding='same')(e1)
  #   e1_d4 = self.conv_block(e1_d4, cat_channels, n=1, name='conv_block_e1_d4')

  #   e2_d4 = tf.keras.layers.MaxPool2D(pool_size=(4, 1), padding='same')(e2)
  #   e2_d4 = self.conv_block(e2_d4, cat_channels, n=1, name='conv_block_e2_d4')

  #   e3_d4 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(e3)
  #   e3_d4 = self.conv_block(e3_d4, cat_channels, n=1, name='conv_block_e3_d4')

  #   e4_d4 = self.conv_block(e4, cat_channels, n=1, name='conv_block_e4_d4')

  #   e5_d4 = tf.keras.layers.UpSampling2D(size=(2, 1), interpolation='bilinear')(e5)
  #   e5_d4 = self.conv_block(e5_d4, cat_channels, n=1, name='conv_block_e5_d4')

  #   # 将e5_d4的尺寸从 (16, 376, 1, 8) 裁剪到 (16, 375, 1, 8)
  #   e5_d4 = tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(e5_d4)  # 裁剪掉行维度的最后一个单位


  #   d4 = tf.keras.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
  #   d4 = self.conv_block(d4, upsample_channels, n=1, name='conv_block_d4')

  #   # Decoder Block d3
  #   e1_d3 = tf.keras.layers.MaxPool2D(pool_size=(4, 1), padding='same')(e1)
  #   e1_d3 = self.conv_block(e1_d3, cat_channels, n=1, name='conv_block_e1_d3')

  #   e2_d3 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(e2)
  #   e2_d3 = self.conv_block(e2_d3, cat_channels, n=1, name='conv_block_e2_d3')

  #   e3_d3 = self.conv_block(e3, cat_channels, n=1, name='conv_block_e3_d3')

  #   e4_d3 = tf.keras.layers.UpSampling2D(size=(2, 1), interpolation='bilinear')(d4)
  #   e4_d3 = self.conv_block(e4_d3, cat_channels, n=1, name='conv_block_d4_d3')

  #   e5_d3 = tf.keras.layers.UpSampling2D(size=(4, 1), interpolation='bilinear')(e5)
  #   e5_d3 = self.conv_block(e5_d3, cat_channels, n=1, name='conv_block_e5_d3')
  #   e5_d3 = tf.keras.layers.Cropping2D(cropping=((2, 0), (0, 0)))(e5_d3)  # 将形状从 (16, 752, 1, 8) 裁剪为 (16, 750, 1, 8)

  # # 然后进行拼接

  #   d3 = tf.keras.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
  #   d3 = self.conv_block(d3, upsample_channels, n=1, name='conv_block_d3')

  #   # Decoder Block d2
  #   e1_d2 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), padding='same')(e1)
  #   e1_d2 = self.conv_block(e1_d2, cat_channels, n=1, name='conv_block_e1_d2')

  #   e2_d2 = self.conv_block(e2, cat_channels, n=1, name='conv_block_e2_d2')

  #   d3_d2 = tf.keras.layers.UpSampling2D(size=(2, 1), interpolation='bilinear')(d3)
  #   d3_d2 = self.conv_block(d3_d2, cat_channels, n=1, name='conv_block_d3_d2')

  #   d4_d2 = tf.keras.layers.UpSampling2D(size=(4, 1), interpolation='bilinear')(d4)
  #   d4_d2 = self.conv_block(d4_d2, cat_channels, n=1, name='conv_block_d4_d2')

  #   e5_d2 = tf.keras.layers.UpSampling2D(size=(8, 1), interpolation='bilinear')(e5)
  #   e5_d2 = self.conv_block(e5_d2, cat_channels, n=1, name='conv_block_e5_d2')
  #   e5_d2= tf.keras.layers.Cropping2D(cropping=((4, 0), (0, 0)))(e5_d2)  # 将形状从 (16, 1504, 1, 8) 裁剪为 (16, 1500, 1, 8)

  #   d2 = tf.keras.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
  #   d2 = self.conv_block(d2, upsample_channels, n=1, name='conv_block_d2')

  #   # Decoder Block d1
  #   e1_d1 = self.conv_block(e1, cat_channels, n=1, name='conv_block_e1_d1')

  #   d2_d1 = tf.keras.layers.UpSampling2D(size=(2, 1), interpolation='bilinear')(d2)
  #   d2_d1 = self.conv_block(d2_d1, cat_channels, n=1, name='conv_block_d2_d1')

  #   d3_d1 = tf.keras.layers.UpSampling2D(size=(4, 1), interpolation='bilinear')(d3)
  #   d3_d1 = self.conv_block(d3_d1, cat_channels, n=1, name='conv_block_d3_d1')

  #   d4_d1 = tf.keras.layers.UpSampling2D(size=(8, 1), interpolation='bilinear')(d4)
  #   d4_d1 = self.conv_block(d4_d1, cat_channels, n=1, name='conv_block_d4_d1')

  #   e5_d1 = tf.keras.layers.UpSampling2D(size=(16, 1), interpolation='bilinear')(e5)
  #   e5_d1 = self.conv_block(e5_d1, cat_channels, n=1, name='conv_block_e5_d1')
  #   e5_d1 = tf.keras.layers.Cropping2D(cropping=((8, 0), (0, 0)))(e5_d1)  # 将形状从 (16, 3008, 1, 8) 裁剪为 (16, 3000, 1, 8)

  #   d1 = tf.keras.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1])
  #   d1 = self.conv_block(d1, upsample_channels, n=1, name='conv_block_d1')

  #   # 最后一层没有归一化和ReLU
  #   d = tf.keras.layers.Conv2D(
  #       self.n_class, kernel_size=(3, 1), padding='same', activation=None, name='output_conv'
  #   )(d1)
  #   output = tf.keras.activations.softmax(d)

  #   # 保存logits
  #   self.logits = output

  #   # 计算最终预测结果，使用 softmax 将 logits 转换为概率分布
  #   self.preds = self.logits  # 直接输出

  #   return self.preds





  def add_loss_op(self):
    if self.loss_type == "cross_entropy":
      with tf.compat.v1.variable_scope("cross_entropy"):
        flat_logits = tf.reshape(self.logits, [-1, self.n_class], name="logits")
        flat_labels = tf.reshape(self.Y, [-1, self.n_class], name="labels")
        if (np.array(self.class_weights) != 1).any():
          class_weights = tf.constant(np.array(self.class_weights, dtype=np.float32), name="class_weights")
          weight_map = tf.multiply(flat_labels, class_weights)
          weight_map = tf.reduce_sum(input_tensor=weight_map, axis=1)
          loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                     labels=flat_labels)

          weighted_loss = tf.multiply(loss_map, weight_map)
          loss = tf.reduce_mean(input_tensor=weighted_loss)
        else:
          loss = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                         labels=flat_labels))

    elif self.loss_type == "IOU":
      with tf.compat.v1.variable_scope("IOU"):
        eps = 1e-7 
        loss = 0
        for i in range(1, self.n_class): 
          intersection = eps + tf.reduce_sum(input_tensor=self.preds[:,:,:,i] * self.Y[:,:,:,i], axis=[1,2])
          union = eps + tf.reduce_sum(input_tensor=self.preds[:,:,:,i], axis=[1,2]) + tf.reduce_sum(input_tensor=self.Y[:,:,:,i], axis=[1,2]) 
          loss += 1 - tf.reduce_mean(input_tensor=intersection / union)
    elif self.loss_type == "mean_squared":
      with tf.compat.v1.variable_scope("mean_squared"):
        flat_logits = tf.reshape(self.logits, [-1, self.n_class], name="logits")
        flat_labels = tf.reshape(self.Y, [-1, self.n_class], name="labels")
        with tf.compat.v1.variable_scope("mean_squared"):
          loss = tf.compat.v1.losses.mean_squared_error(labels=flat_labels, predictions=flat_logits) 
    else:
      raise ValueError("Unknown loss function: " % self.loss_type)

    tmp = tf.compat.v1.summary.scalar("train_loss", loss)
    self.summary_train.append(tmp)
    tmp = tf.compat.v1.summary.scalar("valid_loss", loss)
    self.summary_valid.append(tmp)

    if self.weight_decay > 0:
      with tf.compat.v1.name_scope('weight_loss'):
        tmp = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        weight_loss = tf.add_n(tmp, name="weight_loss")
      self.loss = loss + weight_loss 
    else:
      self.loss = loss 

  def add_training_op(self):
    if self.optimizer == "momentum":
      self.learning_rate_node = tf.compat.v1.train.exponential_decay(learning_rate=self.learning_rate,
                                 global_step=self.global_step,
                                 decay_steps=self.decay_step,
                                 decay_rate=self.decay_rate,
                                 staircase=True)
      optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate_node,
                           momentum=self.momentum)
    elif self.optimizer == "adam":
      self.learning_rate_node = tf.compat.v1.train.exponential_decay(learning_rate=self.learning_rate,
                                 global_step=self.global_step,
                                 decay_steps=self.decay_step,
                                 decay_rate=self.decay_rate,
                                 staircase=True)

      optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_node)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
    tmp = tf.compat.v1.summary.scalar("learning_rate", self.learning_rate_node)
    self.summary_train.append(tmp)

  def add_metrics_op(self):
    with tf.compat.v1.variable_scope("metrics"):

      Y= tf.argmax(input=self.Y, axis=-1)
      confusion_matrix = tf.cast(tf.math.confusion_matrix(
          labels=tf.reshape(Y, [-1]), 
          predictions=tf.reshape(self.preds, [-1]), 
          num_classes=self.n_class, name='confusion_matrix'),
          dtype=tf.float32)

      # with tf.variable_scope("P"):
      c = tf.constant(1e-7, dtype=tf.float32)
      precision_P =  (confusion_matrix[1,1] + c) / (tf.reduce_sum(input_tensor=confusion_matrix[:,1]) + c)
      recall_P = (confusion_matrix[1,1] + c) / (tf.reduce_sum(input_tensor=confusion_matrix[1,:]) + c)
      f1_P = 2 * precision_P * recall_P / (precision_P + recall_P)

      tmp1 = tf.compat.v1.summary.scalar("train_precision_p", precision_P)
      tmp2 = tf.compat.v1.summary.scalar("train_recall_p", recall_P)
      tmp3 = tf.compat.v1.summary.scalar("train_f1_p", f1_P)
      self.summary_train.extend([tmp1, tmp2, tmp3])

      tmp1 = tf.compat.v1.summary.scalar("valid_precision_p", precision_P)
      tmp2 = tf.compat.v1.summary.scalar("valid_recall_p", recall_P)
      tmp3 = tf.compat.v1.summary.scalar("valid_f1_p", f1_P)
      self.summary_valid.extend([tmp1, tmp2, tmp3])

      # with tf.variable_scope("S"):
      precision_S =  (confusion_matrix[2,2] + c) / (tf.reduce_sum(input_tensor=confusion_matrix[:,2]) + c)
      recall_S = (confusion_matrix[2,2] + c) / (tf.reduce_sum(input_tensor=confusion_matrix[2,:]) + c)
      f1_S = 2 * precision_S * recall_S / (precision_S + recall_S)

      tmp1 = tf.compat.v1.summary.scalar("train_precision_s", precision_S)
      tmp2 = tf.compat.v1.summary.scalar("train_recall_s", recall_S)
      tmp3 = tf.compat.v1.summary.scalar("train_f1_s", f1_S)
      self.summary_train.extend([tmp1, tmp2, tmp3])

      tmp1 = tf.compat.v1.summary.scalar("valid_precision_s", precision_S)
      tmp2 = tf.compat.v1.summary.scalar("valid_recall_s", recall_S)
      tmp3 = tf.compat.v1.summary.scalar("valid_f1_s", f1_S)
      self.summary_valid.extend([tmp1, tmp2, tmp3])
      
      self.precision = [precision_P, precision_S]
      self.recall = [recall_P, recall_S]
      self.f1 = [f1_P, f1_S]



  def train_on_batch(self, sess, inputs_batch, labels_batch, summary_writer, drop_rate=0.0):
    feed = {self.X: inputs_batch,
            self.Y: labels_batch,
            self.drop_rate: drop_rate,
            self.is_training: True}

    _, step_summary, step, loss = sess.run([self.train_op,
                                            self.summary_train,
                                            self.global_step,
                                            self.loss],
                                            feed_dict=feed)
    summary_writer.add_summary(step_summary, step)
    return loss

  def valid_on_batch(self, sess, inputs_batch, labels_batch, summary_writer):
    feed = {self.X: inputs_batch,
            self.Y: labels_batch,
            self.drop_rate: 0,
            self.is_training: False}
            
    step_summary, step, loss, preds = sess.run([self.summary_valid,
                                                self.global_step,
                                                self.loss,
                                                self.preds],
                                                feed_dict=feed)
    summary_writer.add_summary(step_summary, step)
    return loss, preds

  def test_on_batch(self, sess, summary_writer):
    feed = {self.drop_rate: 0,
            self.is_training: False}
    step_summary, step, loss, preds, \
    X_batch, Y_batch, fname_batch, \
    itp_batch, its_batch = sess.run([self.summary_valid,
                                     self.global_step,
                                     self.loss,
                                     self.preds,
                                     self.X,
                                     self.Y,
                                     self.input_batch[2],
                                     self.input_batch[3],
                                     self.input_batch[4]],
                                     feed_dict=feed)
    summary_writer.add_summary(step_summary, step)
    return loss, preds, X_batch, Y_batch, fname_batch, itp_batch, its_batch


  def build(self, input_batch=None, mode='train'):
    self.add_placeholders(input_batch, mode)
    self.add_prediction_op()
    if mode in ["train", "valid", "test"]:
      self.add_loss_op()
      self.add_training_op()
      # self.add_metrics_op()
      self.summary_train = tf.compat.v1.summary.merge(self.summary_train)
      self.summary_valid = tf.compat.v1.summary.merge(self.summary_valid)
    return 0