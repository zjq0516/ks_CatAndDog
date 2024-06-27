#基于VGG16构建模型
import numpy as np
import matplotlib.pyplot as plt
from keras import models,layers
from keras.src.applications.vgg16 import VGG16
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow import optimizers
from tensorflow.keras.optimizers import Adam

#1.训练样本的目录
train_dir = './train_dataset/train/train'

#2.验证样本的目录
validation_dir = './train_dataset/train/validation'

#3.测试样本的目录
test_dir = './train_dataset/train/test'

#4.训练集生成器--训练集数据加强
train_datagen = ImageDataGenerator(
    rescale = 1./255,#像素值缩放
    rotation_range = 40,#旋转图片
    width_shift_range = 0.2,#水平平移图片
    height_shift_range = 0.2,#垂直平移图片
    shear_range = 0.2,#剪切图片
    zoom_range = 0.2,#缩放图片
    horizontal_flip = True,#翻转图片
    fill_mode = 'nearest'

)
train_generator = train_datagen.flow_from_directory(
    directory = train_dir,
    target_size = (224,224),#图片大小为244*244
    class_mode = 'binary',
    batch_size = 10#每次扔进神经网络训练的数据为20个
)

#5.验证样本生成器，进行像素值缩放操作
validation_datagen = ImageDataGenerator(
    rescale = 1./255
)
validation_generator = validation_datagen.flow_from_directory(
    directory = validation_dir,
    target_size = (224,224),
    class_mode = 'binary',
    batch_size = 10
)

#6.测试样本生成器
test_datagen = ImageDataGenerator(
    rescale = 1./255
)
test_generator = test_datagen.flow_from_directory(
    directory = test_dir,
    target_size = (224,224),
    class_mode = 'binary',
    batch_size = 10
)
print(train_generator.class_indices)
print(validation_generator.class_indices)
print(test_generator.class_indices)

#7.实例化VGG16模型
conv_base = VGG16(
    weights = 'imagenet',#使用imagenet数据集训练
    include_top = False,    #不包含模型顶部的全连接层，以适应二分类任务
    input_shape = (224,224,3)
)

#8.冻结卷积基-保证其权重在训练过程中不变-不训练这个，参数过多
conv_base.trainable = False

#9.构建网络模型-基于VGG16建立模型
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten(input_shape = conv_base.output_shape[1:])) #Flatten层将特征图展平，图片输出四维，1代表数量
model.add(layers.Dense(256,activation = 'relu'))#具有256个神经元的Dense层
model.add(layers.Dropout(0.5))#Dropout层防止过拟合
model.add(layers.Dense(1,activation = 'sigmoid'))   #二分类

# 10.定义优化器、代价函数、训练过程中计算准确率
optimizer = Adam(learning_rate=0.0005/10)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['acc']
)
model.summary()

# 11.拟合模型
history = model.fit(
    train_generator,
    steps_per_epoch=50,
    epochs=10,  # 20,把整个数据集丢进神经网络训练20次
    validation_data=validation_generator,
    validation_steps=10
)


#12.保存模型
model.save('./model/data/model14_2_VGG 16_cats_vs_dogs_1.h5')

#13.评估测试集、训练集、验证集的准确率
test_eval = model.evaluate(test_generator)
print("测试集准确率：", test_eval)
train_eval = model.evaluate(train_generator)
print("训练集准确率：", train_eval)
val_eval = model.evaluate(validation_generator)
print("验证集准确率：", val_eval)
#14.绘制训练过程中的损失曲线和精度曲线
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc) + 1)
plt.plot(epochs,acc,'bo')
plt.plot(epochs,acc,'b',label = 'Training acc')
plt.plot(epochs,val_acc,'ro')
plt.plot(epochs,val_acc,'r',label = 'Validation acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training and Validation accuracy')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs,loss,'bo')
plt.plot(epochs,loss,'b',label = 'Training Loss')
plt.plot(epochs,val_loss,'ro')
plt.plot(epochs,val_loss,'r',label = 'Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss and Validation Loss")
plt.legend()
plt.show()
