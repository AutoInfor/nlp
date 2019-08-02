#!/usr/bin/env python
# coding: utf-8

# # 使用Keras LSTM模型进行IMDb情感分析,使用LSTM(32)建立32个神经元的LSTM层

# # 数据预处理
# 导入所需模块

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import pandas as pd
from get_data import get_data,corpus2token

# # 读取数据    
#读取数据
train_text_token,train_result_label_encoder,train_df,test_text_token,test_result_label_encoder,test_df,label_encoder=get_data()

#将result转换为One—Hot
train_result_OnHot = np_utils.to_categorical(train_result_label_encoder)
test_result_OneHot = np_utils.to_categorical(test_result_label_encoder,num_classes=len(train_result_OnHot[0]))

#将text转换成数字列表
num_words=5000
maxlen=50
token = Tokenizer(num_words=num_words)
token.fit_on_texts(train_text_token)
print('\ntoken字典',list(token.word_index.keys())[:200])
train_text_seq = token.texts_to_sequences(train_text_token)
test_text_seq = token.texts_to_sequences(test_text_token)
train_text_pad_seq = sequence.pad_sequences(train_text_seq, maxlen=maxlen)
test_text_pad_seq = sequence.pad_sequences(test_text_seq, maxlen=maxlen)
print('\n截长补短[:3]',train_text_pad_seq[:3])


# # 使用Keras LSTM模型进行分析
#导入所需模块
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.layers import Conv1D, MaxPooling1D
import copy
import time
import xlsxwriter
#建立模型
model = Sequential()

# 加入嵌入层
# output_dim = 32 输出维数是32，我们希望将数字列表转换成32维的向量
# input_dim = 4000 输入维数是4000，因为之前建立的字典有4000个单词
# input_length = 100 因为数字列表每一项有100个数字
model.add(Embedding(output_dim=32, input_dim= num_words, input_length=maxlen))

def create_model(model_layer,model,train_text_pad_seq, train_result_OnHot,test_text_pad_seq, test_result_OneHot,dropout,loss,activation,batch_size):
    
    model.add(Dropout(dropout))
    if model_layer=='MLP': 
        # # 建立MLP模型
        # 加入平坦层，平坦层有3200个神经元
        model.add(Flatten())
        
    if model_layer=='RNN': 
        # # 建立RNN模型
        # 加入RNN层
        model.add(SimpleRNN(units=16))
        
    if model_layer=='CNN': 
        # # 建立CNN模型
        #建立卷积层1
        model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout))
        #建立卷积层2
        model.add(Conv1D(filters=36, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout))
        #建立平坦层
        model.add(Flatten())
    
    if model_layer=='LSTM': 
        # # 建立LSTM模型
        # 加入LSTM层
        model.add(LSTM(32))
    
    # 加入隐藏层
    model.add(Dense(units=256, activation='relu'))
    
    model.add(Dense(units=128, activation='relu'))
        
    model.add(Dropout(dropout))

    # 加入输出层
    model.add(Dense(units=len(train_result_OnHot[0]), activation=activation))#多分类

    # 查看模型摘要
    #print(model.summary())

    # 定义训练方式
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])#多分类
        
    # 开始训练一共执行了10个训练周期，误差越来越小，准确率越来越高
    train_history = model.fit(train_text_pad_seq, train_result_OnHot, batch_size=batch_size, epochs=10, verbose=0, validation_split=0.2)

    # # 评估模型准确率
    scores = model.evaluate(test_text_pad_seq, test_result_OneHot, verbose=0)
    print('测试数据的模型准确率'+' model_layer:'+model_layer+' dropout:'+str(dropout)+' loss:'+loss+' scores:'+str(scores[1]))
    
    score_list.append((train_history.history['val_acc'][-1],model_layer+"训练数据",dropout,loss,activation,batch_size,model,train_history))

    score_list.append((scores[1],model_layer+"测试数据",dropout,loss,activation,batch_size,model,train_history))


score_list=[]

parameter_dict={'model_layer':['LSTM','MLP','RNN','CNN'],'dropout':[0.1,0.2,0.3],'batch_size':[200,500],'activation':['softmax'],'loss':['binary_crossentropy','categorical_crossentropy',]}

for model_layer in parameter_dict['model_layer']:
    for dropout in parameter_dict['dropout']:
        for batch_size in parameter_dict['batch_size']:
            for activation in parameter_dict['activation']:
                for loss in parameter_dict['loss'][1:]:
                    model_=copy.deepcopy(model)
                    create_model(model_layer,model_,train_text_pad_seq, train_result_OnHot,test_text_pad_seq, test_result_OneHot,dropout,loss,activation,batch_size)

#显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr',False)
pd.set_option('display.max_colwidth', 30)
df = pd.DataFrame(score_list, columns=['scores','model_layer','dropout','loss','activation','batch_size','model','train_history'])
print(df.sort_values(by='scores',ascending=False).iloc[:, 0:6])

scores,model,train_history=df.sort_values(by='scores',ascending=False).iloc[0,[0,-2,-1]]
print('最高得分',scores)

# # 进行预测
predict = model.predict_classes(test_text_pad_seq)
# 使用一维数组查看预测结果（使用reshape把二维数组predict转换成一维）
predict_classes = predict.reshape(-1)

# # 显示混淆矩阵
import pandas as pd
print("\n混淆矩阵\n",pd.crosstab(test_result_label_encoder, predict, rownames = ['label'], colnames = ['predict']))

# # 查看test_text具体测试结果
def displatest_result_Sentiment():
    test_df['预测结果'] = label_encoder.inverse_transform(predict)
    print('打印test_df预测结果\n',test_df)
    test_df.to_excel("output_"+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+".xlsx",engine='xlsxwriter')


displatest_result_Sentiment()


## 输入文字input_text，就可以输出预测结果
def predict_review(input_text):
    print('\ninput_text',input_text)
    input_token = corpus2token([input_text])
    input_seq = token.texts_to_sequences(input_token)
    pad_input_seq = sequence.pad_sequences(input_seq, maxlen=maxlen)
    predict_result = model.predict_classes(pad_input_seq)#二维数组
    print('预测结果',label_encoder.inverse_transform(predict_result.tolist()))

input_text = '''拨号盘异常自动变成单手操作界面，且界面显示不全，见截图'''
predict_review(input_text)


## 画'acc','val_acc' loss','val_loss'曲线图
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')


# # 保存模型serialize model to JSON

model_json = model.to_json()
with open("SaveModel/Imdb_LSTM_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("SaveModel/Imdb_LSTM_model.h5")
print("Saved model to disk")
