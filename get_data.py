# 读取训练和测试数据，text为分词列表格式，result为label_encoder格式
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re #导入正则表达式模块
from keras.datasets import imdb
def rm_tags(text): #创建rm_tage函数，输入参数是text文字
    re_tag = re.compile(r'<[^>]+>') #创建re_tag为正则表达式变量，赋值为‘<[^>]+>’
    return re_tag.sub('', text) #使用re_tag将text文字中符合正则表达式条件的字符替换成空字符串
def prepare_datasets(corpus, labels, test_data_proportion=0.2):
    """
    使用sklearn中自带的train_test_split函数对数据集进行划分，得到训练集和测试集
    :param corpus: 总数据特征集
    :param labels: 总数据label
    :param test_data_proportion: 测试集所占比例
    :return: 划分结果
    """
    train_X, test_X, train_Y, test_Y = train_test_split(
        corpus, labels, test_size=test_data_proportion, random_state=42)
    return train_X, train_Y, test_X, test_Y
import os
def read_files(filetype):
    #创建read_files函数，输入参数为filetype。读取训练数据时传入‘train’，读取测试数据时传入‘test’
    path = 'aclImdb/'#设置文件的存取路径
    file_list = [] #创建文件列表
    
    positive_path = path + filetype + '/pos/' #设置正面评价的文件目录为positive_path
    for f in os.listdir(positive_path): #用for循环将positive_path目录下的所有文件加入file_list
        file_list += [positive_path + f]
        
    negative_path = path + filetype + '/neg/'#设置负面评价的文件目录为positive_path
    for f in os.listdir(negative_path):#用for循环将negative_path目录下的所有文件加入file_list
        file_list += [negative_path + f]
        
    print('read', filetype, 'files:', len(file_list))#显示读取的filetype目录下的文件个数
    
    all_labels = (["正面"] * 12500 + ["负面"] * 12500) 
    #产生all_labels,前12500项是正面，所以产生12500项1的列表，后12500项是负面，所以产生12500项0的列表
    
    all_texts = [] #设置all_texts为空列表
    
    '''    用fi读取file_list所有文件，使用打开文件为file_input，使用file_input.readlines()读取文件，
    用join连接所有文件内容，然后使用rm_tags删除tag，最后加入all_texrs list
    '''
    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(' '.join(file_input.readlines()))]
            
    return all_labels, all_texts
import numpy as np
np.set_printoptions(threshold=np.inf)    
def get_data_txt(filetype):
    spam_data = pd.read_csv('./data/'+filetype+'/spam_data.txt', header=None, sep='\n')
    ham_data = pd.read_csv('./data/'+filetype+'/ham_data.txt', header=None, sep='\n')
    ham_label = np.ones(len(ham_data))
    spam_label = np.zeros(len(spam_data)) 
    corpus_pd=pd.concat([ham_data,spam_data])    
    labels = np.append(ham_label,spam_label)
    corpus_pd['结果'] = labels
    print(corpus_pd.groupby(corpus_pd.iloc[:,1]).size().reset_index(name='Size').sort_values(by='Size'))
    return corpus_pd,corpus_pd.iloc[:,1].tolist(),corpus_pd.iloc[:,0].tolist()
    
import openpyxl
import jieba
import pandas as pd

def get_data_xlsx(xlsx):
    df=pd.read_excel(xlsx)
    print()
    print(df.groupby(df.iloc[:,1]).size().reset_index(name='Size').sort_values(by='Size'))
    return df,df.iloc[:,1].tolist(),df.iloc[:,0].tolist()

import jieba.posseg as pseg
import jieba
def corpus2token(corpus):
    #检查中文
    zhmodel = re.compile(u'[\u4e00-\u9fa5]')    
    contents = corpus[0]
    match = zhmodel.search(contents)
    if not match:
        return corpus

    stop_flag = ['x', 'c', 'u', 'p', 't', 'uj', 'm', 'f', 'r']
    stopwords = {}.fromkeys(['的','了','手机','包括','等','是','测试','SOD','分值','测试环境','测试步骤','预期','结果','实际','结果','对比','样机','情况'])
    focus_point='''容量 杜比 向导 重启 工程模式 系统 usb 管家 摄像 兼容性 网速 美颜 商店 导航 视频 像素 影响 死机 充电 按键 全屏 sd 过热 电源 预置 影音 人脸 通话记录 黑屏 显示屏 电池组 暗光 widget agps 待机 掌心 音频 指示灯 振动 数据 fota 语音识别 外观 数据业务 nfc 微信 触摸屏 底层 volte 解锁 定位 modem 芯片 音乐 外放 lcd 状态栏 网络 sim卡 开关机 显示 手感 工艺 闹钟 功耗 照片 voip 信号 monitor 蓝牙 发热 gms email 话筒 通话 卡 锁屏 音量 输入法 第三方 包装 ui 处理器 汇分享 ztemarket 音质 听歌 做工 字体 led 驾驶 耳机 和包 助手 结构 fm 重启 电脑 单手操作 设置 铃音 联系人 反应 注册 画质 ims 驱动 射频 充电器 断触 手势 文件 记事本 屏 电信 开关 音响 模组 摄像头 播放器 黑边 黄页 launcher 下载 触感 wifi 双卡 响铃 电话 局方 指南针 与 音效 触屏 电量 备份 声音 语音 分享 车载 移动 计算器 续航 电池 汇 内存 开机 扬声器 文件管理器 镜头 夜景 yellowpage 单手 掌心管家 gps 照相 信息 拍摄 桌面 马达 验证码 图库 屏幕 护眼 人脸识别 对焦 rcs 耗电 网页 秒表 刘海 震动 温控 效果 呼叫 流媒体 日历 指纹 录音机 相机 抖动 传感器 手电筒 拍照 分辨率 应用商店 流量 计时器 浏览器 游戏 升级 配置 vpn 通知栏 上网 拨号盘 水印 recovery qq 第三方 结构件 听筒 杀 切换 后台'''
    token_list = []
    for corpu in corpus:    
        result=[]
        words = pseg.cut(' '.join(jieba.cut(corpu)))
        for word, flag in words:
            word=word.lower()
            if flag not in stop_flag and word not in stopwords:
                if flag=='eng' and word not in focus_point:
                    continue
                result.append(word) 

        token_list.append(' '.join(result))


    return token_list
def get_data():
    # 读取英文电影数据
    #train_result, train_text = read_files('train')
    #test_result, test_text = read_files('test')
    
    #读取EXCEL数据
    train_df,train_result, train_text = get_data_xlsx("训练.xlsx")
    test_df,test_result, test_text = get_data_xlsx("近期版本故障提交列表0801.xlsx")
    
    # 读取垃圾邮件数据
    #train_df,train_result, train_text  = get_data_txt('train')
    #test_df,test_result, test_text  = get_data_txt('test')
    
    #text中文需要分词和去除特殊符号
    train_text_token = corpus2token(train_text)
    test_text_token = corpus2token(test_text)
    
    print('\ntrain_text_token[:3]\n',train_text_token[:3])

    #result字符串数据转换为数字标记
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(train_result)
    train_result_label_encoder=label_encoder.transform(train_result)
    test_result_label_encoder=label_encoder.transform(test_result)
    print("\nresult字符串数据转换为数字标记:")
    for i, item in enumerate(label_encoder.classes_):
        print(item, '-->', i,' ', end="")
    return train_text_token,train_result_label_encoder,train_df,test_text_token,test_result_label_encoder,test_df,label_encoder
    
