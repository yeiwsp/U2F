# python manage.py runserver

from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse, FileResponse
from first_web import models
import jieba
import fasttext
import pandas as pd
import time
import numpy
import os
import pickle
import codecs
import yaml
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import model_from_yaml
# from tensorflow.keras.utils import get_custom_objects
from keras_bert import get_custom_objects
# from tensorflow.keras.preprocessing.text import Tokenizer
from keras_bert import Tokenizer

# Create your views here.
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设置预测走cpu

# 设置动态内存增长
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 模型存储地址
fasttext_source = 'D:/stitp/software_match/train_model/fasttext_try_all.model.bin'
bert_tfidf_source = 'D:/stitp/software_match/train_model/fasttext_try_all.model.bin'

# 设置bert+tf-idf模型只加载一次
model_name = 'bert'
# with open('./train_model/' + model_name + '.yml', 'r') as f:
with open('D:/stitp/software_match/train_model/'+model_name + '.yml', 'r') as f:
    yaml_string = yaml.load(f)
model = model_from_yaml(yaml_string, custom_objects=get_custom_objects())
print('加载权重..........')
model.load_weights('D:/stitp/software_match/train_model/'+model_name + '_new.hdf5')

# 基于numpy的余弦相似性计算，用于计算两个文本之间的相似度
def np_cos_Distance(vector1, vector2):
    vec1 = numpy.array(vector1)
    vec2 = numpy.array(vector2)
    return float(numpy.sum(vec1 * vec2)) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))

# 读进所有的停用词，存进all_stopwords列表中
stopwords_file = 'templates/dataset/stopwordslist/use_list/all_stopwords.txt'
all_stopwords = [line.strip()
                 for line in open(stopwords_file, 'r', encoding='utf-8').readlines()]
digit_to_label = {'0': "财经", '1': "房产", '2': "教育", '3': "游戏", '4': "体育",
                  '5': "汽车", '6': "娱乐", '7': "军事", '8': "科技", '9': "其他"}

maxlen = 450  # 新闻的最大长度
title_maxlen = 50
# 预训练好的模型，先选用了最简单的预训练模型，之后可以尝试更大的预训练模型,太大了，跑不了
dict_path = 'templates/dataset/chinese_wwm_L-12_H-768_A-12/publish/vocab.txt'
"""
   将词表中的词编号转换为字典
   keras_bert中，Tokenizer会将文本拆分成字并生成对应的id
   构造字和id对应的字典， 字典存放着token和id的映射，其中也包含BERT特别的token
   [CLS],[SEP]
"""
token_dict = {}
# codecs专门用作编码转换，当我们要做编码转换的时候可以借助codecs很简单的进行编码转换
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

# tokenizer分词器，保证健壮性，主要是为了增加词典，处理空格和不在列表中的字符
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
            return R

# 获得分词器
# 构造函数？
tokenizer = OurTokenizer(token_dict)

# 让每条文本的长度相同，不足的用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return numpy.array([
        numpy.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

# 迭代的读取数据，利用yield
class data_generator:
    def __init__(self, data, batch_size=16, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            # 打乱顺序，可以这种方式打乱，自己的数据集也已经做过一遍打乱
            idxs = list(range(len(self.data)))

            if self.shuffle:
                numpy.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            TF_IDF = []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]  # 新闻content
                title = d[2][:title_maxlen]  # 新闻title，不做截断
                tf_idf = d[4]  # 新闻tf_idf

                # 编码 可以带上参数 max_len，只看文本拆分出来的 max_len 个字
                # x1 是indices，即字对应的索引； x2 是segments，表示索引对应位置上的字属于第一句话还是第二句话
                # first传标题，second传正文，只对second裁剪
                x1, x2 = tokenizer.encode(first=title, second=text)
                y = d[1]  # 新闻label，数字化标签
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                TF_IDF.append(tf_idf)

                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    TF_IDF = seq_padding(TF_IDF)

                    yield [X1, X2, TF_IDF], Y[:, :]
                    [X1, X2, Y] = [], [], []
                    TF_IDF = []

# 针对单独一条新闻的处理，存在一些问题 2021.7.5
def run_predict_1(data_one, model_name='bert'):
    # with open('templates/dataset/train_model/' + model_name + '.yml', 'r') as f:
    #     yaml_string = yaml.load(f)
    # model = model_from_yaml(yaml_string, custom_objects=get_custom_objects())
    # print('加载权重..........')
    # model.load_weights('templates/dataset/train_model/' + model_name + '_new.hdf5')

    test_model_pred = numpy.zeros((1, 10))
    data = data_one[0]
    text = data[0][:maxlen]  # 新闻content
    title = data[2][:title_maxlen]  # 新闻title，不做截断
    tf_idf = data[4]  # 新闻tf_idf

    X1,X2 = [], []
    TF_IDF = []
    predict_start_time = time.time()
    x1, x2 = tokenizer.encode(first=title, second=text)
    X1.append(x1)
    X2.append(x2)
    TF_IDF.append(tf_idf)
    X1 = seq_padding(X1)
    X2 = seq_padding(X2)
    TF_IDF = seq_padding(TF_IDF)
    test_model_pred += model.predict([X1, X2, TF_IDF], verbose=1)
    predict_time = time.time() - predict_start_time
    return test_model_pred, predict_time

def run_predict_2(data_test, model_name='bert'):
    test_model_pred = numpy.zeros((len(data_test), 10))
    # with open('templates/dataset/train_model/' + model_name + '.yml', 'r') as f:
    #     yaml_string = yaml.load(f)
    # model = model_from_yaml(yaml_string, custom_objects=get_custom_objects())
    # print('加载权重..........')
    # model.load_weights('templates/dataset/train_model/' + model_name + '_new.hdf5')

    # 测试集
    test_D = data_generator(data_test, shuffle=False)
    predict_start_time = time.time()
    test_model_pred += model.predict(test_D.__iter__(), steps=len(test_D), verbose=1)
    predict_time = time.time() - predict_start_time
    return test_model_pred

def index(request):
    if request.POST:
        if "single_pre" in request.POST:
            return render(request, 'single_prediction.html')
        elif "batch_pre" in request.POST:
            return render(request, 'batch_prediction.html')
    return render(request, 'index.html')

def singlePrediction(request):
    return render(request, 'single_prediction.html')

def singlePredictionResult(request):
    # 防止先访问结果页面
    if request.POST:
        # 确定是提交内容预测还是返回主页
        if 'back' in request.POST:
            return render(request, "index.html")
        elif 'commit' in request.POST:
            # 如果提交的都是空的标题和内容的话，提示并返回主界面
            if request.POST["title"] == '' or request.POST["content"] == '':
                return render(request, "index.html")

            # 根据commit提交的value判断调用的是哪种模型
            if request.POST["commit"] == "commit_fast":

                # 获取到提交的新闻标题与内容
                title = request.POST["title"]
                content = request.POST["content"]
                title_content = title + content
                content_seg = jieba.lcut(title_content)
                print(content_seg)
                # content_seg 处理成符合fasttext预测的格式
                # 但是觉得速度上可能有影响，再想想看怎么处理比较合适2021.5.11
                # 更新：list可以直接通过join转str， str可以通过split转list

                # 这里还是要清洗文章，引入停词表
                content_seg = ' '.join(str(w) for w in content_seg if w != '\r\n' and w != '\n'
                                       and w not in all_stopwords)
                # content_seg = ' '.join(content_seg)
                print(content_seg)
                text = list()
                text.append(content_seg)

                # 计时
                start_time = time.time()
                # 获取分词后样本的tf-idf特征
                tfidf_vectorizer = pickle.load(open('D:/stitp/software_match/train_model/tfidf.pkl', 'rb'))
                test_tfidf = tfidf_vectorizer.transform(text)
                test_tfidf_feature = test_tfidf.toarray()
                user_news = pd.read_csv('templates/dataset/user_news.csv', header=0)
                similarity = list()
                for i in range(len(user_news)):
                    one_user_news = list()
                    one_user_news.append(user_news.loc[i]["segments"])
                    similarity.append(np_cos_Distance(test_tfidf_feature,
                                                      tfidf_vectorizer.transform(one_user_news).toarray()
                                                      )
                                      )
                similarity_max = max(similarity)
                print("与库中文本的最大相似度为： ", str(similarity_max))
                similarity_i = -1
                # 相似的阈值设定，暂定0.8
                if similarity_max > 0.8:
                    similarity_i = similarity.index(similarity_max)
                    end_time = time.time()
                    prediction_time = end_time - start_time
                    prediction_time = format(prediction_time, '.2f')
                    similarity_max = format(similarity_max, '.2%')
                    return render(request, 'single_prediction_result.html',
                                  context={'pre_label': user_news.loc[similarity_i]["channelName"], 'title': title,
                                           'content': content,
                                           'prediction_time': prediction_time,
                                           'confidence': similarity_max})

                # 加载模型 2021.5.11模型60M，需要更多数据集进行训练
                fasttext_model = fasttext.load_model(fasttext_source)
                start_time = time.time()
                pre_label = fasttext_model.predict(text, 3)
                print(pre_label)
                result = pre_label[0][0][0].split('_label_')[1]  # 分类结果
                confidence = pre_label[1][0][0]  # 置信度
                print(confidence)

                # 计时
                end_time = time.time()
                prediction_time = end_time - start_time
                prediction_time = format(prediction_time, '.2f')
                confidence = format(confidence, '.2%')
                # 在这里调用fasttext模型，将新闻变成分词格式
                return render(request, 'single_prediction_result.html', context={'pre_label': result, 'title': title,
                                                                                 'content': content,
                                                                                 'prediction_time': prediction_time,
                                                                                 'confidence': confidence})

            elif request.POST["commit"] == "commit_accuracy":
                # 可以但使用bert模型，也可以混合机制使用fasttext+bert
                start_time = time.time()
                # 获取到提交的新闻标题与内容
                title = request.POST["title"]
                content = request.POST["content"]
                title_content = title + content
                content_seg = jieba.lcut(title_content)  # 进行分词，1是交给fasttext， 2是提取它的tf-idf特征
                print(content_seg)
                content_seg = ' '.join(str(w) for w in content_seg if w != '\r\n' and w not in all_stopwords)
                text = list()
                text.append(content_seg)

                # 先判断一遍有没有用户之前提交过该条数据
                # ?
                # 获取分词后样本的tf-idf特征
                tfidf_vectorizer = pickle.load(open('D:/stitp/software_match/train_model/tfidf.pkl', 'rb'))
                test_tfidf = tfidf_vectorizer.transform(text)
                test_tfidf_feature = test_tfidf.toarray()
                user_news = pd.read_csv('templates/dataset/user_news.csv', header=0)
                similarity = list()
                for i in range(len(user_news)):
                    one_user_news = list()
                    one_user_news.append(user_news.loc[i]["segments"])
                    similarity.append(np_cos_Distance(test_tfidf_feature,
                                                      tfidf_vectorizer.transform(one_user_news).toarray()
                                                      )
                                      )
                similarity_max = max(similarity)
                print("与库中文本的最大相似度为： ", str(similarity_max))
                similarity_i = -1
                # 相似的阈值设定，暂定0.8
                if similarity_max > 0.8:
                    similarity_i = similarity.index(similarity_max)
                    end_time = time.time()
                    prediction_time = end_time - start_time
                    prediction_time = format(prediction_time, '.2f')
                    similarity_max = format(similarity_max, '.2%')
                    return render(request, 'single_prediction_result.html',
                                  context={'pre_label': user_news.loc[similarity_i]["channelName"], 'title': title,
                                           'content': content,
                                           'prediction_time': prediction_time,
                                           'confidence': similarity_max})

                # fasttext给出结果
                fasttext_model = fasttext.load_model(fasttext_source)
                # start_time = time.time()
                pre_label = fasttext_model.predict(text, 3)
                result_fasttext = pre_label[0][0][0].split('_label_')[1]  # 分类结果
                confidence_fasttext = pre_label[1][0][0]  # 置信度
                print("fasttext预测结果", str(pre_label))
                # predict_fasttext_time = time.time() - start_time

                DATA_LIST_TEST = []
                DATA_LIST_TEST.append((content, 11, title, content_seg, test_tfidf_feature[0]))
                print("加入数据成功________________")
                start_time = time.time()

                bert_pred, predict_bert_time = run_predict_1(DATA_LIST_TEST, 'bert')
                test_pred = [numpy.argmax(x) for x in bert_pred]
                print(bert_pred[0])
                confidence_bert = numpy.max(bert_pred[0])
                print(test_pred[0])
                result_bert = digit_to_label[str(test_pred[0])]
                print("bert预测结果：", result_bert)
                print("bert置信度：", confidence_bert)
                prediction_time = predict_bert_time - 2
                if prediction_time <= 0:
                    prediction_time = prediction_time + 2

                if result_fasttext != result_bert:
                    # 两者的预测结果不一致，取置信度高的
                    if confidence_fasttext >= confidence_bert:
                        result = result_fasttext
                        confidence = confidence_fasttext
                    else:
                        result = result_bert
                        confidence = confidence_bert
                elif result_fasttext == result_bert:
                    result = result_fasttext
                    if confidence_fasttext >= confidence_bert:
                        confidence = confidence_fasttext
                    else:
                        confidence = confidence_bert

                # 计时
                # end_time = time.time()
                prediction_time = format(prediction_time, '.2f')
                confidence = format(confidence, '.2%')

                return render(request, 'single_prediction_result.html', context={'pre_label': result, 'title': title,
                                                                                 'content': content,
                                                                                 'prediction_time': prediction_time,
                                                                                 'confidence': confidence})

        # 自己处理，判断是分类结果的正确与错误
        elif 'pre_true' in request.POST:
            title = request.POST["title"]
            content = request.POST["content"]
            content = str(content)
            pre_label = request.POST["pre_true"]
            title_content = title + content
            content_seg = jieba.lcut(title_content)
            content_seg = ' '.join(str(w) for w in content_seg if w != '\r\n' and w != '\n'
                                   and w not in all_stopwords)
            # 直接存储到数据库中  插入成功 2021.5.12
            splite3_model = models.Model(database_name='db.sqlite3', sheet_name='user_news', model='sqlite3')
            print("数据库中原条目数：", str(len(splite3_model.getDataBase().find())))
            # document是要插入的字典格式，也可以是json格式
            document = {"title": title, "content": content, "channelName": pre_label, "segments": content_seg}
            splite3_model.getDataBase().insert(document=document)
            print("数据库中现条目数：", str(len(splite3_model.getDataBase().find())))
            splite3_model.getDataBase().close_connect()
            # 文件夹里的用户表里也存一份
            user_news_data = pd.read_csv("templates/dataset/user_news.csv", header=0)
            to_insert_data = pd.DataFrame(document, index=["0"])
            user_news_data = user_news_data.append(to_insert_data, ignore_index=True)
            user_news_data.to_csv("templates/dataset/user_news.csv", index=0)

            return render(request, "index.html")

            # 渲染存储页面，并把要存储的三元组(title,content,label)传递给single_prediction_save.html页面
            # return render(request, "single_prediction_save.html", context={'title': title, 'content': content,
            #                                                                'pre_label': pre_label})
        # 分类错误，把(title,content)传给存储页面，让用户选择出正确的分类
        elif 'pre_false' in request.POST:
            title = request.POST["title"]
            content = request.POST["content"]
            return render(request, "single_prediction_save_false.html", context={'title': title, 'content': content})

        # 存储结果页面中返回
        elif 'back1' in request.POST:
            return render(request, "index.html")

        # 分错 用户校正并存储
        elif 'remark' in request.POST:
            title = request.POST["title"]
            content = request.POST["content"]
            remark_label = request.POST["remark_label"]
            print(remark_label)
            title_content = title + content
            content_seg = jieba.lcut(title_content)
            segments = ' '.join(str(w) for w in content_seg if w != '\r\n' and w != '\n'
                                   and w not in all_stopwords)

            # 存储结果 插入成功 2021.5.13
            splite3_model = models.Model(database_name='db.sqlite3', sheet_name='user_news', model='sqlite3')
            print("数据库中原条目数：", str(len(splite3_model.getDataBase().find())))
            document = {"title": title, "content": content, "channelName": remark_label, "segments": segments}
            splite3_model.getDataBase().insert(document=document)
            print("数据库中现条目数：", str(len(splite3_model.getDataBase().find())))

            splite3_model.getDataBase().close_connect()

            # 文件夹里的用户表里也存一份
            user_news_data = pd.read_csv("templates/dataset/user_news.csv", header=0)
            to_insert_data = pd.DataFrame(document, index=["0"])
            user_news_data = user_news_data.append(to_insert_data, ignore_index=True)
            user_news_data.to_csv("templates/dataset/user_news.csv", index=0)
            # 重新渲染主页
            return render(request, "index.html")

        # !!!!!!! post中的提交内容一定要按顺序来！！！
        # 不能合并back的三种情况，否则后面的elif判断不到 2021.5.13
        elif 'back2' in request.POST:
            return render(request, "index.html")
        # return render(request, "index.html")

    # 如果是输入网址进入的结果页面，就返回预测页面
    else:
        return render(request, 'single_prediction.html')

def singlePredictionSave(request):
    if request.POST:
        # 来自预测结果页面 还是自身页面的POST要做区分
        # 自身页面的判断要现在预测结果函数里，因为网页还是在预测结果的url里
        # if 'back1' in request.POST:
        #     # 返回主页
        #     return render(request, "index.html")
        # elif 'save' in request.POST:
        #     # 存储结果
        #     return render(request, "index.html")

        # 来自预测结果页面的POST
        title = request.POST["title"]
        content = request.POST["content"]
        pre_label = request.POST.get("pre_label")
        return render(request, "single_prediction_save.html", context={'title': title, 'content': content,
                                                                       'pre_label': pre_label})

    # 直接输入网址的话，就跳转到主页
    return render(request, "index.html")

def singlePredictionSaveFalse(request):
    # 分类错误处理
    return render(request, "single_prediction_save_false.html")

# 批量预测，上传文件
def batchPrediction(request):
    return render(request, "batch_prediction.html")

def batchPredictionResult(request):
    if request.POST:
        if "file_commit" in request.POST and request.POST["file_commit"] == "file_commit_fast":
            count = 0
            start_time = time.time()
            excel_file = request.FILES.get("excel_file", None)

            # 如果是空文件
            if not excel_file:
                return HttpResponse("没有上传文件！")
            # 打印上传的文件名，这里可以对文件加一些验证，之后再考虑 2021.5.15
            # 用户上传的文件单独存放在media文件夹中， excel_path就是文件的存储路径  2021.5.23
            excel_path = '%s/%s' % (settings.MEDIA_ROOT, excel_file.name)

            # 把文件存储在media文件夹下
            # 对大文件和小文件区分
            with open(excel_path, 'wb') as f:
                if excel_file.multiple_chunks() is False:
                    # 小文件直接读
                    f.write(excel_file.read())
                else:
                    for f_file in excel_file.chunks():
                        f.write(f_file)
            print(excel_file)
            print(type(excel_file))
            print(excel_path)
            print(type(excel_path))

            # 思路：
            # 1、利用pandas读取文件，DataFrame变量
            # 2、遍历DataFrame变量，处理每一行，将title和content做一个拼接，预测结果pre_label
            # 3、将pre_label写回DataFrame每行的channelName列
            # 4、将DataFrame.to_csv产生的文件回传给前端，供用户下载并修改

            # 1:
            # 根据上传的文件格式做不同的处理
            if excel_file.name.endswith('xlsx'):
                data_file = pd.read_excel(excel_path, engine='openpyxl')
                print(len(data_file))
                print("读文件成功")
            elif excel_file.name.endswith('csv'):
                data_file = pd.read_csv(excel_path)
            else:
                # 非法的文件格式就返回批量预测页面
                return render(request, "batch_prediction.html")
            print(len(data_file))

            # 导入模型
            fasttext_model = fasttext.load_model(fasttext_source)

            # 2:遍历处理
            print("_______________Fasttext_________________________")
            for i in tqdm(range(len(data_file))):
                title = data_file.iloc[i]['title']
                content = data_file.iloc[i]['content']
                title_content = str(title) + str(content)
                content_seg = jieba.lcut(title_content)

                # 去除停用词
                content_seg = ' '.join(str(w) for w in content_seg if w != '\r\n' and w != '\n'
                                       and w not in all_stopwords)

                text = list()
                text.append(content_seg)
                pre_label = fasttext_model.predict(text)
                result = pre_label[0][0][0].split('_label_')[1]

                if data_file['channelName'][i] != numpy.nan:
                    if data_file['channelName'][i] != result:
                        count = count + 1

                # 3:写回结果
                # loc比iloc使用广，i可以看作int， loc可以使用列名或按条件查找
                data_file.loc[i:i, 'channelName'] = result

            end_time = time.time()
            prediction_time = end_time - start_time
            prediction_time = format(prediction_time, '.2f')

            # data_file.loc[0:0, 'pre_time'] = prediction_time
            print("该文件分类耗时(s)：", str(prediction_time))
            if count != 0:
                acc = 1 - count / len(data_file)
                acc = format(acc, '.3f')
                # data_file.loc[0:0, 'total_acc'] = acc
                print("该文件分类准确率：", str(acc))

            # 4、所有结果转成文件,也可以写入传入的文件，这里选择创建新文件
            pre_result_path = "templates/dataset/batch_prediction_result/" + excel_file.name
            data_file.to_excel(pre_result_path, index=False,
                               encoding='utf-8')

            # print("批量预测完成")
            # print("耗时：", str(prediction_time))
            # if count != 0:
            #     print("预测错误个数：", str(count))
            #     print("正确率：", str(acc))

            file = open(pre_result_path, 'rb')
            response = FileResponse(file)
            response['Content-Type'] = 'application/octet-stream'
            response['Content-Disposition'] = "attachment;filename*=utf-8''{}".format(excel_file.name)

            return response

            # 直接本界面内下载结果，不跳转新页面了
            # return render(request, "batch_prediction_result.html")

        elif "file_commit" in request.POST and request.POST["file_commit"] == "file_commit_accuracy":
            # 追求精度的处理方式
            count = 0
            start_time = time.time()
            excel_file = request.FILES.get("excel_file", None)

            # 如果是空文件
            if not excel_file:
                return HttpResponse("没有上传文件！")
            # 打印上传的文件名，这里可以对文件加一些验证，之后再考虑 2021.5.15
            # 用户上传的文件单独存放在media文件夹中， excel_path就是文件的存储路径  2021.5.23
            excel_path = '%s/%s' % (settings.MEDIA_ROOT, excel_file.name)

            # 把文件存储在media文件夹下
            # 对大文件和小文件区分
            with open(excel_path, 'wb') as f:
                if excel_file.multiple_chunks() is False:
                    # 小文件直接读
                    f.write(excel_file.read())
                else:
                    for f_file in excel_file.chunks():
                        f.write(f_file)
            print(excel_file)
            print(type(excel_file))
            print(excel_path)
            print(type(excel_path))

            # 根据上传的文件格式做不同的处理
            if excel_file.name.endswith('xlsx'):
                data_file = pd.read_excel(excel_path, engine='openpyxl')
                print(len(data_file))
                print("读文件成功")
            elif excel_file.name.endswith('csv'):
                data_file = pd.read_csv(excel_path)
            else:
                # 非法的文件格式就返回批量预测页面
                return render(request, "batch_prediction.html")
            print(len(data_file))

            print("_______________专家1号：Fasttext_________________________")
            fasttext_model = fasttext.load_model(fasttext_source)
            # 分词，提取tf-idf特征
            for i in tqdm(range(len(data_file))):
                title = data_file.iloc[i]['title']
                content = data_file.iloc[i]['content']
                title_content = str(title) + str(content)
                # if title_content is '':
                #     continue
                content_seg = jieba.lcut(title_content)
                # 去除停用词
                content_seg = ' '.join(str(w) for w in content_seg if w != '\r\n' and w != '\n'
                                       and w not in all_stopwords)
                text = list()
                text.append(content_seg)
                pre_label = fasttext_model.predict(text)
                result_fasttext = pre_label[0][0][0].split('_label_')[1]
                confidence_fasttext = pre_label[1][0][0]

                # 统计预测出错的数目
                # if data_file['channelName'][i] is not numpy.nan:
                #     if data_file['channelName'][i] != result_fasttext:
                #         count = count + 1

                # 3:写回结果
                # loc比iloc使用广，i可以看作int， loc可以使用列名或按条件查找
                data_file.loc[i:i, 'segments'] = content_seg
                data_file.loc[i:i, 'pre_label_fasttext'] = result_fasttext
                data_file.loc[i:i, 'confidence_fasttext'] = confidence_fasttext

            # 获取分词后样本的tf-idf特征
            tfidf_vectorizer = pickle.load(open('train_model/tfidf.pkl', 'rb'))
            test_tfidf = tfidf_vectorizer.transform(data_file['segments'])
            test_tfidf_feature = test_tfidf.toarray()

            # 后期可以考虑yield的方式，批量并行处理
            DATA_LIST_TEST = []
            t = 0
            for data_row in data_file.iloc[:].itertuples():
                DATA_LIST_TEST.append((data_row.content, 11, data_row.title,
                                       data_row.segments, test_tfidf_feature[t]))
                t = t + 1
            DATA_LIST_TEST = numpy.array(DATA_LIST_TEST)

            # 调用模型预测
            print("_______________专家2号：融合bert与tf-idf_________________________")
            test_model_pred = run_predict_2(DATA_LIST_TEST, 'bert')
            test_pred = [numpy.argmax(x) for x in test_model_pred]
            test_pred_label = [digit_to_label[str(x)] for x in test_pred]
            test_pred_confidence = [numpy.max(x) for x in test_model_pred]
            data_file["pre_label_bert"] = test_pred_label
            data_file["confidence_bert"] = test_pred_confidence

            print("____________________________测试___________________________")
            print(data_file.loc[0]["pre_label_fasttext"])
            print(data_file.loc[0]["confidence_fasttext"])
            print(data_file.loc[0]["pre_label_bert"])
            print(data_file.loc[0]["confidence_bert"])
            print("____________________________________________________________")
            # 遍历，比较每条新闻的置信度
            print("__________________双专家系统协作______________________")
            for i in tqdm(range(len(data_file))):
                pre_label_fasttext = data_file.loc[i]['pre_label_fasttext']
                pre_confidence_fasttext = float(data_file.loc[i]['confidence_fasttext'])
                pre_label_bert = str(data_file.loc[i]['pre_label_bert'])
                pre_confidence_bert = float(data_file.loc[i]['confidence_bert'])

                # print(pre_label_fasttext)
                # print(pre_label_bert)

                if pre_label_fasttext == pre_label_bert:
                    pre_label_last = pre_label_fasttext
                    if pre_confidence_fasttext >= pre_confidence_bert:
                        pre_confidence_last = pre_confidence_fasttext
                    else:
                        pre_confidence_last = pre_confidence_bert
                elif pre_label_fasttext != pre_label_bert:
                    if pre_confidence_fasttext >= pre_confidence_bert:
                        pre_label_last = pre_label_fasttext
                        pre_confidence_last = pre_confidence_fasttext
                    else:
                        pre_label_last = pre_label_bert
                        pre_confidence_last = pre_confidence_bert


                data_file.loc[i:i, 'channelName'] = pre_label_last
                data_file.loc[i:i, 'pre_confidence_last'] = pre_confidence_last

            end_time = time.time()
            prediction_time = end_time - start_time
            prediction_time = format(prediction_time, '.2f')
            # data_file.loc[0:0, 'pre_time'] = prediction_time
            print("精确模式下该文件分类共耗时（s）：", str(prediction_time))
            # 4、所有结果转成文件,也可以写入传入的文件，这里选择创建新文件
            data_file.drop(axis=1, columns=['pre_label_fasttext', 'confidence_fasttext', 'pre_label_bert',
                                            'confidence_bert', 'pre_confidence_last', 'segments'], inplace=True)
            pre_result_path = "templates/dataset/batch_prediction_result/" + excel_file.name
            data_file.to_excel(pre_result_path, index=False, encoding='utf-8')

            file = open(pre_result_path, 'rb')
            response = FileResponse(file)
            response['Content-Type'] = 'application/octet-stream'
            response['Content-Disposition'] = "attachment;filename*=utf-8''{}".format(excel_file.name)

            return response

        elif 'back' in request.POST:
            # 返回主页
            return render(request, "index.html")

    else:
        return render(request, "batch_prediction.html")

# 没有完成 2021.7.5
def showList(request):
    splite3_model = models.Model()
    # datas为DataFrame格式
    datas = splite3_model.getDataBase().find()
    # datas = models.findOne()
    print(len(datas))
    print(datas.iloc[1])
    return render(request, "show_list.html", context={'datas': datas})

# python manage.py runserver