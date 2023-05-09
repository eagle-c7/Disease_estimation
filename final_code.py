#去噪音：副词、形容词如何解决+标点符号除去+姓名除去——找到停用词+标准化数据格式
#贝叶斯公式计算
#取概率前三的
import numpy as np
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.utils import Bunch
from sklearn.naive_bayes import MultinomialNB  # 多项式贝叶斯算法
import paddle # 百度提供的深度学习框架
import re
import xlwt

def readFile(path):
    with open(path, 'r', errors='ignore') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content

paddle.enable_static()
jieba.enable_paddle()

# 获取停用词
stopwordlist = readFile("./stopword.txt").splitlines()

# 训练集数据
df = pd.read_excel("./train.xlsx",header = 0)
data_train = df.values
bunch = Bunch(target = [], label = [], contents = [])
j = 0
for i in data_train:
    j = j + 1
    seg_str = str(i[0])
    seg_str = re.sub('\W*', '',seg_str) #分词前剔除特殊符号、标点符号
    #print("/".join(jieba.lcut(seg_str)))    # 精简模式，返回一个列表类型的结果
    #print("/".join(jieba.lcut_for_search(seg_str)))     # 搜索引擎模式
    bunch.label.append(i[1])
    bunch.contents.append(" ".join(jieba.lcut_for_search(seg_str)))
    #print("/".join(jieba.lcut(seg_str, use_paddle=True)))      # 全模式，使用 'cut_all=True' 指定
bunch.target = list(set(bunch.label))
# print(bunch.target)
# 用bunch存储数据
tfidfspace = Bunch(target = bunch.target, label = bunch.label, tdm = [], vocabulary = {})
vectorizer = TfidfVectorizer(stop_words=stopwordlist, sublinear_tf=True, max_df=0.5)
transformer = TfidfTransformer()
tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
tfidfspace.vocabulary = vectorizer.vocabulary_
bunch_train = tfidfspace
# 生成bunch_train用于后面的测试


# 测试集数据
k = 0
df2 = pd.read_excel("./test.xlsx", header = 0)
data_test = df2.values
bunch_test = Bunch(target = bunch_train.target, label=[], contents=[])
for i in data_test:
    k = k + 1
    seg_str = str(i[0])
    seg_str = re.sub('\W*', '',seg_str) #分词前剔除特殊符号、标点符号
    bunch_test.label.append(i[1])
    bunch_test.contents.append(" ".join(jieba.lcut(seg_str, use_paddle=True)))
# 用bunch_test
testspace = Bunch(target = bunch_test.target, label = bunch_test.label, tdm = [], vocabulary={})
vectorizer = TfidfVectorizer(stop_words=stopwordlist, sublinear_tf=True, max_df=0.5,vocabulary=bunch_train.vocabulary)
transformer = TfidfTransformer()
testspace.tdm = vectorizer.fit_transform(bunch_test.contents)
testspace.vocabulary = bunch_train.vocabulary



#贝叶斯
TrainSet = bunch_train
TestSet = testspace
clf = MultinomialNB(alpha=0.001).fit(TrainSet.tdm,TrainSet.label, None)
        
predicted = clf.predict_proba(TestSet.tdm)
total = len(predicted)
rate = 0
temp = sorted(TestSet.target)

#
dic = {}
for i in temp:
    dic[i] = [1, 1]

# 数据写入excel
book = xlwt.Workbook(encoding='utf-8',style_compression=0)
sheet = book.add_sheet('result',cell_overwrite_ok=True)
col = ('实际类别','预测类别1','预测类别2','预测类别3','是否预测正确')
#写入title
for i in range(0,5):
    sheet.write(0,i,col[i])

i = 0
for flabel, expct_cate in zip(TestSet.label, predicted):
    #找到expct里最大的三个数
    result = []
    max_index1 = np.argsort(expct_cate)[-1]
    max_index2 = np.argsort(expct_cate)[-2]
    max_index3 = np.argsort(expct_cate)[-3]
    result.append(temp[max_index1])
    result.append(temp[max_index2])
    result.append(temp[max_index3])
    flag = False
    dic[flabel][0] = dic[flabel][0] + 1
    if flabel in result:
        rate += 1
        flag = True
        dic[flabel][1] = dic[flabel][1] + 1
    i = i + 1
    sheet.write(i,0,flabel)
    for I in range(0,3):
        sheet.write(i,I+1,result[I])
    sheet.write(i,4,flag)
    #print("实际类别：", flabel, "-->预测类别：", result)
#book.save("./result2.csv")


sheet2 = book.add_sheet('test',cell_overwrite_ok=True)
col2 = ('实际类别','总次数','预测正确次数','预测正确率')
for i in range(0,4):
    sheet2.write(0,i,col2[i])

j = 0
for key in dic:
    j = j + 1
    sheet2.write(j, 0, key)
    sheet2.write(j, 1, dic[key][0])
    sheet2.write(j, 2, dic[key][1])
    sheet2.write(j, 3, dic[key][1]/dic[key][0])

book.save("./result.csv")

print(1.0*rate/total)
print(j, " ", k)
# 去除停用词
# 数据清洗
# 贝叶斯
# 取概率前三