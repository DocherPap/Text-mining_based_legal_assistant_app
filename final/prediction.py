
# coding: utf-8

# In[4]:


import jieba
import logging
import sys
import codecs
import traceback
import pandas as pd
import numpy as np
import re
from collections import Counter
from pymongo import MongoClient 
jieba.load_userdict('D:/model/台灣法律用語.txt')


if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def delCNf(line):
    regex = re.compile('[^\u4e00-\u9fa5a-zA-Z\s]')
    return regex.sub('', line)


def str_replace(str_source, char, *words):
    str_temp = str_source
    for word in words:
        str_temp = str_temp.replace(word, char)
    return str_temp


def seg_words(sentence):
    stopwords = {}.fromkeys([ line.rstrip() for line in open('D:/model/stopword.txt', encoding = 'utf8') ])
    segs = jieba.cut(sentence, cut_all=False)  # 默认是精确模式
    return " ".join(segs)     # 分词，然后将结果列表形式转换为字符串


def segmentation(file):
    files = re.split(",", file)
    lst = []
    for line in files:
        line = delCNf(line)
        line = str_replace(line, "", "\t", "\n", " ", "\u3000", "○")
        seg_list = jieba.cut(line, cut_all = False)
        words = " ".join(seg_list)
        words = "".join(words.split('\n')) # 去除回车符
        words = delCNf(words)
        word = re.split(r" ", words)
        lst.extend(word)
    return lst


from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn import *
from random import randint
from sklearn.externals import joblib
import math
import pickle
import lightgbm as lgb 
import os

def made_matrix(string):
    matrix = pd.read_csv("D:/model/matrix_column.csv", encoding = 'utf8')
    matrix.loc[0] = randint(0,0)

    #for col in matrix.columns:
    #    count = 0
    #    for word in segmentation(string):
     #       if str(word) == str(col):
    #           count += 1
     #   matrix.loc[0, col] = count

    df = pd.Series(segmentation(string)).value_counts()

    for col in matrix.columns:
        for i in range(df.size):
            if df.index[i] == str(col):
                matrix.loc[0,col] = df[i]

    predict_x = matrix.dropna().astype('int')
    print(predict_x)
    
    return predict_x


# In[5]:


def predict_vic(predict_x):
    reload = lgb.Booster(model_file='D:/model/lightgbm_model.txt')
    
    y_pred = reload.predict(predict_x)
    y_pred = [list(x).index(max(x)) for x in y_pred]
    
    return y_pred # 0 勝，1 平， 2 敗訴


def knn_five_judgement(predict_x):
    df_dataframe = pd.read_csv("D:/model/matrix1500_new.csv", encoding = 'utf8').reset_index().dropna()
    
    reload_knn = joblib.load('D:/model/knn_model.model')
    neighborpoint = reload_knn.kneighbors(predict_x, 5, False)
    neighborlst = []
    for idx in neighborpoint[0]:
        neighborlst.append(df_dataframe.loc[idx,'filename'])
        
    return neighborlst

    
conn = MongoClient('mongodb://dajiayiqibiye:wxby@wxby-shard-00-00-7ea9c.mongodb.net:27017,wxby-shard-00-01-7ea9c.mongodb.net:27017,wxby-shard-00-02-7ea9c.mongodb.net:27017/test?ssl=true&replicaSet=WXBY-shard-0&authSource=admin&retryWrites=true')#改成本機ip
db = conn.wxby
case_set = db.case_consult
judgement_set = db.judgement

#a = '201812210147380970557865'
a = sys.argv[1]
condition = {'id': a}
#a = case_set.find_one(condition)
print(a)
#print(type(sys.argv[1]))

string = case_set.find_one(condition).get("content")
#string = "本件上訴人主張：兩造於民國九十九年五月十七日訂立房屋租賃契約書（下稱系爭租約），由被上訴人承租上訴人所有坐落新北市○○區○○路○○○號六樓如一審判決附圖所示斜線部分面積二九六坪之建物（下稱系爭建物）及二汽車車位（下與系爭建物合稱系爭租賃物），租期自九十九年八月一日起至一○二年七月三十一日，每月租金新台幣（下同）十五萬二千二百九十二元。詎被上訴人自一○○年十月一日起，僅繳交租金之半額即七萬六千一百四十六元。經上訴人於一○一年五月四日以郵局存證信函催告被上訴人於函到後五日內給付所積欠之租金，再於一○一年六月二十二日以存證信函催告被上訴人於函到後五日內給付積欠租金，逾期即以該函為終止租約之意思表示，被上訴人於一○一年六月二十五日收受該存證信函，仍置之不理，則系爭租約業於同年六月三十日終止，被上訴人共積欠九個月租金差額六十八萬五千三百一十四元。另依租約第十二條第四款之約定，被上訴人應自一○一年七月一日起至返還系爭租賃物時止，按月給付上訴人租金二倍計算即三十萬四千五百八十四元之違約金，其中一○一年七、八月之違約金共六十萬九千一百六十八元，經以押租金四十三萬五千一百二十元扣抵後，上訴人尚得請求十七萬四千零四十八元等情。爰依系爭租約之約定，求為命被上訴人給付八十五萬九千三百六十二元及自一○一年九月一日起至返還系爭租賃物之日止，按月給付三十萬四千五百八十四元之判決（原審就其中租金二十五萬零一百九十四元本息及上訴人另請求被上訴人給付電費八萬五千六百一十八元本息暨返還系爭租賃物部分，為被上訴人敗訴之判決，被上訴人未聲明不服，該未繫屬本院部分，不予贅列）。被上訴人則以：系爭建物為無法辦理工廠登記之違章建築，伊乃於九十九年七月二十六日同時與上訴人及偉灃實業股份有限公司（下稱偉灃公司）分別就系爭建物及新北市○○區○○路○○○號六樓（下稱一二五號六樓建物）簽訂房屋租賃契約書（下稱系爭七月租約、偉灃公司租約），約定每月租金各為七萬六千一百四十六元，伊按時繳納租金予上訴人及偉灃公司，並無欠租，上訴人逕自終止系爭租約，不生效力等語，資為抗辯。原審廢棄第一審所為命被上訴人給付六十萬九千一百六十八元本息及自一○一年九月一日起按月給付三十萬四千五百八十四元部分之判決，改判駁回上訴人該部分之訴，無非以：兩造於九十九年五月十七日訂立系爭租約，約定每月租金為十五萬二千二百九十二元之事實，為被上訴人所不爭執。依證人即負責上訴人租賃管理業務之陳淑芬之證言及被上訴人員工林中一予上訴人之電子郵件表示「合約書修正如上，請參照，主要內容為將租金平均分配至二份合約中，若無需修正部分，我們可約時間用印簽約，另進行換票事宜（我司將以貴公司兩家子公司名義開票，平均租金於兩家公司）」，可證被上訴人因系爭建物未辦保存登記，為以一二五號六樓建物作為形式工廠登記之用，乃與上訴人合意將系爭租約分為兩份，並將系爭租約之租金平均分配至七月租約、偉灃公司租約，然兩造權利義務關係仍以系爭租約內容定之。而系爭租約第三條約定租金為每月十五萬二千二百九十二元（含稅），於每月一日給付，被上訴人既不爭執自九十九年八月起至一○○年九月止，均支付每月租金十五萬二千二百九十二元，則被上訴人應係自一○○年十月一日起未給付其中一半租金七萬六千一百四十六元，至一○一年二月二十九日止共積欠租金三十八萬零七百三十元，依土地法第一百條第三款規定，先就押租金四十三萬五千一百二十元扣抵後，尚有押租金五萬四千三百九十元未抵付，縱不續抵付所剩押租金，自一○一年三月一日至同年五月三十一日止共三個月，既已給付一半租金，合計仍僅欠租一個半月，未達二個月租金，上訴人於一○一年五月四日即以存證信函催告被上訴人於函到後五日內給付所積欠自一○○年十月一日至一○一年五月三十一日之八個月之租金，與系爭租約第十條第三項約定「租約終止：…三、乙方拖欠租金達二個月以上，經甲方『二次』書面催告仍不支付或履行者。」不符，其催告不合法。上訴人再於一○○年六月二十二日以存證信函催告被上訴人於函到後五日內給付租金，並以逾期即以該函為終止兩造間租賃契約之意思表示，因未經二次合法催告，其終止亦不合法。故兩造租賃關係仍應依系爭租約第二條約定，至一○二年七月三十一日始因屆滿而告消滅。上訴人依系爭租約第三條約定，請求一○一年三月一日起至同年六月三十日止所積欠之租金共二十五萬零一百九十四元，即屬有據，逾此請求，為無理由。又上訴人在系爭租約屆滿前，已持一審判決聲請假執行，被上訴人並已搬遷，自無系爭租約第十二條第四款約定之租約終止仍未返還租賃物情形，上訴人請求被上訴人自一○一年七月起（先扣抵押租金）至返還租賃物止，按月給付租金二倍計算之違約金三十萬四千五百八十四元，亦屬無據等詞，為其判斷之基礎。按土地法第一百條係就出租人得收回租賃房屋之事由所為之規定，與民法第四百四十條第一項出租人定期催告承租人支付租金之規定，乃屬二事。故出租人基於土地法第一百條第三款承租人欠租之事由收回房屋，仍應依民法第四百四十條第一項規定，對於支付租金遲延之承租人定相當期限催告其支付（司法院院解字第三四八九號解釋參照）。又承租人遲延支付租金時，出租人即得限期催告承租人支付租金，且其催告租金額超過承租人應付之金額時，僅超過部分不發生效力，並非該催告全不發生效力（本院六十六年台上字第一二四號判例參照）。查本件兩造所約定之租金為每月一日給付十五萬二千二百九十二元，被上訴人自一○○年十月一日起每月未給付一半之租金七萬六千一百四十六元，已有遲延給付租金之情事，乃原審確認之事實，依民法第四百四十條第一項規定，上訴人得定相當期限，催告被上訴人給付租金，則上訴人先後於一○一年五月四日、一○一年六月二十二日分別定期催告被上訴人給付積欠租金，能否謂不生催告之效力？且上訴人於起訴狀已表明兩造間之租約業已終止，而被上訴人於一○一年九月七日收受起訴狀繕本，有送達證書可稽（一審卷六五頁），斯時被上訴人遲付租金之總額，經以押租金扣抵後，是否未達二個月之租金額？能否認上訴人以起訴狀繕本之送達為終止租約之意思表示？況上訴人係主張以押租金扣抵一○一年七、八月之違約金共六十萬九千一百六十八元，請求其餘額十七萬四千零四十八元（一審卷一一六頁反面），原審既認兩造間之租賃關係並未終止，自應先將押租金扣抵一○一年六月三十日以前之欠租，則能否認上訴人無請求一○一年七、八月租金之意？均有再進一步釐清之必要。原審未遑詳加調查審認，徒以上述理由為上訴人不利之判決，尚嫌速斷。上訴論旨，執以指摘原判決不利於己部分不當，求予廢棄，非無理由。據上論結，本件上訴為有理由"

old = case_set.find_one(condition)
case_set.update_one(old,{"$set": {"state": 1}})

predict_x = made_matrix(string)
y_pred = predict_vic(predict_x)
neighborlst = knn_five_judgement(predict_x)

print(neighborlst)
#for i in neighborlst:
 #   print(i)
print(y_pred)

neighborlstu = []
for i in neighborlst:
    j_id = judgement_set.find_one({"j_id": {"$regex": i.split("_")[1].split(".")[0]}}).get("_id")
    neighborlstu.append(j_id)


old_one = {"id": case_set.find_one(condition).get("id")}
new_one = {"$set": { "result": y_pred[0], "neighborlst": neighborlstu, "state":2}}
case_set.update_one(old_one, new_one)
conn.close()
print("finish")

#old_one = {"id": case_set.find_one(condition).get("id")}
#new_one = {"$set": { "neighborlst": neighborlst}}
#case_set.update_one(old_one, new_one)

