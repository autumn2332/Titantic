import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


#导入数据
#训练数据集
train = pd.read_csv("../CSV/train.csv")
# 测试数据集
test = pd.read_csv("../CSV/test.csv")
# 训练数据集有891条数据
# print ('训练数据集:',train.shape,'测试数据集:',test.shape)
# rowNum_train=train.shape[0]
# rowNum_test=test.shape[0]
# print('kaggle训练数据集有多少行数据：',rowNum_train,
#      ',kaggle测试数据集有多少行数据：',rowNum_test,)
#合并数据集，方便同时对两个数据集进行清洗
full = train.append( test , ignore_index = True )#使用append进行纵向堆叠


#年龄(Age)
train["Age"]=train["Age"].fillna(full["Age"].mean())
full['Age']=full['Age'].fillna(full['Age'].mean())

# 对训练集按年龄分类
children = train[train["Age"]<12]
juvenile = train[(train["Age"]>=12)&(train["Age"]<18)]
adult =train[(train["Age"]>18)&(train["Age"]<=65)]
agedness=train[train["Age"]>=65]
# 各年龄段生存人数
children_survived_sum=children["Survived"].sum()
juvenile_survived_sum=juvenile["Survived"].sum()
adult_survived_sum=adult["Survived"].sum()
agedness_survived_sum=agedness["Survived"].sum()
# 各年龄段生存率
children_survived_rate=children["Survived"].mean()
juvenile_survived_rate=juvenile["Survived"].mean()
adult_survived_rate=adult["Survived"].mean()
agedness_survived_rate=agedness["Survived"].mean()
# 绘制年龄的饼状图和柱状图
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
x=["children","juvenile","adult","agedness"]
y=[children_survived_sum,juvenile_survived_sum,adult_survived_sum,agedness_survived_sum]
width=0.5
rect=ax1.bar(x,y,width,color='b')
for x_pos,y_pos in enumerate(y):
    plt.text(x_pos,y_pos+5,y_pos,ha='center',fontsize=16)
plt.title("各年龄段存活人数")
ax2=fig.add_subplot(1,2,2)
percentage=[children_survived_rate,juvenile_survived_rate,adult_survived_rate,agedness_survived_rate]
ax2.pie(x=percentage,labels=x,autopct='%1.0f%%')
plt.show()

#船票价格(Fare)
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
full['Embarked'] = full['Embarked'].fillna( 'S' )

# 由于船票价格和船舱等级所表达的含义是一致的，所以只绘制船舱等级的柱状图和饼状图
train["Embarked"] = train["Embarked"].fillna('S')
Embarked_df=pd.DataFrame()
Embarked_df=train[["Pclass","Survived"]].groupby(["Pclass"]).count()

fig = plt.figure()
ax1=fig.add_subplot(1,2,1)
width=0.5
ax1.bar([1,2,3],Embarked_df["Survived"],width)
plt.title("船舱各等级存活人数")
ax2=fig.add_subplot(1,2,2)
ax2.pie(Embarked_df["Survived"],labels=[1,2,3],autopct="%1.0f%%")
plt.show()

full['Cabin'] = full['Cabin'].fillna( 'U' )
sex_mapDict = {'male':1,'female':0}

#map函数：对Series每个数据应用自定义的函数计算
full['Sex']=full['Sex'].map(sex_mapDict)
survived_df=train[train["Survived"]==1]

# 对Sex属性进行分析
survived_male= survived_df["Sex"][survived_df["Sex"]=="male"].count()
survived_female=survived_df["Sex"][survived_df["Sex"]=="female"].count()
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
width=0.5
ax1.bar(["male","female"],[survived_male,survived_female])
plt.title("男女性存活数")
ax2=fig.add_subplot(1,2,2)
ax2.pie([survived_male,survived_female],labels=["male","female"],autopct="%1.0f%%")
plt.show()


embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies(full['Embarked'] , prefix='Embarked' )
full = pd.concat([full,embarkedDf],axis=1)
full.drop('Embarked',axis=1,inplace=True)
pclassDf = pd.DataFrame()

#使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies(full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)

#删掉客舱等级（Pclass）这一列
full.drop('Pclass',axis=1,inplace=True)
def getTitle(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3

titleDf = pd.DataFrame()
train_title_df =pd.DataFrame()
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = full['Name'].map(getTitle)
train_title_df["Title"] = train["Name"].map(getTitle)
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)
train_title_df["Title"]=train_title_df["Title"].map(title_mapDict)
train=pd.concat([train,train_title_df],axis=1)

# 将各个阶层分类
Royalty =train[train["Title"]=="Royalty"]
Officer =train[train["Title"]=="Officer"]
Mrs=train[train["Title"]=="Mrs"]
Miss=train[train["Title"]=="Miss"]
Master=train[train["Title"]=="Master"]
Mr = train[train["Title"]=="Mr"]
# 统计各个阶层总人数
Royalty_count_rate =Royalty["Survived"].mean()
Officer_count_rate =Officer["Survived"].mean()
Mrs_count_rate=Mrs["Survived"].mean()
Miss_count_rate=Miss["Survived"].mean()
Master_count_rate=Master["Survived"].mean()
Mr_count_rate=Mr["Survived"].mean()
total_count_rate=[Royalty_count_rate,
                  Officer_count_rate,
                  Master_count_rate,
                  Miss_count_rate,
                  Mrs_count_rate,
                  Mr_count_rate]
total_count=pd.Series()
total_count=Series(total_count_rate)
total_count=total_count.fillna(0)
print(total_count)

# 统计各个阶层存活人数
Royalty_survived_sum = Royalty["Survived"].sum()
Officer_survived_sum = Officer["Survived"].sum()
Master_survived_sum = Master["Survived"].sum()
Miss_survived_sum = Miss["Survived"].sum()
Mrs_survived_sum= Mrs["Survived"].sum()
Mr_survived_sum =Mr["Survived"].sum()
survived_list_sum =[Royalty_survived_sum,
                    Officer_survived_sum,
                    Master_survived_sum,
                    Miss_survived_sum,
                    Mrs_survived_sum,
                    Mr_survived_sum]

print(survived_list_sum)

plt.figure()
plt.pie(total_count,labels=["皇室","官员","教师","小姐","女士","先生"],autopct="%1.0f%%")
plt.title("不同阶层的人的存活率")
plt.show()

#使用get_dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,titleDf],axis=1)

#删掉姓名这一列
full.drop('Name',axis=1,inplace=True)
#存放客舱号信息
cabinDf = pd.DataFrame()

'''
客场号的类别值是首字母，例如：
C85 类别映射为首字母C
'''

full['Cabin'] = full[ 'Cabin' ].map( lambda c : c[0])#客舱号的首字母代表处于哪个，U代表不知道属于哪个船舱

##使用get_dummies进行one-hot编码，列名前缀是Cabin
cabinDf = pd.get_dummies(full['Cabin'] , prefix = 'Cabin' )

#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,cabinDf],axis=1)

#删掉客舱号这一列
full.drop('Cabin',axis=1,inplace=True)
familyDf = pd.DataFrame()

#  对是否有父母进行分析，因为不同代直系亲属和Parch所表达的含义是一样的，所以就不分析了
parch_df=train[train["Parch"]!=0]
no_parch_df=train[train["Parch"]==0]
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
parch_list=[parch_df["Survived"][parch_df["Survived"]==1].count(),parch_df["Survived"][parch_df["Survived"]==0].count()]
ax1.pie(parch_list,labels=["存活","死亡"],autopct="%1.0f%%")
plt.title("有父母存活人数与死亡人数")

ax2=fig.add_subplot(1,2,2)
no_parch_list=[no_parch_df["Survived"][no_parch_df["Survived"]==0].count(),no_parch_df["Survived"][no_parch_df["Survived"]==1].count()]
ax2.pie(no_parch_list,labels=["死亡","存活"],autopct="%1.0f%%")
plt.title("没有父母存活人数与死亡人数")
plt.show()

'''
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己
（因为乘客自己也是家庭成员的一个，所以这里加1）
'''
familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
familyDf[ 'Family_Small' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
full = pd.concat([full,familyDf],axis=1)
full.drop('FamilySize',axis=1,inplace=True)
corrDf = full.corr()
# 打印相关系数
print(corrDf['Survived'].sort_values(ascending=False))

full_X = pd.concat( [titleDf,#头衔
                     pclassDf,#客舱等级
                     familyDf,#家庭大小
                     full['Fare'],#船票价格
                     full['Sex'],#性别
                     full["Age"],#年龄
                    ] , axis=1 )

sourceRow=891
'''
sourceRow是我们在最开始合并数据前知道的，原始数据集有总共有891条数据
从特征集合full_X中提取原始数据集提取前891行数据时，我们要减去1，因为行号是从0开始的。
'''
#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']

#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]

#建立模型用的训练数据集和测试数据集
size=np.arange(0.6,1,0.1)
scorelist=[[],[],[],[],[],[]]
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(source_X ,source_y,train_size=0.8)
print("训练集特征：{0},训练集标签{1}".format(train_X.shape,train_y.shape))
print("训练集特征：{0},训练集标签{1}".format(test_X.shape,test_y.shape))
# 对train_x,test_x进行标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x_std=sc.fit_transform(train_X)
test_x_std=sc.transform(test_X)
#逻辑回归
# 导入相应的包
from sklearn.linear_model import LogisticRegression
# 创建模型
model = LogisticRegression()
# 训练模型
model.fit( train_X , train_y )

LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,
                   max_iter=100,multi_class='ovr',n_jobs=1,penalty='12',random_state=None,
                   solver='liblinear',tol=0.0001,verbose=0,warm_start=False)
# 得出模型正确率
print("模型正确率",model.score(test_x_std, test_y))

pred_x_std=sc.fit_transform(pred_X)
pred_y=model.predict(pred_x_std)

pred_df=pd.DataFrame({'PassengerId':test.PassengerId,'Survived':pred_y})

pred_df['Survived']=pred_df['Survived'].astype('int')
pred_df.to_csv("predict.csv",index=False)


