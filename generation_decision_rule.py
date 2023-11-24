"""
为金融反欺诈系统生成用于判断交易数据是否为欺诈交易的决策规则
基于随机森林模型
根据规则可以对交易数据进行判定，从而实现欺诈交易的自动识别，给出认为是欺诈的置信度信息，以及认为接下来需要为交易账户进行的操作建议。

主要步骤如下：
1. 加载数据：使用pandas的`read_csv`方法从CSV文件中读取数据。
2. 数据预处理：将日期字段转换为时间戳，然后删除原日期字段。
3. 特征和标签分离：'is_fraud'是目标标签，我们需要预测的值，其余的是特征。
4. 特征缩放：使用`StandardScaler`对特征进行缩放，使其均值为0，标准差为1。
5. 分离训练集和测试集：使用`train_test_split`将数据集分为80％的训练集和20％的测试集。
6. 特征选择：使用随机森林分类器的`SelectFromModel`方法，选择重要的特征。
7. 训练模型：使用训练集训练随机森林模型。
8. 预测：使用训练好的模型对测试集进行预测。
9. 输出分类报告和混淆矩阵：使用`classification_report`和`confusion_matrix`从不同角度评估模型性能。
10. 输出置信度信息：输出模型预测每个样本为欺诈的概率。
11. 超参数优化：使用网格搜索（`GridSearchCV`）寻找最佳的超参数组合。
12. 使用最佳参数重新训练模型，进行预测，并输出相关结果。
在这个过程中，我们会得到两个模型（原始的随机森林模型和经过超参数优化的随机森林模型），并可以比较两者的性能。
同时，我们还可以观察到模型预测的欺诈概率，这非常有助于理解模型的决策过程。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# 加载数据集
data = pd.read_csv("financial_data.csv")

# 数据预处理
data['date'] = pd.to_datetime(data['date'])
data['timestamp'] = data['date'].values.astype(np.int64) // 10 ** 9
data = data.drop(['date'], axis=1)

# 特征和标签分离
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# 特征缩放
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分离训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 特征选择（基于重要性）
selector = SelectFromModel(rf)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# 训练模型
rf.fit(X_train, y_train)

# 使用模型进行预测
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)

# 输出分类结果报告
print(classification_report(y_test, y_pred))

# 输出混淆矩阵
print(confusion_matrix(y_test, y_pred))

# 输出置信度信息
print("Predicted fraud probabilities:", y_pred_proba[:, 1])

# 超参数优化
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rf.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters: ", CV_rf.best_params_)

# 使用最佳参数重新训练模型
rf_best = RandomForestClassifier(n_estimators=CV_rf.best_params_['n_estimators'],
                                 max_features=CV_rf.best_params_['max_features'],
                                 max_depth=CV_rf.best_params_['max_depth'],
                                 criterion=CV_rf.best_params_['criterion'],
                                 random_state=42)
rf_best.fit(X_train, y_train)

# 使用最佳模型进行预测
y_pred_best = rf_best.predict(X_test)
y_pred_proba_best = rf_best.predict_proba(X_test)

# 输出最佳模型的分类结果报告和混淆矩阵
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))

# 输出最佳模型的置信度信息
print("Predicted fraud probabilities (best model):", y_pred_proba_best[:, 1])