"""
为金融反欺诈系统实现规则的自动优化，以提高对金融交易数据判定的准确率
我们会根据获得欺诈交易的历史数据，使用遗传算法生成最优规则，然后将这些规则应用于新的交易数据
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

historical_data = pd.read_csv('historical_data.csv')

# 由于我们没有实际的金融交易数据，所以这里假设欺诈预测函数，它依赖于两个规则
def fraud_prediction(data, rule1, rule2):
    # 在实际情况下，需要使用更复杂的模型
    predictions = (data['feature1'] > rule1) & (data['feature2'] < rule2)
    return predictions

# 因为想要优化的是预测的准确率，所以需要创建一个适应度函数
# 适应度函数将规则作为输入，并返回预测的准确率
def fitness(individual):
    rule1, rule2 = individual
    predictions = fraud_prediction(historical_data, rule1, rule2)
    accuracy = accuracy_score(historical_data['is_fraud'], predictions)
    return accuracy,

# 创建问题类型
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 最大化适应度（预测准确率）
creator.create("Individual", list, fitness=creator.FitnessMax)

# 初始化规则
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.random)  # 规则是0-1之间的浮点数
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)  # 两个规则
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册遗传算法操作
toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# 初始化种群
pop = toolbox.population(n=50)

# 执行遗传算法
result = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, verbose=True)

best_individual = tools.selBest(result[0], 1)[0]
print("Best individual is %s, %s" % (best_individual, best_individual.fitness.values))