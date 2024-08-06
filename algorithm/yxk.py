import os
from surprise import Dataset, Reader, KNNWithMeans, KNNWithZScore
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# 根据用户的历史评分为其推荐产品。为特定用户推荐的产品列表，帮助他们发现感兴趣的产品
# 推荐与用户已评分物品相似的其他物品
# 它的核心思想是，如果一个用户对某些物品给出了高分，那么这些物品应该与其他类似的物品相似。
# 这种方法不直接利用用户的特征或物品的特征，而是依赖于物品之间的相似性来进行推荐。
# 工作原理
# 计算物品相似度：
#
# 通过分析所有用户对物品的评分，计算物品之间的相似度。
# 例如，如果两个物品被许多相同的用户评分，则这两个物品被认为是相似的。
# 生成推荐：
#
# 对于某个用户，根据他们已评分的物品，找到与这些物品相似的其他物品。
# 将这些相似的物品推荐给用户。

# 包含用户ID、产品ID及其评分。评分可以表示用户对某个产品的兴趣程度。假设我们有以下用户历史订单及产品特征：
# 用户1：户外、亲子
# 用户2：摄影、户外
# 用户3：亲子、摄影
#
# 产品1：户外
# 产品2：亲子
# 产品3：摄影
# 产品4：户外、亲子
# # # # # # # # # # # # # # # #
# 我们假设评分为用户对产品的兴趣程度（1到5之间）。下面是示例数据：
# 用户1 对 产品1 评分 4（用户1喜欢户外）
# 用户1 对 产品2 评分 5（用户1喜欢亲子）
# 用户2 对 产品1 评分 3（用户2比较喜欢户外）
# 用户2 对 产品3 评分 4（用户2喜欢摄影）
# 用户3 对 产品2 评分 5（用户3喜欢亲子）
# 用户3 对 产品3 评分 3（用户3比较喜欢摄影）


# 物品A和物品B的相似度：
#
# 用户1对A评分5，对B评分3。
# 用户2对A评分4，对B评分3。
# 用户3对A评分2，没有对B评分。
# 使用这些评分计算A和B的相似度，假设计算结果为0.9。
# # # # # # # # # # # # # # # #

# 定义文件格式
reader = Reader(line_format='user item rating', sep='::')

# 从文件加载数据
rating_file = os.path.join('yxk', 'ratings_prod_3w.dat')
data = Dataset.load_from_file(rating_file, reader=reader)

# 拆分数据集
trainset, testset = train_test_split(data, test_size=0.25)

# 使用基于物品的协同过滤算法
# KNNWithMeans：这个算法在计算相似度时，会考虑评分的平均值。也就是说，它会减去用户或物品的平均评分，然后再计算相似度。这有助于减少评分的偏差，从而使得推荐更加准确。
# KNNBasic：这是最基本的KNN算法，直接计算用户或物品之间的相似度，而不考虑评分的平均值。这种方法可能会受到评分偏差的影响。
# 数据偏差大：如果你的评分数据存在较大的偏差（例如，某些用户总是给出高分或低分），可以优先考虑 KNNWithMeans，因为它能够有效消除这种偏差。
# 评分分布广：如果你的评分数据分布较广，且不同用户或物品的评分方差差异较大，可以考虑 KNNWithZScore，因为它能够对评分进行标准化，使得相似度计算更加稳定和准确。
# 参数 sim_options={'user_based': False} 表示使用基于物品的相似度计算，而不是基于用户的相似度计算
# algo = KNNWithMeans(sim_options={'user_based': False})
# 'name': 'cosine'
# 'name': 'pearson',
# 'name': 'jaccard',
# 'name': 'manhattan'
# 'name': 'euclidean'
algo = KNNWithZScore(sim_options={'name': 'pearson', 'user_based': False})

# 使用矩阵分解算法 SVD
# algo = SVD()

# 训练模型
algo.fit(trainset)

# 进行预测
predictions = algo.test(testset)

# 计算准确性，较低的 RMSE 表示预测更准确。
rmse = accuracy.rmse(predictions)


# 计算精确率和召回率
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """Return precision and recall at k metrics for each user."""
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4.0)

# 计算平均精确率和召回率
average_precision = sum(prec for prec in precisions.values()) / len(precisions)
average_recall = sum(rec for rec in recalls.values()) / len(recalls)

# 最好是 ： RMSE 低、Precision 高、Recall 高
print(f"RMSE: {rmse}")
# 平均精确率 (Precision): 0.816 表示推荐的物品中，有 81.6% 是用户实际喜欢的。
print(f"Average Precision: {average_precision}")
# 平均召回率 (Recall): 0.472 表示用户实际喜欢的物品中，有 47.2% 被成功推荐。
print(f"Average Recall: {average_recall}")


# 均方差（MSD）作为相似度度量
# 推荐函数
# 这个函数的目的是为特定用户推荐他们尚未评分的产品
# 1. 获取所有产品及用户已评分的产品。
# 2. 识别用户未评分的产品。
# 3. 为这些未评分的产品预测评分。
# 4. 根据预测评分排序，并返回评分最高的前 top_n 个产品。
def recommend_for_user(user_id, top_n=10):
    # 获取用户未评分的所有项目
    trainset = algo.trainset
    # 获取所有项目的内部ID
    all_items = set(trainset.all_items())
    # trainset.to_inner_uid(user_id) 是将原始用户ID转换为内部用户ID的函数。surprise 库内部会将用户ID和项目ID映射为内部的连续整数ID，以便优化计算。
    # 原始用户ID 是用户在数据集中使用的ID，比如 '1'。
    # 内部用户ID 是 surprise 库为用户分配的内部ID，通常是一个整数，比如 0。
    # trainset.ur 是一个字典，其中键是内部用户ID，值是一个列表，列表中的元素是用户评分过的项目及其评分。获取该用户评分过的所有项目及其评分
    # (j for (j, _) in user_ratings) 这是一个生成器表达式，用于遍历 user_ratings 列表，并提取每个元组中的项目ID（内部ID）。
    # j 是项目的内部ID，_ 是评分，我们在这里不需要评分，所以用 _ 忽略。
    # set(...)将生成器表达式转换为集合，去除重复的项目ID，确保每个项目ID只出现一次。
    # 获取指定用户已经评分的所有项目（产品）的内部ID，并将其存储在一个集合（set）中
    user_rated_items = set(j for (j, _) in trainset.ur[trainset.to_inner_uid(user_id)])
    # 计算用户未评分的项目
    items_to_predict = all_items - user_rated_items

    # 预测评分，预测用户未评分项目的评分
    # trainset.to_raw_iid(item) 将项目的内部ID转换为原始ID，因为 algo.predict 需要原始ID。
    # algo.predict 预测用户 user_id 对某个项目的评分
    # 预测用户 user_id 对项目 item 的评分。
    predictions = [algo.predict(user_id, trainset.to_raw_iid(item)) for item in items_to_predict]
    # algo.predict 返回一个包含预测结果的对象，这个对象通常包括以下属性：
    # uid：用户ID
    # iid：项目ID
    # r_ui：实际评分（如果有的话）
    # est：预测评分
    # details：预测的其他详细信息（如置信区间）

    # 过滤出符合特征的产品，例如筛选出“亲子”类产品
    filtered_predictions = [pred for pred in predictions if is_family_friendly(pred.iid)]

    # 这里 uid 是用户ID，iid 是项目ID，est 是预测评分。
    # 排序的关键是每个预测对象的 est 属性（即预测评分）。我们使用 lambda 函数来提取 est 属性
    # 根据预测评分排序并返回评分最高的前 top_n 个项目
    recommendations = sorted(filtered_predictions, key=lambda x: x.est, reverse=True)[:top_n]

    # 假设predictions的值为：
    # predictions = [
    #     Prediction(uid='1', iid='10', r_ui=None, est=4.2, details={'was_impossible': False}),
    #     Prediction(uid='1', iid='20', r_ui=None, est=3.8, details={'was_impossible': False}),
    #     Prediction(uid='1', iid='30', r_ui=None, est=4.5, details={'was_impossible': False}),
    #     Prediction(uid='1', iid='40', r_ui=None, est=4.0, details={'was_impossible': False}),
    #     Prediction(uid='1', iid='50', r_ui=None, est=4.7, details={'was_impossible': False})
    # ]

    return recommendations


def is_family_friendly(item_id):
    # 这里需要一个函数来检查产品是否符合“亲子”类特征
    # 假设我们有一个特征字典或者数据结构来检查特征
    return item_id
    # family_friendly_items = {'10588', '13367'}  # 示例数据
    # return item_id in family_friendly_items


# 获取与特定产品相似的产品
def get_similar_items(item_id, top_n=10):
    inner_id = algo.trainset.to_inner_iid(item_id)
    neighbors = algo.get_neighbors(inner_id, k=top_n)
    neighbors = [algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors]

    return neighbors


# 示例：为用户1推荐产品
user_id = '1087475'
recommendations = recommend_for_user(user_id)
print(f"Recommendations for user {user_id}:")
for rec in recommendations:
    print(f"Item {rec.iid} with predicted rating {rec.est}")

# 如果用户1对物品A、B、C的评分较高，算法会找出与A、B、C相似的物品，并推荐这些物品。
# 这些推荐物品可能是用户未评分的物品，但与用户已评分的物品具有高相似度。

# 示例：获取与产品88888相似的产品
# similar_items = get_similar_items('88888')
# print("Items similar to 88888:")
# for item in similar_items:
#     print(f"Item {item}")


# In[1]
print("Hello, world!")

# In[2]
a = 10
b = 20
c = a + b
print(c)
# In[]
print(1)
