import streamlit as st
import datetime
from pymongo import MongoClient

# 如果连接不存在，创建连接
if "mongo_client" not in st.session_state:
    uri = st.secrets['URI']
    st.session_state.mongo_client = MongoClient(uri)

# 获取数据库和集合
db = st.session_state.mongo_client["robus_database"]
collection = db["robus_collection"]


# 定义函数用于保存数据到 MongoDB
def save_data(user, date, amount, category, note, custom_keys):
    record = {
        'user': user,
        'date': datetime.datetime.combine(date, datetime.datetime.min.time()),
        'amount': amount,
        'category': category,
        'note': note,
        **custom_keys
    }
    collection.insert_one(record)


# 定义多用户隔离功能，支持分页和排序
def get_user_data(user, page_num, page_size=10):
    skip = (page_num - 1) * page_size
    cursor = collection.find({'user': user}).sort('date', -1).skip(skip).limit(page_size)
    return list(cursor)


# 新增数据页面
st.title("新增数据")
user = st.text_input("用户名")
date = st.date_input("日期", datetime.date.today())
amount = st.number_input("金额", min_value=0.0, step=0.01)
category = st.selectbox("类别", ["食物", "交通", "娱乐", "其他"])
note = st.text_area("备注")

# 自定义 key
custom_keys = {}
st.write("自定义 key:")
key_count = st.number_input("自定义 key 数量", min_value=0, max_value=10, step=1)
for i in range(int(key_count)):
    key = st.text_input(f"Key {i + 1} 名称")
    value = st.text_input(f"Key {i + 1} 值")
    if key and value:
        custom_keys[key] = value

if st.button("保存"):
    if user:
        save_data(user, date, amount, category, note, custom_keys)
        st.success("数据已保存!")
    else:
        st.error("请填写用户名。")
