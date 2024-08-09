import streamlit as st
import pandas as pd
import datetime
from pymongo import MongoClient

# 如果连接不存在，创建连接
if "mongo_client" not in st.session_state:
    uri = st.secrets['URI']
    st.session_state.mongo_client = MongoClient(uri)

# 获取数据库和集合
db = st.session_state.mongo_client["robus_database"]
collection = db["robus_collection"]


# 定义多用户隔离功能，支持分页和排序
def get_user_data(user, page_num, page_size=10):
    skip = (page_num - 1) * page_size
    cursor = collection.find({'user': user}).sort('date', -1).skip(skip).limit(page_size)
    return list(cursor)


# 获取用户数据总数
def get_user_data_count(user):
    return collection.count_documents({'user': user})


# 自定义CSS样式
st.markdown("""
<style>
    .stDataFrame {
        font-size: 12px;
        width: 100% !important;
    }
    .data-container {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("数据统计")
user = st.sidebar.text_input("输入用户名以查看统计")

if user:
    # 获取用户数据总数
    total_records = get_user_data_count(user)

    if total_records > 0:
        # 计算总页数
        total_pages = (total_records + 9) // 10  # 向上取整

        # 使用标签页来展示不同的统计视图
        tab1, tab2, tab3, tab4 = st.tabs(["按日统计", "按月统计", "按年统计", "全部数据"])

        # 获取所有数据用于统计
        all_data = list(collection.find({'user': user}))
        df = pd.DataFrame(all_data)
        df['_id'] = df['_id'].astype(str)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        with tab1:
            st.header("按日统计")
            daily_expense = df.resample('D').sum(numeric_only=True)['amount']
            st.line_chart(daily_expense)

        with tab2:
            st.header("按月统计")
            monthly_expense = df.resample('M').sum(numeric_only=True)['amount']
            st.line_chart(monthly_expense)

        with tab3:
            st.header("按年统计")
            yearly_expense = df.resample('Y').sum(numeric_only=True)['amount']
            st.line_chart(yearly_expense)

        with tab4:
            st.header("全部数据")

            # 初始化当前页码
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1

            # 创建一个容器来包含数据和分页按钮
            with st.container():
                data_container = st.container()

                with data_container:
                    # 显示当前页码和总页数
                    st.write(f"当前第 {st.session_state.current_page} 页，共 {total_pages} 页")

                    # 获取当前页的数据
                    page_data = get_user_data(user, st.session_state.current_page)
                    if page_data:
                        df_page = pd.DataFrame(page_data)
                        df_page['_id'] = df_page['_id'].astype(str)
                        df_page['date'] = pd.to_datetime(df_page['date'])

                        # 调整列的顺序和显示
                        columns_to_display = ['date', 'amount', 'category', 'note'] + [col for col in
                                                                                       df_page.columns if
                                                                                       col not in ['_id', 'user',
                                                                                                   'date', 'amount',
                                                                                                   'category',
                                                                                                   'note']]

                        # 应用样式并显示表格，设置高度和宽度自适应
                        st.dataframe(df_page[columns_to_display].style.set_table_styles(
                            [{
                                'selector': 'table',
                                'props': [('width', '100%')]
                            }]
                        ), use_container_width=True)
                    else:
                        st.write("该页无数据。")

                    # 添加分页按钮
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("上一页") and st.session_state.current_page > 1:
                            st.session_state.current_page -= 1
                            st.experimental_rerun()

                    with col2:
                        st.write(f"第 {st.session_state.current_page} 页")

                    with col3:
                        if st.button("下一页") and st.session_state.current_page < total_pages:
                            st.session_state.current_page += 1
                            st.experimental_rerun()

                # 为容器添加CSS类
                st.markdown(f"""
                    <style>
                        div.stContainer > div:nth-child({data_container.id}) {{
                            border: 1px solid #ccc;
                            border-radius: 5px;
                            padding: 10px;
                            margin-bottom: 10px;
                        }}
                    </style>
                """, unsafe_allow_html=True)
    else:
        st.write("该用户暂无数据。")
