import streamlit as st
import numpy as np
import pandas as pd
import pickle
import surprise

                    
######### Data ##############
######### Sản phẩm
df_sanpham = pd.read_csv('San_pham.csv')
######### Đánh giá
df_danhgia = pd.read_csv('Danh_gia.csv')[['ma_san_pham', 'ma_khach_hang', 'so_sao']]
df_danhgia = df_danhgia.merge(df_sanpham[['diem_trung_binh', 'ma_san_pham']], left_on='ma_san_pham', right_on='ma_san_pham', how='left')

######### Khách hàng
df_mkh = pd.read_csv('Khach_hang.csv').sample(20, random_state=42) # MKH dùng làm thông tin đăng nhập

# Tạo danh sách mã khách hàng
lst_account = list(df_mkh['ma_khach_hang'])
new_account = ['[Mới]abc', "[Mới]eijs", "[Mới]pemcjj"]
lst_account.extend(new_account)

# Lọc sản phẩm có trên 100 bình luận và điểm trung bình >=4
sp_mua_nhieu_nhat = df_danhgia.value_counts(subset='ma_san_pham')
sp_mua_nhieu_nhat = sp_mua_nhieu_nhat[sp_mua_nhieu_nhat > 100].index.to_list()
sp_mua_nhieu_nhat = df_sanpham.loc[(df_sanpham['diem_trung_binh'] >= 4) & df_sanpham['ma_san_pham'].isin(sp_mua_nhieu_nhat), ['ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh']].reset_index(drop=True)
######### CB dataframe
cb_df = pd.read_csv('gensim_df.csv', index_col=0)

######### Model #############
with open('knnbasic_cf.pkl', 'rb') as f:
    cf_model = pickle.load(f) # model CF

######## Function ###########
def user_rcmmt(df_danhgia, userId):
    # danh sách MSP mà KH đã đánh giá
    lst_msp = df_danhgia.loc[(df_danhgia['ma_khach_hang'] == userId) & (df_danhgia['so_sao'] >=4)]['ma_san_pham'].unique()
    
    # lọc sản phẩm theo MSP
    df_score = df_sanpham.loc[df_sanpham['ma_san_pham'].isin(lst_msp), ['ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh']]
    
    # CF dataframe
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(lambda x: cf_model.predict(userId, x).est) # est: get EstimateScore
    df_score = df_score.drop_duplicates()
    df_score = df_score.sort_values(by=['EstimateScore'], ascending=False)
    return df_score


def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
    # Get the index of the product that matches the ma_san_pham
    matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    if not matching_indices:
        print(f"No product found with ID: {ma_san_pham}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim.loc[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar products (Ignoring the product itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top n most similar products as a DataFrame
    return df.iloc[product_indices]

def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:   
                    st.write(product['ten_san_pham'])
                    st.write(product['diem_trung_binh'], ':star:')          
                    expander = st.expander(f"Mô tả")
                    product_description = product['mo_ta']
                    truncated_description = ' '.join(product_description.split()[:50]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")
                    
def reset_state(): 
    if 'selected_ma_san_pham' in st.session_state: 
        del st.session_state.selected_idx
        
######## Banner #############
######## Sidebar ############
st.sidebar.image('hasaki_logo.png')
genre = st.sidebar.radio(label= ' ', options=["Giới thiệu", "Giao diện"], key='Giới thiệu')

# Phần giới thiệu
if genre == "Giới thiệu":
    st.image('MY.jpg')
    st.divider()
    st.header('BUSINESS UNDERSTANDING')
    st.write('Hasaki.vn - một hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với mạng lưới rộng khắp Việt Nam.\
        \nHasaki cho phép khách hàng lựa chọn và mua sắm mỹ phẩm. Đọc và để lại các đánh giá/nhận xét về sản phẩm, dịch vụ.\
        \nTừ dữ liệu khách hàng, cần cung cấp những gợi ý phù hợp với nhu cầu của họ nhằm đạt được sự hài lòng khách hàng cao hơn và tăng doanh thu.')
    st.markdown('# Xác định vấn đề')
    st.write('Từ những dữ liệu khách hàng để lại khi mua sắm,\
        xây dựng một hệ thống đề xuất những sản phẩm phù hợp cho với nhu cầu, sở thích người dùng.\
        Giúp tăng trãi nghiệm cho người dùng và giúp\
         tối ưu hóa doanh thu và tăng tỷ lệ chuyển đổi bằng cách giới thiệu các sản phẩm có khả năng cao được mua.')
    st.markdown('# Phương pháp')
    st.markdown("**Recommender system**")
    st.write('Recommendation system là hệ thống giúp đề xuất những sản phẩm, dịch vụ, ... phù hợp cho người dùng,\
        dựa trên sở thích và hành vi của họ.')
    st.markdown('**Hệ thống đề xuất được sử dụng**\
        \n1. Content-Based Filtering: gợi ý dựa trên nội dung khách hàng đang xem. Dùng cho khách hàng có hoặc không có đăng nhập,\
            hoặc người dùng đăng nhập nhưng là khách hàng mới, chưa có lịch sử mua hàng.\
        \n2. Collaborative Filtering: gợi ý dựa trên sản phẩm tương tự giữa những người dùng. \
            Được sử dụng cho những khách hàng cũ và có tài khoản đăng nhập.')
    st.divider()
    st.write('Hãy chọn vào phần "Giao diện" để trãi nghiệm nhé!')
    st.sidebar.title("Recommender System")
    st.sidebar.markdown('**Giáo viên hướng dẫn:** Mrs. Thùy Phương')
    st.sidebar.markdown('**Người thực hiện:** Lê Gia Linh & Phạm Tường Hân')
else: 
    
    st.image('https://hotro.hasaki.vn/files/banner-he-thong-cua-hang-hasaki-30102024.jpg')
    # tên đăng nhập
    choice = st.sidebar.selectbox('Tên đăng nhập', lst_account, index=None) # sign in
    # Mật khẩu
    st.sidebar.text_input('Mật khẩu', type='password', value=choice) 
    st.sidebar.markdown('*Ghi chú: mã khách hàng mới nằm ở cuối danh sách chọn.\
        Giao diện người dùng mới tương tự như giao diện khách hàng không đăng nhập*')
        
    ######## Đề xuất dựa trên người dùng
    if (choice is None):
        st.write('Trạng thái: Chưa đăng nhập')
        st.write('Bạn có thể chọn và xem sản phẩm ở trạng thái này')
        st.divider()
        st.markdown('**SẢN PHẨM CÓ SỐ LƯỢT BÌNH LUẬN NHIỀU NHẤT**')
        # Đề xuất dựa trên sản phẩm có lượt bình luận nhiều nhất
        display_recommended_products(sp_mua_nhieu_nhat.head(), cols=len(sp_mua_nhieu_nhat.head()))

    elif choice in new_account:
        st.write('Trạng thái: Đã đăng nhập')
        st.write('Chào người dùng mới!')
        st.divider()
        st.markdown('**SẢN PHẨM CÓ SỐ LƯỢT BÌNH LUẬN NHIỀU NHẤT**')
        # Đề xuất dựa trên sản phẩm có lượt bình luận nhiều nhất
        display_recommended_products(sp_mua_nhieu_nhat.head(), cols=len(sp_mua_nhieu_nhat.head()))
    else:
        st.write('Trạng thái: Đã đăng nhập')
        st.divider()       
        st.markdown('**SẢN PHẨM TƯƠNG TỰ NHỮNG SẢN PHẨM BẠN ĐÃ MUA**')
        de_xuat = user_rcmmt(df_danhgia, choice)
        display_recommended_products(de_xuat.head(), cols=len(de_xuat.head()))
        
    ####### Danh sách sản phẩm trong selectbox
    product_options = [(row.ma_san_pham, row.ten_san_pham)
                    for index, row in 
                    df_sanpham.iterrows()]
        
    ##### Hộp chọn
    selected_product = st.selectbox('',options=product_options,
                        placeholder='Chọn sản phẩm', 
                        index=None, format_func=lambda x: f'{x[0]} - {x[1]}', 
                        label_visibility='hidden',)
    if selected_product is None:
        st.session_state.selected_ma_san_pham = ''
    else:
        st.session_state.selected_ma_san_pham = selected_product[0]
        
        if st.session_state.selected_ma_san_pham:
            st.write('KẾT QUẢ TÌM KIẾM')
            st.write(selected_product[0])
            st.header(selected_product[1])
            selected_product = df_sanpham.loc[df_sanpham['ma_san_pham'] == st.session_state.selected_ma_san_pham]
            col1, col2 = st.columns(2)
            col1.markdown('**Giá: {:,.0f} VND**'.format(selected_product['gia_ban'].values[0]))
            col2.markdown('**Số sao: {:,.0f} :star:**'.format(selected_product['diem_trung_binh'].values[0]))
            
            tab1, tab2 = st.tabs(["Mô tả", "Phân loại"])
            with tab1:
                st.write(selected_product['mo_ta'].values[0])
            with tab2:
                st.write(selected_product['phan_loai'].values[0])
            
            
            st.divider()
            ##### 
            st.markdown("**SẢN PHẨM TƯƠNG TỰ**")
            container = st.container(border=True)
            # lấy ra các sản phẩm liên quan
            sp_lienquan = get_recommendations(df = df_sanpham, ma_san_pham=st.session_state.selected_ma_san_pham, cosine_sim=cb_df, nums=3)
            display_recommended_products(sp_lienquan, cols=len(sp_lienquan))
        
