import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Call st.set_page_config as the very first Streamlit command
st.set_page_config(
    page_title="Aplikasi Prediksi Promosi Jabatan dengan Metode Naive Bayes",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None
)

# Define the sidebar menu
with st.sidebar:
    selected = option_menu("Menu Utama", ["Dashboard", "Visualisasi Data", "Perhitungan"], icons=['house', 'pie-chart'],
     menu_icon="cast", default_index=0)

# Define a function for the "Dashboard" page
def dashboard_page():
    st.markdown("---")
    st.title('Aplikasi Prediksi Promosi Jabatan dengan Metode Naive Bayes')
    st.write("Naive Bayes efektif dalam menangani data kategoris dan atribut yang berbeda jenis. Ini membuatnya berguna untuk prediksi promosi jabatan di mana Anda mungkin memiliki data seperti departemen, pendidikan, jenis kelamin, dan atribut-atribut kategoris lainnya.")
    st.markdown("---")

# Define a function for the "Visualisasi Data" page
def data_visualization_page():
    st.title('')
    
    # Judul Utama
    st.title("Analisis Berkas CSV")
    
    # Unggah berkas
    uploaded_file = st.file_uploader("Unggah berkas CSV", type=["csv"])
    
    if uploaded_file is not None:
        st.write("Berkas berhasil diunggah.")
        
        # Baca berkas CSV ke dalam DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Tampilkan informasi dasar tentang data
        st.subheader("Ikhtisar Data")
        st.write("Jumlah baris:", df.shape[0])
        st.write("Jumlah kolom:", df.shape[1])
        
        # Tampilkan beberapa baris pertama data
        st.subheader("Pratinjau Data")
        st.dataframe(df.head())
        
        # Analisis data dan visualisasi dapat ditambahkan di sini
    
        # Contoh: Gambar diagram batang dari kolom tertentu
        st.subheader("Diagram Batang")
        st.write(" grafik batang yang menampilkan sebaran data dalam kolom yang dipilih.")
        selected_column = st.selectbox("Pilih kolom untuk diagram batang", df.columns, key="bar_chart_selectbox")
        st.bar_chart(df[selected_column])

        # Contoh: Tampilkan statistik ringkas
        st.subheader("Statistik Ringkas")
        st.write(" ringkasan statistik tentang data numerik dalam dataset. Statistik ini mencakup informasi seperti jumlah data, rata-rata, deviasi standar, nilai minimum, dan nilai maksimum.")
        st.write(df.describe(), key="summary_stats")
        
        # Contoh: Visualisasikan Prediksi Naive Bayes
        if st.checkbox("Visualisasi Prediksi Naive Bayes"):
            # Anda dapat membuat prediksi di sini menggunakan model Naive Bayes
            # Sebagai contoh, mari kita asumsikan Anda memiliki daftar prediksi sebagai "naive_bayes_predictions"
            naive_bayes_predictions = [0, 1, 1, 0, 1, 0, 1]  # Contoh prediksi
            
            # Buat DataFrame Pandas untuk menyimpan prediksi
            st.write(" ringkasan statistik tentang data numerik dalam dataset. Statistik ini mencakup informasi seperti jumlah data, rata-rata, deviasi standar, nilai minimum, dan nilai maksimum.")
            prediction_df = pd.DataFrame({'Prediksi': naive_bayes_predictions})
            
            # Gambar diagram batang untuk visualisasi prediksi
            st.write("grafik batang yang menunjukkan hasil prediksi menggunakan model Naive Bayes")
            st.bar_chart(prediction_df)

            # Izinkan pengguna memilih kolom numerik dan tampilkan histogram dari nilainya.
            st.subheader("Histogram employee_id")
            st.write("histogram yang menunjukkan bagaimana data terdistribusi dalam kolom employee_id. Histogram membagi data menjadi beberapa interval atau bin dan menunjukkan seberapa sering data jatuh ke dalam setiap interval.")
            plt.hist(df["employee_id"], bins=20, alpha=0.7, color='b')
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

            # Buat peta panas korelasi untuk visualisasi hubungan antara kolom numerik.
            st.subheader("Peta Panas Korelasi")
            st.write("peta panas yang menggambarkan seberapa erat hubungan antara kolom numerik dalam data. Korelasi positif ditunjukkan oleh warna yang lebih terang, sementara korelasi negatif ditunjukkan oleh warna yang lebih gelap.")
            correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="viridis")
            st.pyplot()

            # Buat plot pencocokan pasangan untuk visualisasi hubungan pasangan kolom numerik.
            st.subheader("Plot Pencocokan Pasangan")
            st.write("kumpulan plot pencocokan pasangan antara kolom numerik dalam data. Ini membantu pengguna untuk melihat hubungan antara setiap pasang atribut.")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            sns.pairplot(df[numeric_cols])
            st.pyplot()

            # Izinkan pengguna memilih kolom numerik dan tampilkan diagram kotak dari nilainya.
            st.subheader("Diagram Kotak")
            st.write("diagram kotak yang menunjukkan distribusi data dalam kolom numerik yang dipilih. Diagram kotak ini menyoroti statistik seperti median, kuartil, dan pencilan.")
            selected_column_box = st.selectbox("Pilih kolom numerik untuk diagram kotak", df.select_dtypes(include=['int64', 'float64']).columns, key="box_selectbox")
            sns.boxplot(data=df, x=selected_column_box)
            st.pyplot()

            # Buat diagram batang dari kolom kategoris untuk visualisasi distribusi kategori.
            st.subheader("Diagram Batang (Count Plot)")
            st.write(" diagram batang yang menunjukkan sebaran kategori dalam kolom kategoris yang dipilih.")
            selected_column_count = st.selectbox("Pilih kolom kategoris untuk diagram batang", df.select_dtypes(include=['object']).columns, key="count_selectbox")
            sns.countplot(data=df, x=selected_column_count)
            st.pyplot()

            # Izinkan pengguna memilih dua kolom numerik dan tampilkan plot pencocokan pasangan dari nilai keduanya.
            st.subheader("Plot Pencocokan (Scatter Plot)")
            st.write(" plot pencocokan pasangan antara dua kolom numerik yang dipilih oleh pengguna. Ini membantu pengguna memahami hubungan antara dua atribut dalam bentuk plot pencocokan pasangan.")
            x_column = st.selectbox("Pilih kolom sumbu X untuk plot pencocokan pasangan", df.select_dtypes(include=['int64', 'float64']).columns, key="x_selectbox")
            y_column = st.selectbox("Pilih kolom sumbu Y untuk plot pencocokan pasangan", df.select_dtypes(include=['int64', 'float64']).columns, key="y_selectbox")
            plt.scatter(df[x_column], df[y_column])
            st.pyplot()

            from wordcloud import WordCloud
            st.subheader("Word Cloud untuk Data Teks")
            selected_text_column = st.selectbox("Pilih kolom teks", df.select_dtypes(include=['object']).columns, key="text_selectbox")
            wordcloud = WordCloud(width=800, height=400).generate(' '.join(df[selected_text_column]))
            st.image(wordcloud.to_image())

            st.subheader("Heatmap Korelasi")
            numerical_columns = df.select_dtypes(include=['int64', 'float64'])
            sns.heatmap(numerical_columns.corr(), annot=True, cmap='coolwarm')
            st.pyplot()

            st.subheader("Diagram Pie")
            selected_pie_column = st.selectbox("Pilih kolom untuk diagram pie", df.select_dtypes(include=['object']).columns, key="pie_selectbox")
            pie_data = df[selected_pie_column].value_counts()
            st.write(pie_data)
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
            st.pyplot()

# Define a function for the "Perhitungan" page
def Perhitungan_page():
    st.title('Perhitungan Prediksi Promosi Jabatan')
    
    # Tambahkan kode visualisasi data di sini
    # Muat model yang telah dilatih sebelumnya
    prom_prediction = pickle.load(open('revisi.sav', 'rb'))
    
    # Bidang masukan untuk pengguna memasukkan data
    employee_id = st.text_input('Masukkan Nomer Induk Karyawan')
    department_options = ["Sales & Marketing","Operations","Technology","Analytics","R&D","Procurement","Finance","HR","Legal"]
    selected_department = st.selectbox('Pilih Departemen', department_options)
    region_options = ["region_7","region_22","region_19","region_23","region_26","region_2","region_20","region_34","region_1","region_4","region_29","region_31","region_15","region_14","region_11","region_5","region_28","region_17","region_13","region_16","region_25","region_10","region_27","region_30","region_12","region_21","region_8","region_32","region_6","region_33","region_24","region_3","region_9","region_18"]
    selected_region = st.selectbox('Pilih Wilayah', region_options)
    education_options = ["Master's & above","Bachelor's","Below Secondary"]
    selected_education = st.selectbox('Pilih Pendidikan', education_options)
    gender_options = ["f","m"]
    selected_gender = st.selectbox('Pilih Jenis Kelamin', gender_options)
    recruitment_channel_options = ["sourcing","other","referred"]
    selected_recruitment_channel = st.selectbox('Pilih Saluran Perekrutan', recruitment_channel_options)
    no_of_trainings = st.text_input('Jumlah Pelatihan')
    age = st.text_input('Usia')
    previous_year_rating = st.text_input('Peringkat Tahun Sebelumnya')
    length_of_service = st.text_input('Lama Kerja')
    awards_won = st.text_input('Penghargaan yang Diterima')
    avg_training_score = st.text_input('Skor Pelatihan Rata-rata')
    
    prom_diagnosis = ''
    
    # Tangani klik tombol untuk membuat prediksi
    if st.button('UJI PREDIKSI PROMOSI'):
        # Ubah masukan pengguna menjadi bilangan bulat jika memungkinkan
        try:
            department = department_options.index(selected_department)  # Menggunakan indeks dalam daftar sebagai representasi bilangan bulat
            region = region_options.index(selected_region)
            education = education_options.index(selected_education)
            gender = gender_options.index(selected_gender)
            recruitment_channel = recruitment_channel_options.index(selected_recruitment_channel)
            no_of_trainings = int(no_of_trainings)
            age = int(age)
            previous_year_rating = int(previous_year_rating)
            length_of_service = int(length_of_service)
            awards_won = int(awards_won)
            avg_training_score = int(avg_training_score)
        except ValueError:
            st.warning("Bidang masukan harus berupa bilangan bulat. Harap masukkan nilai numerik yang valid.")
            return
    
        # Bagian kode Anda untuk membuat prediksi dapat ditempatkan di sini
        # Pastikan Anda tidak mengulangi kode untuk memuat model yang telah ada
    
        # Lakukan prediksi menggunakan model
        prom_prediction = prom_prediction.predict([[department, region, education, gender, recruitment_channel, no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score]])
    
        if prom_prediction[0] == 1:
            prom_diagnosis = 'Naik Jabatan'
        else:
            prom_diagnosis = 'Tidak Naik Jabatan'
    
        # Tampilkan hasil prediksi
        st.success(prom_diagnosis)

# Tergantung pada halaman yang dipilih, tampilkan konten yang sesuai
if selected == "Dashboard":
    dashboard_page()
elif selected == "Visualisasi Data":
    data_visualization_page()
elif selected == "Perhitungan":
    Perhitungan_page()
