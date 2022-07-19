import streamlit as st
import main3
import librosa

# Sección de introducción
st.title("Klasifikasi Genre Musik Menggunakan CNN")
st.write(
    """
    Muhammad Nafis (175150200111008)
    """
)

file_uploader = st.sidebar.file_uploader(label="upload audio", type=".wav")

if file_uploader is not None:
    st.write(file_uploader)
    if st.button("Cari Genre"):
        main3.ubah_spect(file_uploader)
        st.write(
            """
            Prediksi Genre :
            """
        )
        st.write(main3.prediksi())


if file_uploader is not None:
    # y,sr = librosa.load(file_uploader)
    suara = file_uploader.read()
    st.audio(suara)
    # if st.button("Cari Genre"):
    #     main3.ubah_spect(y,sr)
    #     st.write(
    #         """
    #         Prediksi Genre :
    #         """
    #     )
    #     st.write(main3.prediksi())

# if file_uploader is not None:
#     st.write(file_uploader)
#     if st.button("Cari Genre"):
#         main3.ubah_spect(file_uploader)
#         st.write(
#             """
#             Prediksi Genre :
#             """
#         )
#         st.write(main3.prediksi())


   # main3.ubah_spect(file_uploader)
   # st.write(
   #     """
   #     Prediksi Genre :
   #     """
   # )
   # st.write(main3.prediksi())



