import streamlit as st

from PIL import Image
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

# st.session_state['answer'] = ''

# st.write(st.session_state)

# realans = ['', 'abc', 'edf']

# if  st.session_state['answer'] in realans:
#     answerStat = "correct"
# elif st.session_state['answer'] not in realans:
#     answerStat = "incorrect"

# st.write(st.session_state)
# st.write(answerStat)
        
model = load_model('fruit.hdf5')

labels = {0: 'Trai_bo', 1: 'Trai_cam', 2: 'Trai_chuoi', 3: 'Trai_dao', 4: 'Trai_tao'}

fruits = ['Trai_bo','Trai_cam','Trai_chuoi','Trai_dao','Trai_tao']


def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Can't able to fetch the Calories")
        print(e)

def processed_img(img_path):
    img=load_img(img_path,target_size=(96,96,3))
    img=img_to_array(img)
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis = -1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()

def run():
    st.title("Phân loại trái cây🍍")
    img_file = st.file_uploader("Chọn hình ảnh", type=["jpg"])
    if img_file is not None:
        img = Image.open(img_file).resize((255,255))
        st.image(img,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # if st.button("Predict"):
        if img_file is not None:
            result= processed_img(save_image_path)
            print(result)
            if result in fruits:
                st.info('**Phân loại trái cây**')
            st.success("**Đây là trái : "+result+'**')  
run()
