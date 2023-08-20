# To run: streamlit run streamlitApp.py

# Importing dependencies
import streamlit as st
import os
import imageio

import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Setting layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Adding a bit structure to our page;
# Adding sidebar
with st.sidebar:
    # st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.image("./Header-icon/H4.jpeg", width=270)
    st.title("Lip Reader - MTP Phase I")
    st.info("This app is a part of a Master Thesis Project (2023-2024) under Guha's Gangsters!")

# Title
st.title('ASR made beautiful - Full Stack App v1.0')

# Generating a list of inputs - Videos
options = os.listdir(os.path.join('..', 'data', 's1'))
# print(options)

# Now creating a drop down to select them
selected_video = st.selectbox('Choose video', options)

# Generating two columns
col1, col2 = st.columns(2)

if options:
    with col1:
        # st.text('Column 01')
        st.info('Below is the converted video in .mp4 format using ffmpeg')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        
        # Converting .mpg to .mp4 as streamlit has some issues
        # We do this using ffmpeg
        # '-y' : Yes we wanna do it;
        # Saves the file in "/app/"
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Now rendering it to display on app
        # 'rb' : read as binary
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)


    with col2:
        # st.text('Column 02')
        st.info('All that model gets to learn from after pre-processing')
        # load_data(): Expects input as a tf tensor
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        # Saving and displaying the GIF
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of our ML model as tokens')
        # Using our beautiful model
        model = load_model()
        
        # Making predictions
        # Need to make the batch size 1 as we have only 1 example
        # So we wrap it inside another array
        yhat = model.predict(tf.expand_dims(video, axis=0))

        # Before decoder: Uncomment if you wanna see the duplicates
        # st.text(tf.argmax(yhat, axis=1))

        # After decoder:
        # tf.keras.backend.ctc_decoder(predictions, len(predictions), algo for decoding)
        # 'greedy': Take the most probable prediction when it comes to gen. output;
        # Predictin is nested in bunch of arrays thus we take [0][0]
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        st.info('Decoded raw tokens into words')
        # Converting predictions to text - readable text;
        converted_predictions = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_predictions)