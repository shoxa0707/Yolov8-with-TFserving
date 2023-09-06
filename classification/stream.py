import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image
from resolution import resolution
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time

def img2tensor(image):
    image = cv2.resize(image, (320, 320))
    image = image/255.
    image = np.expand_dims(image, 0)
    image = image.tolist()
    return image

def get_input(model_name):
     # Get the serving_input key
    loaded_model = tf.saved_model.load(model_name)
    input_name = list(
        loaded_model.signatures["serving_default"].structured_input_signature[1].keys()
    )[0]
    return input_name

GRPC_URL = "localhost:8500"
REST_URL = "http://localhost:8501/v1/models/seatbelt:predict"

CLASS = ["Worn ✅", "Not worn ❌", "Undefined 🤷‍♂️"]
st.markdown("<h1 style='text-align: center; color: red;'>Streamlit ➕ TensorflowServing</h1>", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color:Blue; font-size: 50px;"><strong>Classification and Resolution</strong></p>', unsafe_allow_html=True)
st.markdown('<style>body{color: White; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

option = st.selectbox(
    'What kind of service do you want?',
    ('Seatbelt classification', 'Image super resolution (4x scale)'))

if option == "Seatbelt classification":
    
    
    st.title(':green[Seatbelt classification]')

    img_file_buffer = st.file_uploader("upload your image", type=["png", "jpg", "jpeg"])
    col1, col2 = st.columns(2)
    with col1:
        res = st.radio(
            "Do you want the resolution used for your image?",
            ('Yes', 'No'))
    with col2:
        api = st.radio(
            "Which API do you want to predict of tfserving?",
            ('REST API', 'GRPC API'))
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert('RGB')
        img_array = np.array(image)
        if res == "Yes":
            img = cv2.resize(resolution(img_array), (320, 320))
        elif res == "No":
            img = cv2.resize(img_array, (320, 320))
        st.image(img)
        
        if st.button('Predict'):
            start = time.time()
            im = img2tensor(img)
            if api == 'REST API':
                json_data = {
                "instances": im
                }
                # Send the request to the Prediction API
                response = requests.post(REST_URL, json=json_data)
                # Retrieve the highest probablity index of the Tensor (actual prediction)
                print(response)
                pred = response.json()['predictions'][0]
            elif api == "GRPC API":
                import grpc
                from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
                
                # Create a channel that will be connected to the gRPC port of the container
                channel = grpc.insecure_channel("localhost:8500")

                # Create a stub made for prediction
                # This stub will be used to send the gRPCrequest to the TF Server
                stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
                
                #############################################
                # input_name = get_input("your model path") #
                #############################################
                input_name = "rescaling_2_input"
                # Create a gRPC request made for prediction
                request = predict_pb2.PredictRequest()

                # Set the name of the model, for this use case it is "model"
                request.model_spec.name = "seatbelt"

                # Set which signature is used to format the gRPC query
                # here the default one "serving_default"
                request.model_spec.signature_name = "serving_default"

                # Set the input as the data
                # tf.make_tensor_proto turns a TensorFlow tensor into a Protobuf tensor
                request.inputs[input_name].CopyFrom(tf.make_tensor_proto(im))

                # Send the gRPC request to the TF Server
                output = stub.Predict(request)
                pred = np.array([output.outputs['dense_8'].float_val])[0]
            
            result = pd.DataFrame({CLASS[0].split()[0]:[round(100*pred[0], 2)], CLASS[1].split()[0]:[round(100*pred[1], 2)], CLASS[2].split()[0]:[round(100*pred[2], 2)]})
            result.index = ["Prediction(%)"]
            clas = CLASS[np.argmax(pred)]
            
            st.write("Prediction information:")
            st.dataframe(result)
            st.write(f"Prediction process time: {round(time.time() - start, 4)} s")
            st.write(f'Final result: {clas}')


elif option == "Image super resolution (4x scale)":
    st.title(':blue[Image super resolution (4x scale)]')

    img_file_buffer = st.file_uploader("upload", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img_array = np.array(image)
        res_img = resolution(img_array)
        # convert RGB to BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite('resolution.png', res_img)

        fig1 = plt.figure(figsize = (10, 10))
        plt.subplot(121)
        plt.imshow(img_array)
        plt.title('original image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(res_img)
        plt.title('Resolution image')
        plt.axis('off')
        plt.subplots_adjust(wspace=.025, hspace=.025)
        fig1.set_facecolor("black")
        st.pyplot(fig1)

        with open("resolution.png", "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="resolution.png",
                    mime="image/png"
                )
    else:
        try:
            os.remove('resolution.png')
        except:
            pass
        
