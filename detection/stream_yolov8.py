import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import tempfile
import subprocess
import matplotlib.pyplot as plt
import requests
import time
import json

# TF-Serving URLs for both gRPC and REST APIs
GRPC_URL = "localhost:8500"
REST_URL = "http://localhost:8501/v1/models/yolo:predict"

################################
######### NMS functions ########
################################

def box_iou_batch(
	boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
        
    return area_inter / (area_a[:, None] + area_b - area_inter)

def non_max_suppression(
   predictions: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# pprepare image to model input
def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)

################################
##### TF Serving functions #####
################################

def get_input(model_name):
     # Get the serving_input key
    loaded_model = tf.saved_model.load(model_name)
    input_name = list(
        loaded_model.signatures["serving_default"].structured_input_signature[1].keys()
    )[0]
    return input_name

def rest(image, url):
    # Prepare the data that is going to be sent in the POST request
    json_data = {
        "instances": image.tolist()
    }
    # Send the request to the Prediction API
    response = requests.post(url, json=json_data)
    # Retrieve the highest probablity index of the Tensor (actual prediction)
    pred = response.json()['predictions'][0]
    return np.array(pred).T

def grpc(image, url, input_name='x', model_name='yolo', output_name='output0'):
    import grpc
    from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

    # Create a channel that will be connected to the gRPC port of the container
    channel = grpc.insecure_channel(
        url,
        options=[
        ("grpc.max_send_message_length", 10000000),
        ("grpc.max_receive_message_length", 10000000),
        ]
    )

    # Create a stub made for prediction
    # This stub will be used to send the gRPCrequest to the TF Server
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    #############################################
    # input_name = get_input("your model path") #
    #############################################
    input_name = input_name
    # Create a gRPC request made for prediction
    request = predict_pb2.PredictRequest()

    # Set the name of the model, for this use case it is "model"
    request.model_spec.name = model_name

    # Set which signature is used to format the gRPC query
    # here the default one "serving_default"
    request.model_spec.signature_name = "serving_default"

    # Set the input as the data
    # tf.make_tensor_proto turns a TensorFlow tensor into a Protobuf tensor
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(image))

    # Send the gRPC request to the TF Server
    output = stub.Predict(request)
    pred = np.array([output.outputs[output_name].float_val])[0]
    pred = pred.reshape((5, 8400))
    
    return pred.T

st.markdown("<h1 style='text-align: center; color: green;'>FACE DETECTION</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: red;'>Streamlit ➕ TFserving</h1>", unsafe_allow_html=True)

thresh = 0.7
with open('coco.names') as f:
    classes = f.read().split('\n')
option = st.selectbox(
    'What type of information do you want detected?',
    ('Video', 'Image'))

if option == "Video":
    video_data = st.file_uploader("upload", ['mp4','mov', 'avi'])
    col1, col2 = st.columns(2)
    with col1:
        genre = st.radio(
            "Choose how you want to see the result:",
            ('Show realtime', 'Write to file'))
    with col2:
        api = st.radio(
            "Which API do you want to predict of tfserving?",
            ('REST API', 'GRPC API'))
    if st.button("Let's started"):
        if video_data:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(video_data.read())

            my_bar = st.progress(0, text="Detection in progress. Please wait...")

            cap = cv2.VideoCapture(temp_filename)
            nums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if genre == "Write to file":
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                out = cv2.VideoWriter('result.mp4', fourcc, fps, (frame_width, frame_height))
            if genre == "Show realtime":
                imagepl = st.empty()

            if (cap.isOpened()== False):
                st.write("Error opening video file! Upload another video.")
            fpsst = st.empty()
            num = 0
            while(cap.isOpened()):
                ret, frame = cap.read()
                if num == int(nums):
                    my_bar.progress(num / nums, text="100% done.")
                else:
                    my_bar.progress(num / nums, text=f"{int(100*num/nums)}% done. Please wait...")
                num += 1
                if ret == True:
                    start = time.time()
                    image, ratio, dwdh = letterbox(frame, auto=False)
                    image = np.expand_dims(image, 0)
                    image = np.transpose(image, (0, 3, 1, 2))
                    image = np.ascontiguousarray(image)

                    im = image.astype(np.float32)
                    im /= 255
                    if api == 'REST API':
                        pred = rest(image=im, url=REST_URL)
                    elif api == 'GRPC API':
                        pred = grpc(image=im, url=GRPC_URL, input_name='images', model_name='yolo', output_name='output0')
                        
                    boxes = np.array(list(filter(lambda x: x[4] >= thresh, pred)))
                    boxes = xywh2xyxy(np.concatenate((boxes, np.zeros((boxes.shape[0], 1))), axis=1))
                    indices = non_max_suppression(boxes, 0.3)
                    for i in boxes[indices]:
                        bbox = i[:4].copy()
                        bbox[0] -= dwdh[0]
                        bbox[1] -= dwdh[1]
                        bbox[2] -= dwdh[0]
                        bbox[3] -= dwdh[1]
                        bbox /= ratio
                        bbox = bbox.round().astype(np.int32).tolist()
                        color = (0,255,0)
                        cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)

                        cv2.putText(frame,
                                    f'{classes[int(i[5])]}:{round(i[4]*100, 2)} %', (bbox[0], bbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, [0, 0, 255],
                                    thickness=1)

                    if genre == "Write to file":
                        out.write(frame)
                    if genre == "Show realtime":
                        imagepl.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    end = time.time()
                    fps = round(1 / (end - start), 2)
                    fpsst.write("FPS : " + str(fps))
                else:
                    break
                    
            if genre == "Write to file":
                out.release()
                subprocess.call(args=f"ffmpeg -y -i result.mp4 -c:v libx264 convert.mp4", shell=True)

                video_file = open('convert.mp4', 'rb')
                video_bytes = video_file.read()

                st.video(video_bytes)

                # Download button
                with open("result.mp4", "rb") as file:
                    btn = st.download_button(
                            label="Download video",
                            data=file,
                            file_name="result.mp4",
                            mime="video/mp4"
                        )
    
elif option == "Image":
    image_data = st.file_uploader("upload", ['jpg','png', 'jpeg'])
    api = st.radio(
        "Which API do you want to predict of tfserving?",
        ('REST API', 'GRPC API'))
    if image_data is not None:
        with open("image.png", 'wb') as f:
            f.write(image_data.read())
        
        start = time.time()
        image = cv2.imread("image.png")
        real = image.copy()
        frame, ratio, dwdh = letterbox(image, auto=False)
        frame = np.expand_dims(frame, 0)
        frame = np.ascontiguousarray(frame)

        im = frame.astype(np.float32)
        im /= 255

        if api == 'REST API':
            pred = rest(image=im, url=REST_URL)
        elif api == 'GRPC API':
            pred = grpc(image=im, url=GRPC_URL, input_name='images', model_name='yolo', output_name='output_0')
            
        filt = np.array(list(filter(lambda x: x[4] >= thresh, pred)))
        filt[:, :4] = 640 * filt[:, :4]
        boxes = filt[:, :5]
        boxes = xywh2xyxy(np.concatenate((boxes, np.expand_dims(np.argmax(filt[:, 5:], axis=1), 1)), axis=1))
        indices = non_max_suppression(boxes, 0.3)
        color = (0, 255, 0)
        for i in boxes[indices]:
            bbox = i[:4].copy()
            bbox[0] -= dwdh[0]
            bbox[1] -= dwdh[1]
            bbox[2] -= dwdh[0]
            bbox[3] -= dwdh[1]
            bbox /= ratio
            bbox = bbox.round().astype(np.int32).tolist()
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)

            cv2.putText(image,
                        f'{classes[int(i[5])]}:{round(i[4]*100, 2)} %', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, [0, 0, 255],
                        thickness=1)

        cv2.imwrite('result.png', image)

        fig1 = plt.figure(figsize = (10, 10))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(real, cv2.COLOR_BGR2RGB))
        plt.title('original image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Detection image')
        plt.axis('off')
        plt.subplots_adjust(wspace=.025, hspace=.025)
        fig1.set_facecolor("black")
        st.pyplot(fig1)
        st.write('Detection time: '+str(round(time.time() - start, 4)) + ' s')

        with open("result.png", "rb") as file:
            btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name="result.png",
                    mime="image/png"
                )
