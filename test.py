import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image
import imutils
import os
import tempfile
import time
from io import BytesIO
import base64
import av
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import asyncio
from aiortc.contrib.media import MediaPlayer

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():

    selected_box = st.sidebar.selectbox(
    'Menu',
    ('Welcome','Image Processing', 'Object Detection', 'Mediapipe','Object Detection-Streaming', 'Mediapipe-Streaming')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Image Processing':
       # st.title('Counting Stars in the given image')

        imgproc = st.sidebar.radio(
            "Select Type",
            ('Threshold', 'Canny Edge', 'Counting Stars', 'Your Image'))


        if imgproc == 'Threshold':
            st.subheader('Threshold')
            if st.button('See Original Image of Stars'):
                
                original = Image.open('sample.jpg')
                st.image(original, use_column_width=True)
                
            image = cv2.imread('sample.jpg')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            x = st.sidebar.slider('Change Threshold value',min_value = 0, max_value = 255, value = 50)  
            ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
            thresh1 = thresh1.astype(np.float64)
            st.image(thresh1, use_column_width=True,clamp = True)
        
        elif imgproc == 'Canny Edge':
            st.subheader('Canny Edge Detection')
            image = cv2.imread('sample.jpg')
            low = st.sidebar.slider('low threshold',min_value = 0,max_value = 255, value = 50)
            high = st.sidebar.slider('high threshold',min_value = 0,max_value = 255, value = 100)
            edges = cv2.Canny(image,low,high)
            edges = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
            #cv2.imwrite('sample-canny.jpg',edges)
            st.image(edges,use_column_width=True,clamp=True)

        elif imgproc == 'Counting Stars':
            st.subheader('Blur + Canny + Contour')
            image = cv2.imread('sample.jpg')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_value = st.sidebar.slider('blur', min_value = 1, max_value = 21, value = 3, step =2) # 무조건 홀수여야 함
            blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
            canny_low = st.sidebar.slider('canny low threshold',min_value = 0,max_value = 255, value = 50)
            canny_high = st.sidebar.slider('canny high threshold',min_value = 0,max_value = 255, value = 100)
            canny = cv2.Canny(blur, canny_low, canny_high, 3)
            dilated = cv2.dilate(canny, (1, 1), iterations=0)
            cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            for c in cnts:
                ((x, y), _) = cv2.minEnclosingCircle(c)
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.putText(image, str(len(cnts)), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image,use_column_width=True,clamp=True)
            st.write('Number of Stars:', str(len(cnts)))

        if imgproc == 'Your Image':
            your_imgproc = st.radio(
            "Select Type: Your Image",
            ('Threshold', 'Canny Edge', 'Counting'))
            image_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg', 'webp'])
            if image_file is not None:
                our_image = Image.open(image_file)
                st.text("Original Image")
                st.image(our_image)
                frame = np.array(our_image)
                if your_imgproc == 'Threshold':
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    x = st.sidebar.slider('Change Threshold value',min_value = 0, max_value = 255, value = 50)  
                    ret,thresh1 = cv2.threshold(image,x,255,cv2.THRESH_BINARY)
                    thresh1 = thresh1.astype(np.float64)
                    st.image(thresh1, use_column_width=False,clamp = True)

                elif your_imgproc == 'Canny Edge':
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    low = st.sidebar.slider('low threshold',min_value = 0,max_value = 255, value = 50)
                    high = st.sidebar.slider('high threshold',min_value = 0,max_value = 255, value = 100)
                    edges = cv2.Canny(image,low,high)
                    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
                    st.image(edges, use_column_width=False, clamp=True)

                elif your_imgproc == 'Counting':
                    st.subheader('Number of Contour')
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur_value = st.sidebar.slider('blur', min_value = 1, max_value = 21, value = 3, step =2) # 무조건 홀수여야 함
                    blur = cv2.GaussianBlur(image, (blur_value, blur_value), 0)
                    canny_low = st.sidebar.slider('canny low threshold',min_value = 0,max_value = 255, value = 50)
                    canny_high = st.sidebar.slider('canny high threshold',min_value = 0,max_value = 255, value = 100)
                    canny = cv2.Canny(blur, canny_low, canny_high, 3)
                    dilated = cv2.dilate(canny, (1, 1), iterations=0)
                    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)

                    for c in cnts:
                        ((x, y), _) = cv2.minEnclosingCircle(c)
                        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                        cv2.putText(image, str(len(cnts)), (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    st.image(image, use_column_width=False,clamp=True)
                    st.write('Number of Contours:', str(len(cnts)))


    if selected_box == 'Object Detection':
        type = st.sidebar.radio("Select Input Type", ["Image", "Video", "Camera"])
        cfg_type = st.sidebar.selectbox("Select Config Type", ["V4", "Breed"])
        if cfg_type == 'V4':
            weights_path = 'yolov4.weights'
            cfg_path = 'yolov4-custom.cfg'
            names_path = 'coco.names'

        if cfg_type == 'Breed':
            weights_path = 'yolov4-breed_best.weights'
            cfg_path = 'yolov4-breed.cfg'
            names_path = 'breed.names'

        YOLO_net = cv2.dnn.readNet(weights_path,cfg_path)
        classes = []
        with open(names_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = YOLO_net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]


        if type == 'Image':
            image_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg', 'webp'])
            if image_file is not None:
                our_image = Image.open(image_file)
                st.text("Original Image")
                st.image(our_image)
                frame = np.array(our_image)
                if st.button("Process"):    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, c = image.shape
                    blob = cv2.dnn.blobFromImage(image,
                                                0.00392, (416, 416), (0, 0, 0),
                                                True,
                                                crop=False)
                    YOLO_net.setInput(blob)
                    outs = YOLO_net.forward(output_layers)

                    class_ids = []
                    confidences = []
                    boxes = []

                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                center_x = int(detection[0] * w)
                                center_y = int(detection[1] * h)
                                dw = int(detection[2] * w)
                                dh = int(detection[3] * h)
                                x = int(center_x - dw / 2)
                                y = int(center_y - dh / 2)
                                boxes.append([x, y, dw, dh])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            score = confidences[i]
                            caption = "{}: {:.4f}".format(label, score)
                            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
                            cv2.putText(image, caption, (x, y - 20), cv2.FONT_ITALIC, 0.5,
                                        (0, 0, 255), 1)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    st.image(image, use_column_width=False,clamp=True)
                    st.write(caption)            


        if type == 'Video':
            video_file = st.sidebar.file_uploader("Upload Video", type=['avi','mp4','mov'])
            if video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(video_file.read())    
                run = st.checkbox('Run')
                FRAME_WINDOW = st.image([])
                cap = cv2.VideoCapture(tfile.name)
                while run:
                    ret, frame = cap.read()
                    h, w, c = frame.shape
                    blob = cv2.dnn.blobFromImage(frame,
                                                0.00392, (416, 416), (0, 0, 0),
                                                True,
                                                crop=False)
                    YOLO_net.setInput(blob)
                    outs = YOLO_net.forward(output_layers)

                    class_ids = []
                    confidences = []
                    boxes = []

                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                center_x = int(detection[0] * w)
                                center_y = int(detection[1] * h)
                                dw = int(detection[2] * w)
                                dh = int(detection[3] * h)
                                x = int(center_x - dw / 2)
                                y = int(center_y - dh / 2)
                                boxes.append([x, y, dw, dh])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

                    for i in range(len(boxes)):
                        if i in indexes:
                            x, y, w, h = boxes[i]
                            label = str(classes[class_ids[i]])
                            score = confidences[i]
                            caption = "{}: {:.4f}".format(label, score)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                            cv2.putText(frame, caption, (x, y - 20), cv2.FONT_ITALIC, 0.5,
                                        (0, 0, 255), 1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)


                               
        if type == 'Camera':
            st.sidebar.write('You selected WebCam. Please check the connection.')
            st.title('Real-Time Object Detection')
            run = st.checkbox('Run')
            FRAME_WINDOW = st.image([])
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            while run:
                ret, frame = cap.read()
                h, w, c = frame.shape
                blob = cv2.dnn.blobFromImage(frame,
                                            0.00392, (416, 416), (0, 0, 0),
                                            True,
                                            crop=False)
                YOLO_net.setInput(blob)
                outs = YOLO_net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * w)
                            center_y = int(detection[1] * h)
                            dw = int(detection[2] * w)
                            dh = int(detection[3] * h)
                            x = int(center_x - dw / 2)
                            y = int(center_y - dh / 2)
                            boxes.append([x, y, dw, dh])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        score = confidences[i]
                        caption = "{}: {:.4f}".format(label, score)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                        cv2.putText(frame, caption, (x, y - 20), cv2.FONT_ITALIC, 0.5,
                                    (0, 0, 255), 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)


    if selected_box == 'Mediapipe':
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        mp_face_detection = mp.solutions.face_detection
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose
        type = st.sidebar.radio("Select Input Type", ["Image", "Video", "Camera"])
        media_type = st.sidebar.radio("Select Type", ["Hand", "Pose", "Face"])
        if type == 'Image':
            image_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg', 'webp'])
            if image_file is not None:
                our_image = Image.open(image_file)
                st.text("Original Image")
                st.image(our_image, width = 300, use_column_width=False)
                frame = np.array(our_image)
                
                if media_type == 'Hand':
                    if st.button("Process"):
                        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = hands.process(image)
                            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    finger1 = int(hand_landmarks.landmark[4].x * 100)
                                    finger2 = int(hand_landmarks.landmark[8].x * 100)
                                    dist = abs(finger1 - finger2)
                                    cv2.putText(image,
                                                text='f1=%d f2=%d dist=%d ' %
                                                (finger1, finger2, dist),
                                                org=(10, 30),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=1,
                                                color=255,
                                                thickness=3)
                                    
                                    mp_drawing.draw_landmarks(image, hand_landmarks,
                                                            mp_hands.HAND_CONNECTIONS)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(image)
                                st.image(image, width = 300, use_column_width=False, clamp=True)


                if media_type == 'Pose':
                    if st.button("Process"):
                        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(image)
                            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = Image.fromarray(image)
                            st.image(image, width = 300, use_column_width=False, clamp=True)
                            
                if media_type == 'Face':
                    face_type = st.sidebar.radio("Select Task", ['draw', "blur", "sticker"])
                    cute = cv2.imread(r'C:\Users\jskim\Desktop/cute_sticker.jpg')
                    if st.button("Process"):
                        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            result = face_detection.process(image)
                            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            if result.detections:
                                for detection in result.detections:
                                    if face_type == 'draw':
                                        mp_drawing.draw_detection(image, detection)
                                    else:
                                        bboxC = detection.location_data.relative_bounding_box
                                        ih, iw, ic = image.shape
                                        bbox = (int(bboxC.xmin * iw) - 40,int(bboxC.ymin * ih) - 150,int(bboxC.width * iw) + 80,int(bboxC.height * ih) + 200,)
                                        x, y, w, h = bbox
                                        x1, y1 = x + w, y + h
                                        try:
                                            face_image = image[y:y1, x:x1].copy()
                                            if face_type == 'blur':
                                                face_image = cv2.GaussianBlur(face_image,(99,99), 30)
                                            if face_type == 'sticker':
                                                face_image = cv2.resize(cute, dsize=(x1 - x, y1 - y), interpolation = cv2.INTER_AREA)
                                            image[y:y1, x:x1] = face_image
                                        except:
                                            pass
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                image = Image.fromarray(image)
                                st.image(image, width = 300, use_column_width=False, clamp=True)

                    
        if type == 'Video':
            video_file = st.sidebar.file_uploader("Upload Video", type=['avi','mp4','mov'])
            if video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(video_file.read())    
                cap = cv2.VideoCapture(tfile.name)
                if media_type == 'Hand':
                    st.title('Hand')
                    run = st.checkbox('Run')
                    FRAME_WINDOW = st.image([])
                    with mp_hands.Hands(max_num_hands=1,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as hands:
                        while run:
                            ret, frame = cap.read()
                            if ret:
                                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = hands.process(image)
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                                if results.multi_hand_landmarks:
                                    for hand_landmarks in results.multi_hand_landmarks:
                                        finger1 = int(hand_landmarks.landmark[4].x * 100)
                                        finger2 = int(hand_landmarks.landmark[8].x * 100)
                                        dist = abs(finger1 - finger2)
                                        cv2.putText(image,
                                                    text='f1=%d f2=%d dist=%d ' %
                                                    (finger1, finger2, dist),
                                                    org=(10, 30),
                                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                    fontScale=1,
                                                    color=255,
                                                    thickness=3)
                                        
                                        mp_drawing.draw_landmarks(image, hand_landmarks,
                                                                mp_hands.HAND_CONNECTIONS)

                            if not ret:
                                break

                            else:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(image)
                        

                if media_type == 'Pose':
                    st.title('Pose')
                    run = st.checkbox('Run')
                    FRAME_WINDOW = st.image([])
                    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
                        while run:
                            ret, frame = cap.read()
                            if ret:
                                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = pose.process(image)
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                                mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                            if not ret:
                                break

                            else:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(image)


                if media_type == 'Face':
                    st.title('Face')
                    face_type = st.sidebar.radio("Select Task", ['draw', "blur", "sticker"])
                    run = st.checkbox('Run')
                    FRAME_WINDOW = st.image([])
                    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                        while run:
                            ret, frame = cap.read()
                            if ret:
                                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                result = face_detection.process(image)
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                if result.detections:
                                    for detection in result.detections:
                                        if face_type == 'draw':
                                            mp_drawing.draw_detection(image, detection)
                                        else:
                                            bboxC = detection.location_data.relative_bounding_box
                                            ih, iw, ic = image.shape
                                            bbox = (int(bboxC.xmin * iw) - 40,int(bboxC.ymin * ih) - 150,int(bboxC.width * iw) + 80,int(bboxC.height * ih) + 200,)
                                            x, y, w, h = bbox
                                            x1, y1 = x + w, y + h
                                            try:
                                                face_image = image[y:y1, x:x1].copy()
                                                if face_type == 'blur':
                                                    face_image = cv2.GaussianBlur(face_image,(99,99), 30)
                                                if face_type == 'sticker':
                                                    face_image = cv2.resize(cute, dsize=(x1 - x, y1 - y), interpolation = cv2.INTER_AREA)
                                                image[y:y1, x:x1] = face_image
                                            except:
                                                pass
                            if not ret:
                                break
                            else:                                   
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                FRAME_WINDOW.image(image)

        if type == 'Camera':
            cap = cv2.VideoCapture(0)
            if media_type == 'Hand':
                st.title('Hand')
                run = st.checkbox('Run')
                FRAME_WINDOW = st.image([])
                with mp_hands.Hands(max_num_hands=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
                    while run:
                        ret, frame = cap.read()
                        if ret:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = hands.process(image)
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            if results.multi_hand_landmarks:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    finger1 = int(hand_landmarks.landmark[4].x * 100)
                                    finger2 = int(hand_landmarks.landmark[8].x * 100)
                                    dist = abs(finger1 - finger2)
                                    cv2.putText(image,
                                                text='f1=%d f2=%d dist=%d ' %
                                                (finger1, finger2, dist),
                                                org=(10, 30),
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=1,
                                                color=255,
                                                thickness=3)
                                    
                                    mp_drawing.draw_landmarks(image, hand_landmarks,
                                                            mp_hands.HAND_CONNECTIONS)

                        if not ret:
                            break

                        else:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(image)
                    

            if media_type == 'Pose':
                st.title('Pose')
                run = st.checkbox('Run')
                FRAME_WINDOW = st.image([])
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as pose:
                    while run:
                        ret, frame = cap.read()
                        if ret:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = pose.process(image)
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                        if not ret:
                            break

                        else:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(image)


            if media_type == 'Face':
                st.title('Face')
                face_type = st.sidebar.radio("Select Task", ['draw', "blur", "sticker"])
                run = st.checkbox('Run')
                FRAME_WINDOW = st.image([])
                with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                    while run:
                        ret, frame = cap.read()
                        if ret:
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            result = face_detection.process(image)
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            if result.detections:
                                for detection in result.detections:
                                    if face_type == 'draw':
                                        mp_drawing.draw_detection(image, detection)
                                    else:
                                        bboxC = detection.location_data.relative_bounding_box
                                        ih, iw, ic = image.shape
                                        bbox = (int(bboxC.xmin * iw) - 40,int(bboxC.ymin * ih) - 150,int(bboxC.width * iw) + 80,int(bboxC.height * ih) + 200,)
                                        x, y, w, h = bbox
                                        x1, y1 = x + w, y + h
                                        try:
                                            face_image = image[y:y1, x:x1].copy()
                                            if face_type == 'blur':
                                                face_image = cv2.GaussianBlur(face_image,(99,99), 30)
                                            if face_type == 'sticker':
                                                face_image = cv2.resize(cute, dsize=(x1 - x, y1 - y), interpolation = cv2.INTER_AREA)
                                            image[y:y1, x:x1] = face_image
                                        except:
                                            pass
                        if not ret:
                            break
                        else:                                   
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(image)                       


 
    if selected_box == 'Object Detection-Streaming':
        object_detection_streaming()
    if selected_box == 'Mediapipe-Streaming':
        media_pipe_hands()

def object_detection_streaming():   
    class VideoProcessor:
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            YOLO_net = cv2.dnn.readNet('yolov4.weights', 'yolov4-custom.cfg')
            classes = []
            with open('coco.names', 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            layer_names = YOLO_net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]
            h, w, c = image.shape
            blob = cv2.dnn.blobFromImage(image,
                                        0.00392, (416, 416), (0, 0, 0),
                                        True,
                                        crop=False)
            YOLO_net.setInput(blob)
            outs = YOLO_net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        dw = int(detection[2] * w)
                        dh = int(detection[3] * h)
                        x = int(center_x - dw / 2)
                        y = int(center_y - dh / 2)
                        boxes.append([x, y, dw, dh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    score = confidences[i]
                    caption = "{}: {:.4f}".format(label, score)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    cv2.putText(image, caption, (x, y - 20), cv2.FONT_ITALIC, 0.5,
                                (0, 0, 255), 1)
            return av.VideoFrame.from_ndarray(image, format="bgr24")    
    webrtc_streamer(key="object-detection",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                   )
def media_pipe_hands():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    class MPHandVideoProcessor:
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
                results = hands.process(image)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        finger1 = int(hand_landmarks.landmark[4].x * 100)
                        finger2 = int(hand_landmarks.landmark[8].x * 100)
                        dist = abs(finger1 - finger2)
                        cv2.putText(image,
                                    text='f1=%d f2=%d dist=%d ' %
                                    (finger1, finger2, dist),
                                    org=(10, 30),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1,
                                    color=255,
                                    thickness=3)
                        
                        mp_drawing.draw_landmarks(image, hand_landmarks,
                                                mp_hands.HAND_CONNECTIONS)
                return av.VideoFrame.from_ndarray(image, format="bgr24")
    webrtc_streamer(key="mediapipe-hands",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=MPHandVideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
    )
    
def hand(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger1 = int(hand_landmarks.landmark[4].x * 100)
            finger2 = int(hand_landmarks.landmark[8].x * 100)
            dist = abs(finger1 - finger2)
            cv2.putText(image,
                        text='f1=%d f2=%d dist=%d ' %
                        (finger1, finger2, dist),
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=255,
                        thickness=3)
            
            mp_drawing.draw_landmarks(image, hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS)
    return image

    


                              
    
         
def welcome():
    st.title('Computer Vision Demonstration')
    st.subheader('Jisoo Kim')
    st.write('')



         
if __name__ == '__main__':
    main()
