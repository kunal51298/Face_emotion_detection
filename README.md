# Face_emotion_detection
Introduction
“Emotion recognition is a technique that allows reading the emotions on a human face using advanced image processing.”

Emotion recognition is one of the many facial recognition technologies that have developed and grown through the years. Currently, facial emotion recognition software is used to allow a certain program to examine and process the expressions on a human’s face.

What is the use and why it is important?

Use of technology to help people with emotion recognition is a relatively nascent research area. Facial expressions are a form of nonverbal communication. Various studies have been done for the classification of these facial expressions. There is strong evidence for the universal facial expressions of seven emotions which include: neutral happy, sadness, anger, disgust, fear, and surprise. So it is very important to detect these emotions on the face as it has wide applications in the field of Computer Vision and Artificial Intelligence. These fields are researching on the facial emotions to get the sentiments of the humans automatically.



Problem Statement
The Indian education landscape has been undergoing rapid changes for the past 10 years owing to the advancement of web-based learning services, specifically, eLearning platforms.

Global E-learning is estimated to witness an 8X over the next 5 years to reach USD 2B in 2021. India is expected to grow with a CAGR of 44% crossing the 10M users mark in 2021. Although the market is growing on a rapid scale, there are major challenges associated with digital learning when compared with brick and mortar classrooms. One of many challenges is how to ensure quality learning for students.

Digital platforms might overpower physical classrooms in terms of content quality but when it comes to understanding whether students are able to grasp the content in a live class scenario is yet an open-end challenge.

In a physical classroom during a lecturing teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention. Digital classrooms are conducted via video telephony software program (exZoom) where it’s not possible for medium scale class (25-50) to see all students and access the mood.

Because of this drawback, students are not focusing on content due to lack of surveillance.While digital platforms have limitations in terms of physical surveillance but it comes with the power of data and machines which can work for you. It provides data in the form of video, audio, and texts which can be analysed using deep learning algorithms. Deep learning backed system not only solves the surveillance issue, but it also removes the human bias from the system, and all information is no longer in the teacher’s brain rather translated in numbers that can be analysed and tracked.

We will solve the above-mentioned challenge by applying deep learning algorithms to live video data. The solution to this problem is by recognizing facial emotions.

Presentation link - https://drive.google.com/drive/folders/1xSdt6RAJcM2KCPjRJEgGn9t-qshJ_liX?usp=sharing
Dataset Information
The model is trained on the FER-2013 dataset .This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised. Here is the dataset link:- https://www.kaggle.com/msambare/fer2013

Model Creation
access all created models from here= https://drive.google.com/drive/folders/18mlUOB_1lFsZSu-d-m04H7wmuLF9A2RU?usp=sharing

Using DeepFace
Deepface is a lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python. It is a hybrid face recognition framework wrapping state-of-the-art models: VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace and Dlib. Those models already reached and passed the human level accuracy. The library is mainly based on TensorFlow and Keras.



Inference after using DeepFace
Most of the emotions predicted by DeepFace model were incorrect.

Using Transfer Learning
Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.From the practical standpoint, reusing or transferring information from previously learned tasks for the learning of new tasks has the potential to significantly improve the sample efficiency

VGG-16
VGG16 is a convolution neural net (CNN ) architecture which was used to win ILSVR(Imagenet) competition in 2014. Most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

ResNet-50
ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pre-trained version of the network trained on more than a million images from the ImageNet database. The pre-trained network can classify images into 1000 object categories, such as a keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images.



Inference of ResNet-50
The training and validation accuracy is 85.71% and training & validation loss also decreased below 1.80 After verifying from live video this model doesn't seem to work very well in recognizing face emotions. Most of the time it predicted happy face although it was some other expression!!

Custom CNN Model
Created a custom CNN model using Conv2D, MaxPooling, BatchNormalization, Dropout and Dense layers. Activation function used is "ReLU". Output layer has 7 nodes with activation function as "Softmax". Adam Optimizer is used in this model. Total params: 4,496,903



The training accuracy obtained from this model is 66.64% and validation accuracy is 66.10% after 50 epochs. Model is performing good in live video feed.

Loss and accuracy plot


Dependencies
Python-3
Tensorflow
Keras
Opencv
Streamlit
Streamlit-webrtc
Deployment of Streamlit Webapp on Heroku
For this project I have made a front end application using streamlit .Streamlit doesn’t provide the live capture feature itself, instead uses a third party API. Therefore, used streamlit-webrtc which helped to deal with real-time video streams. Image captured from the webcam is sent to FaceEmotion function to detect the emotion. Then this model was deployed on heroku.

App Link deployed on Heroku- https://face-detector-almabetter.herokuapp.com/

Basic Requirements to deploy on heroku
Procfile
setup.sh
requirements.txt
Conclusion
Finally build a Face Emotion Recognition webapp using streamlit and deployed on Heroku, which predicts the face emotions on live webcam.

The model created with CNN layers gave training accuracy of 66.64% and validation accuracy of 66.10% after 50 epochs.

Drawback- not classifying well on disgust images.

Link of Demo video(working in local) - https://drive.google.com/drive/folders/1IxT8R_uNzqsx7BE3lrOX2P-0cCP8zbQ2?usp=sharing
