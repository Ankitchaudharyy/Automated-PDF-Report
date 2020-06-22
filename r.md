# *****CARTOON EMOTION RECOGNITION***** 

This work deals with recognizing emotions from facial expressions of cartoon characters. 
<br />
Emotions are Happy, Angry, Sad, and Surprise.

Download Mask-RCNN model - https://drive.google.com/file/d/1j1FuWvGQQjX5pGDk2b4hyT2iwq3kX1-l/view?usp=sharing.
<br />
Place it in 'models' Directory.
Run the following commands - 

```python
!python setup.py
```

```python
!python main.py ----maskrcnnweight [MaskRCNN_Weight] \
                --emotionweight [Emotion_Model_Weight] \
                --imagefolder [Test_Images_Directory] \
                --video [Test_Video_Directory] \
                --output [Output_Directory] \
                --savecsv [Predictions]
```
