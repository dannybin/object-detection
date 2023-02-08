import av
import sys
from matplotlib import image
import numpy as np
import PIL
import os
import tensorflow as tf
import cv2


### TODO:

# Load pipeline config and build a detection model
##configs = config_util.get_configs_from_pipeline_file("model/pipeline.config")
##detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Load mobilenet SSD model
modelFile = "model/mobilenet_iter_73000.caffemodel"
configFile = "model/deploy.prototxt.txt"
videoFile = "video/nyc.mkv"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# Restore checkpoint
##ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
##ckpt.restore('model/ckpt-7').expect_partial()

##category_index = label_map_util.create_category_index_from_labelmap('model/label_map.pbtxt')

##@tf.function
##def detect_fn(image):
##    image, shapes = detection_model.preprocess(image)
##    prediction_dict = detection_model.predict(image, shapes)
##    detections = detection_model.postprocess(prediction_dict, shapes)
##    return detections


def main():

    cap = cv2.VideoCapture(2)
    bicycleTotalCount = bicyclePreviousCount = 0
    carTotalCount = carPreviousCount = 0
    catTotalCount = catPreviousCount = 0
    dogTotalCount = dogPreviousCount = 0
    motorbikeTotalCount = motorbikePreviousCount = 0
    aeroplaneTotalCount = aeroplanePreviousCount = 0
    birdTotalCount = birdPreviousCount = 0
    boatTotalCount = boatPreviousCount = 0
    busTotalCount = busPreviousCount = 0
    chairTotalCount = chairPreviousCount = 0
    horseTotalCount = horsePreviousCount = 0
    trainTotalCount = trainPreviousCount = 0
    sofaTotalCount = sofaPreviousCount = 0
    personTotalCount = personPreviousCount = 0
    screenTotalCount = screenPreviousCount = 0
    sheepTotalCount = sheepPreviousCount = 0
    tableTotalCount = tablePreviousCount = 0
    plantTotalCount = plantPreviousCount = 0
    cowTotalCount = cowPreviousCount = 0
    bottleTotalCount = bottlePreviousCount = 0
    backgroundTotalCount = backgroundPreviousCount = 0

    while(True):
      # Capture the video frame
      # by frame
      ret, image_np = cap.read()
      if(ret == False):
        continue
      image_np_copy = image_np
      bicycleCurrentCount = 0
      carCurrentCount = 0
      catCurrentCount = 0
      dogCurrentCount = 0
      motorbikeCurrentCount = 0
      birdCurrentCount = 0
      boatCurrentCount = 0
      aeroplaneCurrentCount = 0
      trainCurrentCount = 0
      horseCurrentCount = 0
      chairCurrentCount = 0
      busCurrentCount = 0
      sofaCurrentCount = 0
      personCurrentCount = 0
      screenCurrentCount = 0
      sheepCurrentCount = 0
      tableCurrentCount = 0
      plantCurrentCount = 0
      cowCurrentCount = 0
      bottleCurrentCount = 0
      backgroundCurrentCount = 0

      
## convert NP array to 3 channels, needed for to_ndarray()
#                image_np = np.expand_dims(image_np, axis=-1)
#                print(image_np.shape)
#                print(type(image_np))
      h, w = image_np_copy.shape[:2]
### use DNN facial detection model
      blob = blob = cv2.dnn.blobFromImage(image_np_copy, size=(300, 300), ddepth=cv2.CV_8U)
      net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5, 127.5, 127.5])
      detections = net.forward()
      
      for i in np.arange(0,detections.shape[2]): 
        confidence = detections[0, 0, i, 2] 

### adjust the confidence of the detection
        if confidence > 0.8:

          idx = int(detections[0, 0, i, 1])
          print(CLASSES[idx])
          if(CLASSES[idx] == "bicycle"):
            bicycleCurrentCount += 1
          elif(CLASSES[idx] == "car"):
            carCurrentCount += 1
          elif(CLASSES[idx] == "cat"):
            catCurrentCount += 1
          elif(CLASSES[idx] == "dog"):
            dogCurrentCount += 1
          elif(CLASSES[idx] == "motorbike"):
            motorbikeCurrentCount += 1
          elif(CLASSES[idx] == "bird"):
            birdCurrentCount += 1
          elif(CLASSES[idx] == "boat"):
            boatCurrentCount += 1
          elif(CLASSES[idx] == "aeroplane"):
            aeroplaneCurrentCount += 1
          elif(CLASSES[idx] == "train"):
            trainCurrentCount += 1
          elif(CLASSES[idx] == "horse"):
            horseCurrentCount += 1
          elif(CLASSES[idx] == "chair"):
            chairCurrentCount += 1
          elif(CLASSES[idx] == "bus"):
            busCurrentCount += 1
          elif(CLASSES[idx] == "sofa"):
            sofaCurrentCount += 1
          elif(CLASSES[idx] == "person"):
            personCurrentCount += 1
          elif(CLASSES[idx] == "tvmonitor"):
            screenCurrentCount += 1
          elif(CLASSES[idx] == "diningtable"):
            tableCurrentCount += 1
          elif(idx == "sheep"):
            sheepCurrentCount += 1
          elif(CLASSES[idx] == "pottedplant"):
            plantCurrentCount += 1
          elif(CLASSES[idx] == "cow"):
            cowCurrentCount += 1
          elif(CLASSES[idx] == "bottle"):
            bottleCurrentCount += 1
          elif(CLASSES[idx] == "background"):
            backgroundCurrentCount += 1

          box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
          (x, y, x1, y1) = box.astype("int")

### draw red box around the detected face and return the new numpy array of the frame
          image_np_copy = cv2.rectangle(image_np_copy, (x, y), (x1, y1), (0, 0, 255), 2)
        image_np = image_np_copy
      
      if(bicycleCurrentCount > bicyclePreviousCount):
        bicycleTotalCount += (bicycleCurrentCount-bicyclePreviousCount)
      if(carCurrentCount > carPreviousCount):
        carTotalCount += (carCurrentCount-carPreviousCount)
      if(catCurrentCount > catPreviousCount):
        catTotalCount += (catCurrentCount-catPreviousCount)
      if(dogCurrentCount > dogPreviousCount):
        dogTotalCount += (dogCurrentCount-dogPreviousCount)
      if(motorbikeCurrentCount > motorbikePreviousCount):
        motorbikeTotalCount += (motorbikeCurrentCount-motorbikePreviousCount)
      if(birdCurrentCount > birdPreviousCount):
        birdTotalCount += (birdCurrentCount-birdPreviousCount)
      if(boatCurrentCount > boatPreviousCount):
        boatTotalCount += (boatCurrentCount-boatPreviousCount)
      if(aeroplaneCurrentCount > aeroplanePreviousCount):
        aeroplaneTotalCount += (aeroplaneCurrentCount-aeroplanePreviousCount)
      if(trainCurrentCount > trainPreviousCount):
        trainTotalCount += (trainCurrentCount-trainPreviousCount)
      if(horseCurrentCount > horsePreviousCount):
        horseTotalCount += (horseCurrentCount-horsePreviousCount)
      if(chairCurrentCount > chairPreviousCount):
        chairTotalCount += (chairCurrentCount-chairPreviousCount)
      if(busCurrentCount > busPreviousCount):
        busTotalCount += (busCurrentCount-busPreviousCount)
      if(sofaCurrentCount > sofaPreviousCount):
        sofaTotalCount += (sofaCurrentCount-sofaPreviousCount)
      if(personCurrentCount > personPreviousCount):
        personTotalCount += (personCurrentCount-personPreviousCount)
      if(screenCurrentCount > screenPreviousCount):
        screenTotalCount += (screenCurrentCount-screenPreviousCount)
      if(tableCurrentCount > tablePreviousCount):
        tableTotalCount += (tableCurrentCount-tablePreviousCount)
      if(sheepCurrentCount > sheepPreviousCount):
        sheepTotalCount += (sheepCurrentCount-sheepPreviousCount)
      if(plantCurrentCount > plantPreviousCount):
        plantTotalCount += (plantCurrentCount-plantPreviousCount)
      if(cowCurrentCount > cowPreviousCount):
        cowTotalCount += (cowCurrentCount-cowPreviousCount)
      if(bottleCurrentCount > bottlePreviousCount):
        bottleTotalCount += (bottleCurrentCount-bottlePreviousCount)
      if(backgroundCurrentCount > backgroundPreviousCount):
        backgroundTotalCount += (backgroundCurrentCount-backgroundPreviousCount)

      bicyclePreviousCount = bicycleCurrentCount  
      carPreviousCount = carCurrentCount 
      catPreviousCount = catCurrentCount 
      dogPreviousCount = dogCurrentCount 
      motorbikePreviousCount = motorbikeCurrentCount 
      boatPreviousCount = boatCurrentCount 
      birdPreviousCount = birdCurrentCount 
      aeroplanePreviousCount = aeroplaneCurrentCount 
      trainPreviousCount = trainCurrentCount 
      horsePreviousCount = horseCurrentCount 
      chairPreviousCount = chairCurrentCount 
      busPreviousCount = busCurrentCount 
      sofaPreviousCount = sofaCurrentCount 
      personPreviousCount = personCurrentCount 
      screenPreviousCount = screenCurrentCount 
      tablePreviousCount = tableCurrentCount 
      sheepPreviousCount = sheepCurrentCount  
      plantPreviousCount = plantCurrentCount 
      cowPreviousCount = cowCurrentCount 
      bottlePreviousCount = bottleCurrentCount 
      backgroundPreviousCount = backgroundCurrentCount 
      
      # Display the resulting frame
      cv2.imshow('frame', image_np)
      print(
            "person:", personTotalCount,
            " bus:", busTotalCount, 
            " car:", carTotalCount, 
            " dog:", dogTotalCount,
            " cat:", catTotalCount,
            " motorbike:", motorbikeTotalCount,
            " aeroplane:", aeroplaneTotalCount, 
            " bicycle:", bicycleTotalCount, 
            " bird:", birdTotalCount,
            " boat:", boatTotalCount,
            " bottle:", bottleTotalCount, 
            " chair:", chairTotalCount,
            " cow:", cowTotalCount,
            " diningtable:", tableTotalCount,
            " horse:", horseTotalCount,
            " pottedplant:", plantTotalCount,
            " sheep:", sheepTotalCount,
            " sofa:", sofaTotalCount, 
            " train:", trainTotalCount, 
            " screen:", screenTotalCount,
            " background:", backgroundTotalCount,)
      print("")
      # the 'q' button is set as the
      # quitting button you may use any
      # desired button of your choice
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    

    

    source = av.open("pipe:", format="avi", mode="r")
    
    source_v = source.streams.video[0]
    source_a = source.streams.audio[0]

    sink = av.open("pipe:", format="avi", mode="w")
    sink_v = sink.add_stream(template=source_v)
    sink_a = sink.add_stream(template=source_a)

    for packet in source.demux():
        if packet is None:
            continue
        for frame in packet.decode():
            index = frame.index

            if packet.stream.type == 'audio':
              packet.stream = sink_a

            if packet.stream.type == 'video':

                print("********************************************************************")

                ret, image_np = cap.read()
                image_np_copy = image_np
### test code to make sure the frames are being read properly
#                cv2.imshow("test", image_np)
#                if cv2.waitKey(1) & 0xFF == ord('q'):
#                  cap.release()
#                  cv2.destroyAllWindows()
#                  break

## convert NP array to 3 channels, needed for to_ndarray()
#                image_np = np.expand_dims(image_np, axis=-1)
#                print(image_np.shape)
#                print(type(image_np))
                h, w = image_np_copy.shape[:2]
### use DNN facial detection model
                blob = cv2.dnn.blobFromImage(cv2.resize(image_np_copy, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
                print(blob.shape)
                net.setInput(blob)
                faces = net.forward()
                
                for i in range(faces.shape[2]): 
                  confidence = faces[0, 0, i, 2] 
                  
### adjust the confidence of the detection
                  if confidence > 0.7:
                    
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")

### draw red box around the detected face and return the new numpy array of the frame
                    image_np_copy = cv2.rectangle(image_np_copy, (x, y), (x1, y1), (0, 0, 255), 2)
                  image_np = image_np_copy
### render using opencv
                cv2.imshow("test", image_np)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                  cv2.destroyAllWindows()
                  break
#                print(image_np.shape)
#                frame = av.VideoFrame.from_ndarray(image_np, format="rgb24")
#                print(frame)
#                for packet in sink_v.encode(frame):
#                  sink_v.mux(packet)  
            sink.mux(packet)
    sink.close()

if __name__ == '__main__':
    main()