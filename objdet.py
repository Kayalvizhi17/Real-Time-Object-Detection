import cv2
import numpy as np
import time
#import matplotlib.pyplot as plt
import os
from gtts import gTTS 

#xpoints = np.array([0, 6])
#ypoints = np.array([0, 250])

#plt.show()

#Loading the weights file
model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
print("Loading the model...")

#Saving all the class names to the list classes
classes = []
with open("coco.names", "r") as f:
    classes = 	[
    			line.strip() for line in f.readlines()
    		]

#Getting the layers of the network
#layer_names = model.getLayerNames()
#Deciding the output layer names from the YOLO model
output_layers = model.getUnconnectedOutLayersNames() #82,94,106
#[
#			layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()
#		]
print("Loaded successfully")
#print(output_layers)

video_capture = cv2.VideoCapture("video1.mp4")
print("Fetched video successfully")

while True:
    #Starts to capture frame-by-frame
    check, frame = video_capture.read()
    #print(frame.shape)
    if not check:
    	continue
    frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #print(frame.shape)
    height, width, n_channels = frame.shape

    #Using blob function of opencv to preprocess the frames
    blobbed = cv2.dnn.blobFromImage(frame, 0.004, (416, 416), swapRB=True, crop=False)
    #print(blobbed.shape)
    #Detecting objects
    model.setInput(blobbed)
    starting_time = time.time()
    outs = model.forward(output_layers)
    end_time = time.time()
    #print(outs)
    print(end_time - starting_time)

    #Finding the classes and calculating the coordinates for bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    centers = []
    for out in outs:
        for detection in out:
        	scores = detection[5:]
        	class_id = np.argmax(scores)
        	confidence = scores[class_id]
        	if confidence > 0.7:
        		x_coordinate = int(detection[0] * width)
        		y_coordinate = int(detection[1] * height)
        		#print(detection[0],detection[1],detection[2],detection[3])
        		_width = int(detection[2] * width)
        		_height = int(detection[3] * height)
        		
        		#Rectangle coordinates
        		rect_x = int(x_coordinate - _width / 2)
        		rect_y = int(y_coordinate - _height / 2)
        		
        		#apx_distance = round (((1 - (detection[3] - detection[1]))**4),1)
        		#print(apx_distance, "e")
        		#print(width,height,x_coordinate,y_coordinate,_width,_height,rect_x,rect_y)
        		
        		boxes.append([rect_x, rect_y, _width, _height])
        		confidences.append(float(confidence))
        		#print(confidences)
        		class_ids.append(class_id)
        		centers.append((x_coordinate, y_coordinate))
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #Fixing the font and generating random colors
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    #Showing the labels and bounding boxes on screen
    assists = ''
    labels = []
    for b in range(len(boxes)):
    	if b in indexes:
    		rect_x, rect_y, _width, _height = boxes[b]
    		label = str(classes[class_ids[b]])
    		if label not in labels:
    			labels.append(label)
    			labels.append(1)
    		else:
    			for i in range(len(labels)):
    				if(labels[i] == label):
    					labels[i+1] = labels[i+1] + 1
    					break
    					
    		#assists = assists + label
    		color = colors[class_ids[b]]
    		cv2.rectangle(frame, (rect_x, rect_y), (rect_x + _width, rect_y + _height), color, 1)
    		cv2.putText(frame, label, (rect_x, rect_y), font, 1, color, 2)
    		center_x, center_y = centers[b][0], centers[b][1]
    		if center_x <= width/3:
    			W_pos = "in left."
    		elif center_x <= (width/3)*2 and center_x > width/3:
    			W_pos = "at straight."
    		else:
    			W_pos = "at right."
    		if center_y <= height/3:
    			H_pos = label+" too far "
    		elif center_y <= (height/3)*2 and center_y > height/3:
    			H_pos = label+" at safer distance "
    		else:
    			H_pos = "Warning! "+label+" too close "
    		assists = assists + " " + H_pos + W_pos
        	#print(center_x,center_y)
        	
	
    cv2.imshow("Image",cv2.resize(frame, (600,600)))
    print(assists)
    print(labels)
    if(assists != ''):  
    	print(assists)
    	myobj = gTTS(text=assists, lang='en', slow=False)
    	myobj.save("detections.mp3")
    	os.system("mpg321 -q detections.mp3")
    
    #Setting alphabet 'q' for quit option
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#Releasing and destroying the window        
video_capture.release()
cv2.destroyAllWindows()
#print(indexes,len(indexes),'1')

    	#print(indexes,len(indexes),'2')
    #print(indexes,len(indexes),'3')
   # if len(indexes) > 0:
   # 	for i in indexes.flatten():
   # 		# find positions
