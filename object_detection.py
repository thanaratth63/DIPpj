import numpy as np
import cv2

CLASSES=["BACKGROUND", "AEROPLANE","BICYCLE", "BIRD", "BOAT", 
"BOTTLE", "BUS", "CAR","CAT","CHIR","COW","DININGTABLE", "DOG", 
"HORSE", "MOTOBIKE", "PERSON", "POTTEDPLANT","SHEEP", "SOFA", 
"TRAIN", "TVMONITOR"]

#สีกรอบแบบสุ่มโดยใช้ np ขนาดเท่ากับจำนวนคลาส
COLORS = np.random.uniform(0,255,size=(len(CLASSES),3))
#เรียกตัวnetwork
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt",
"./MobileNetSSD/MobileNetSSD.caffemodel")
#เรียกใช้กล้องหรือรูปภาพ,วิดีโอ
cap = cv2.VideoCapture(0)
#loopในการอ่านค่า
while True:
    ret,frame = cap.read()
    if ret:
        # get ความสูงและความกว้าง
        (h,w)=frame.shape[:2] 
#preprocess input ,รูป,scale factor ,ขนาดรูป,mean suctraction ปรับค่าแสง(Normalize input)
        blob = cv2.dnn.blobFromImage(frame,0.007843,(320,320),127.5) 
        net.setInput(blob)
        detections = net.forward()
        #ค่า% หาค่ารูป 
        for i in np.arange(0,detections.shape[2]):
            #สูตรในการหา
            percent = detections[0,0,i,2]
            #threshold
            if percent > 0.4:
                class_index = int(detections[0,0,i,1])
                #สูตรกรอบ
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX,startY,endX,endY) = box.astype("int")
                #บอกว่าสิ่งของนั้นคืออะไร
                label = "{} [{:.2f}%]".format(CLASSES[class_index],percent*100)
                #box
                cv2.rectangle(frame, (startX,startY), (endX,endY),COLORS[class_index],2)
                cv2.rectangle(frame, (startX-1,startY-30), (endX+1,startY),COLORS[class_index],cv2.FILLED)
                #text
                y = startY -15 if startY-15 > 15 else startY+15
                cv2.putText(frame,label,(startX+20,startY-2),cv2.FONT_HERSHEY_DUPLEX,0.6,(0,0,255),1)
        
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF==ord('q'): #กดตัวqหยุดการทำงาน
            break
cap.release() 
cv2.destroyAllWindows() #เคลียร์window

