import cv2
import numpy as np
import face_recognition 
import os
from datetime import datetime
import shutil

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

###path of image folder 
path="E:\\My Projects\\Smart attendence recorder\\Using Face Recognition\\img"
imga=[]
std_name=[]
std_roll=[]

#lets convert touple to a string
def toup_to_str(toup):
        str=""
        for item in toup:
            str=str+item
        return str
#lets convert a list to string
def list_to_str(lst):
    str=""
    for item in lst:
        str=str+item
    return str

## taking the image in for loop and saving as list  && taking out the name 
for cu_img in os.listdir(path):
    current_image = cv2.imread(f'{path}/{cu_img}')
    imga.append(current_image)
    n=[]
    x=list_to_str(os.path.splitext(cu_img)[0])
    n=(x.split(","))
    
    std_name.append(n[0])
    std_roll.append(n[1])
   
print(str(std_name))
print(str(std_roll))


 
##face encoding  to detect the imag && comparision (using hog alogo to encode)
def face_encode (imga):
    encode_list=[]
    for img in imga:
        img =cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

## test the funtion
# print (face_encode(imga))

encode_k =face_encode(imga)
print("All encoding done !!!!!!")


#creating the csv file with the help of python
timeNow= datetime.now()
dstr = timeNow.strftime("%d_%m_%Y")
tstr = timeNow.strftime("%S")
attn = "attendance_of_" +str(dstr)+".csv"
fn=open(attn , "w")
fn.write("Name,Roll_No.,Time,Date\n")
fn.close()

#making csv file and saving data inside
def attendance(name):
    timeNow= datetime.now()
    dstr = timeNow.strftime("%d_%m_%Y")
    tstr = timeNow.strftime("%S")
    attn = "attendance_of_" +str(dstr)+".csv"

    with open ("attendance_of_" +str(dstr)+".csv" , "r+") as  f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # nameList =[]
        if name not in  nameList:
            timeNow= datetime.now()
            dstr = timeNow.strftime("%d/%m/%Y")
            tstr = timeNow.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{std_roll[std_name.index(name)]},{tstr},{dstr}")


# To capture video from webcam. 
cap = cv2.VideoCapture(0)
while True:
    # Read the frame
    ret, img4 = cap.read()
    #put a text on the frame
    msg="Press 'ESC' to exit"
    cv2.putText(img4, str(msg) , (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    # Convert to grayscale
    face = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(img4,1.1,4)
    
   

    faces_std=face_recognition.face_locations(img4)
    encode_cur_frame= face_recognition.face_encodings(img4 , faces_std)

    # face matching  and face distance
    for encodeface , faceloke in zip(encode_cur_frame , faces_std):
        matche=face_recognition.compare_faces(encode_k ,encodeface)
        # print(matche)
        facedis= face_recognition.face_distance(encode_k ,encodeface)
        # print(facedis) 
        match_index=np.argmin(facedis)
        # print(match_index)
        if matche[match_index]:
            name = std_name[match_index].upper()
            print(name)
            
    #calling the funtion 
        attendance(name)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        rec1=cv2.rectangle(img4, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(rec1, str(std_name[match_index].upper()) , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)



    #  Display
    cv2.imshow('img', img4)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()



# making the report dir && move the csv file
name = "attendance_report"
for fil in os.listdir():
    if fil != name:
        os.mkdir(name)
        break
path_of_fold=os.getcwd()
for file in os.listdir():
    if file.endswith('.csv'):
        shutil.move(file , os.path.join(path_of_fold , name))