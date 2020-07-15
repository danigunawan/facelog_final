import cv2
import datetime
import os
import numpy
import skvideo.io
import ServerApiConstanst as apiConstanst
import time
from threading import Thread
capture_path = '/home/dsoft/Projects/FaceLog/FaceLog.Server/server/public/'
class Utitlities:
    def __init__(self):
        pass
    
    def create_video(self, file_name, frames): 
        write = skvideo.io.FFmpegWriter(file_name, inputdict={'-r': str(10), }, outputdict={'-r': str(10),})
        for i in range(len(frames)):
            write.writeFrame(frames[i])
        write.close()
        print("Created video: ", file_name)

    def get_name_with_proba(self, data):
        name = self.get_name(data)
        proba = data['proba']
        text = "{}_{:.2f}%".format(name, proba * 100)
        return text

    def get_name(self, data):
        name = data['name']
        return name

    def is_face_valid(self, data):
        x = data['startX'] 
        y = data['startY'] 
        w = data['endX'] 
        h = data['endY'] 
        if (w - x) * (h - y) > 10000:
            return True
        else:
            return False

    def get_crop_coordinate(self, data):
        frame = data['frame']
        high, width = frame.shape[:2]
        x = data['startX'] 
        y = data['startY'] 
        w = data['endX'] 
        h = data['endY'] 
        cr_x = x - int(width / 20) if x - int(width / 20) > 0 else x
        cr_y = y - int(high / 10) if y - int(high / 10) > 0 else y
        cr_w = w + int(width / 20) if w +int(width / 20) < width else w
        cr_h = h + int(high / 10) if h + int(high / 10) < high else h
        return cr_x, cr_y, cr_w, cr_h

    def save_train_data_image(self, file, name):
        print(file)
        train_dir = 'train_dir/' + name
        
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        file.save(os.path.join(train_dir, name +".jpg"))
        return name

    def capture_evidence(self, frame, name, camera, shouldCrop):
        time_now = str(datetime.datetime.now())
        day = time_now.split(' ')[0]
        time = time_now.split(' ')[1].split('.')[0]
        folder_path = '/evidence/' + day + '/' + name + '/' + camera
        evidence_path = capture_path + folder_path
        if not os.path.exists(evidence_path):
            os.makedirs(evidence_path)
        crop_image = frame
        
        if shouldCrop:
            cr_x, cr_y, cr_w, cr_h = self.get_crop_coordinate(data)
            crop_image = frame[cr_y: cr_h, cr_x: cr_w]
            
        if len(crop_image) != 0:
            file_name = folder_path + '/' + time + '.jpg'
            cv2.imwrite(capture_path + file_name, crop_image)
            print('Saved face: ', file_name)
        return file_name

    def draw_box(self, data, frame, display_text):
        x = data['startX']
        y = data['startY']
        w = data['endX']
        h = data['endY']
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 255), 2)
        if display_text == True:
            yW = y - 10 if y - 10 > 10 else y + 10
            cv2.putText(frame, self.get_name(data), (x, yW), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
       

    def draw_conner_box(self,frame,left,top,right,bottom,color,size= 30,thickness = 3):
        cv2.line(frame, (left+size,top), (left,top), color, thickness)
        cv2.line(frame, (left,top +size), (left,top), color, thickness)
        cv2.line(frame, (right-size,top), (right,top), color, thickness)
        cv2.line(frame, (right,top +size), (right,top), color, thickness)
        cv2.line(frame, (left,bottom-size), (left,bottom), color, thickness)
        cv2.line(frame, (left+size,bottom), (left,bottom), color, thickness)
        cv2.line(frame, (right,bottom-size), (right,bottom), color, thickness)
        cv2.line(frame, (right-size,bottom), (right,bottom), color, thickness)
        

    def draw_box_with_name(self, frame, top, bottom, right, left, color,name,color_name):
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.rectangle(frame, (left, top), (right, bottom), color, 1)
        #Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.5, color_name, 3)

    def check_valid_image_extension(self, file_name):
        return '.' in file_name and file_name.rsplit('.', 1)[1].lower() in apiConstanst.ALLOWED_EXTENSIONS

    def get_record_file_path(self):
        folder_path = apiConstanst.RECORD_PERSON_PATH
        current_time = str(datetime.datetime.now())
        date = current_time.split(' ')[0]
        time = current_time.split(' ')[1]
        save_folder = folder_path + '/' + date
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        name_record_file_mp4 = save_folder + '/' + str(time) + '.mp4'
        return name_record_file_mp4