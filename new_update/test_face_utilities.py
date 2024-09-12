from face_utilities import Face_utilities
import cv2
from imutils import face_utils
import numpy as np
import time


def flow_process(frame, i, fu, last_rects, last_shape, last_age, last_gender):
    display_frame = frame.copy()  
    rects = last_rects
    age = last_age
    gender = last_gender
    shape = last_shape
    
    # convert the frame to gray scale before performing face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # get all faces as rectangles every 3 frames
    if i % 3 == 0:
        rects = fu.face_detection(frame)
    
    # check if there is any face in the frame, if not, show the frame and move to the next frame
    if len(rects) == 0:
        return display_frame, None, last_rects, last_age, last_gender, last_shape
    
    # draw face rectangle, only grab one face in the frame
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # crop the face from frame
    face = frame[y:y+h, x:x+w]
    
    if i % 6 == 0:
        # detect age and gender and put it into the frame every 6 frames
        age, gender = fu.age_gender_detection(face)
        
    overlay_text = "%s, %s" % (gender, age)
    cv2.putText(display_frame, overlay_text, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    if i % 3 == 0:
        # get 68 facial landmarks and draw it into the face every 3 frames
        shape = fu.get_landmarks(frame, "68")
    
    for (lx, ly) in shape: 
        cv2.circle(face, (lx, ly), 1, (0, 0, 255), -1)
        
    # get the mask of the face
    remapped_landmarks = fu.facial_landmarks_remap(shape)
    mask = np.zeros((face.shape[0], face.shape[1]))
    cv2.fillConvexPoly(mask, remapped_landmarks[0:27], 1) 
    
    aligned_face = fu.face_alignment(frame, shape)
    
    aligned_shape = fu.get_landmarks(aligned_face, "68")
    
    cv2.rectangle(aligned_face, (aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                  (aligned_shape[12][0], aligned_shape[33][1]), (0, 255, 0), 0)
    cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                  (aligned_shape[48][0], aligned_shape[33][1]), (0, 255, 0), 0)
    
    # assign to last params
    last_rects = rects
    last_age = age
    last_gender = gender
    last_shape = shape
    
    return display_frame, aligned_face, last_rects, last_age, last_gender, last_shape


if __name__ == "__main__":
    
    cap = cv2.VideoCapture("1.mp4")
    fu = Face_utilities()
    i = 0
    last_rects = None
    last_shape = None
    last_age = None
    last_gender = None
    
    t = time.time()
    
    while True:
        # calculate time for each loop
        t0 = time.time()
        
        ret, frame = cap.read()
        
        if frame is None:
            print("End of video")
            break
        
        # Use flow_process to handle frame processing
        display_frame, aligned_face, last_rects, last_age, last_gender, last_shape = flow_process(
            frame, i, fu, last_rects, last_shape, last_age, last_gender
        )
        
        if aligned_face is None:
            cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("frame", frame)
            print(time.time() - t0)
            
            cv2.destroyWindow("face")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # display the frame and aligned face
        cv2.imshow("frame", display_frame)
        cv2.imshow("face", aligned_face)
        
        i += 1
        print(time.time() - t0)
        
        # waitKey to show the frame and break loop whenever 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    print(time.time() - t)
