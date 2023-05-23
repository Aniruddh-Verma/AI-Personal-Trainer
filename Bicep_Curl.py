import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Curl Counter Vaariables 
counter = 0 
stage = None


# Single Arm Curling      
def calculate_bicep_curl_angle(a,b,c):
      a=np.array(a) #First
      b=np.array(b) # mid
      c=np.array(c) #End

      radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
      angle = np.abs(radians*180.0/ np.pi) 

      if angle > 180.0:
            angle = 360-angle
      return angle

# Double Arm Curling
def both_bicep_curl_angle(left=[],right=[]):
      left_angle = calculate_bicep_curl_angle(left[0],left[1],left[2])
      right_angle = calculate_bicep_curl_angle(right[0],right[1],right[2])
      print(f'Left Angle: {left_angle} Right Angle: {right_angle}')
      return left_angle,right_angle

# Getting Coordinates of joints 
def get_coords(landmarks,point):
      return  [landmarks[point.value].x,landmarks[point.value].y]

# Visualizing Text
def render_bicep_curl(image,points,win_size=[640,480]):
      white = (255,255,255)
      cv2.putText(image,f'{points["elbow"]:.1f}',tuple(np.multiply(points["elbow"],win_size).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,white,2, cv2.LINE_AA)
      cv2.circle(image,tuple(np.multiply(points["elbow"],[640,480]).astype(int)),5,white,cv2.FILLED)
      cv2.circle(image,tuple(np.multiply(points["shoulder"],[640,480]).astype(int)),5,white,cv2.FILLED)
      cv2.circle(image,tuple(np.multiply(points["wrist"],[640,480]).astype(int)),5,white,cv2.FILLED)
      cv2.line(image,tuple(np.multiply(points["elbow"],[640,480]).astype(int)),tuple(np.multiply(points["shoulder"],[640,480]).astype(int)),white,2)
      cv2.line(image,tuple(np.multiply(points["wrist"],[640,480]).astype(int)),tuple(np.multiply(points["elbow"],[640,480]).astype(int)),white,2)

     

def show_excercise(video, ex=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, *points):
      global counter,stage
      with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:    #importing Pose model as a variable pose
            while video.isOpened():       
                  state,image = video.read()  
                  
                  # RECOLOR IMAGE
                  image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                  image.flags.writeable = False

                  # Make Detection
                  results = pose.process(image)

                  # recoloring image back to BGR 
                  image.flags.writeable = True
                  image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

                  # Extracting Landmarks 
                  try:


                        landmarks=results.pose_landmarks.landmark
                        # print(f'Landmarks : {landmarks}')
                        if len(landmarks)>0:
                              # Getting Coordinates
                              joints = {}
                              if len(points) == 3 and ex==1:
                                    for name, point in points.items():
                                          joints[name] = get_coords(landmarks,point)
                                    
                                    angle = calculate_bicep_curl_angle(joints['shoulder'],joints['elbow'],joints['wrist'])
                                    
                                    #Visualize Text 
                                    render_bicep_curl(img,results,points)

                        
                                    # Curl Counter Logic 
                                    if angle > 160:
                                          stage= "Down" 
                                    if angle < 55 and stage == "Down":
                                          stage= "Up"
                                          counter += 1
                                          print(counter)
                              

                              if len(points) == 6 and ex==2:
                                    left_shoulder = [landmarks[points[0].value].x,landmarks[points[0].value].y]
                                    left_elbow = [landmarks[points[1].value].x,landmarks[points[1].value].y]
                                    left_wrist = [landmarks[points[2].value].x,landmarks[points[2].value].y]
                                    right_shoulder = [landmarks[points[3].value].x,landmarks[points[3].value].y]
                                    right_elbow = [landmarks[points[4].value].x,landmarks[points[4].value].y]
                                    right_wrist = [landmarks[points[5].value].x,landmarks[points[5].value].y]
                              print(f'Left Shoulder: {left_shoulder} Left Elbow: {left_elbow} Left Wrist: {left_wrist}')
                              # Calculating Angles
                              left_angle,right_angle = both_bicep_curl_angle([left_shoulder,left_elbow,left_wrist],[right_shoulder,right_elbow,right_wrist])
                              cv2.putText(image,f'{left_angle:.1f}',
                                          tuple(np.multiply(left_elbow,[640,480]).astype(int)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2, cv2.LINE_AA)
                              cv2.putText(image,f'{right_angle:.1f}',
                                          tuple(np.multiply(right_elbow,[640,480]).astype(int)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2, cv2.LINE_AA)
                              cv2.circle(image,tuple(np.multiply(left_elbow,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.circle(image,tuple(np.multiply(left_shoulder,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.circle(image,tuple(np.multiply(left_wrist,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.line(image,tuple(np.multiply(left_elbow,[640,480]).astype(int)),tuple(np.multiply(left_shoulder,[640,480]).astype(int)),(255,255,255),2)
                              cv2.line(image,tuple(np.multiply(left_wrist,[640,480]).astype(int)),tuple(np.multiply(left_elbow,[640,480]).astype(int)),(255,255,255),2)
                              cv2.circle(image,tuple(np.multiply(right_elbow,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.circle(image,tuple(np.multiply(right_shoulder,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.circle(image,tuple(np.multiply(right_wrist,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.line(image,tuple(np.multiply(right_elbow,[640,480]).astype(int)),tuple(np.multiply(right_shoulder,[640,480]).astype(int)),(255,255,255),2)
                              cv2.line(image,tuple(np.multiply(right_wrist,[640,480]).astype(int)),tuple(np.multiply(right_elbow,[640,480]).astype(int)),(255,255,255),2)
                              

                              # Curl Counter Logi
                              if left_angle > 160 and right_angle > 160:
                                    stage= "Down"
                              if left_angle < 55 and right_angle < 55 and stage == "Down":
                                    stage= "Up"
                                    counter += 1
                                    print(counter)  
                        else: print("No pose detected")

                  except:
                        pass      

                  # Render curl counter
                  # Setup status box
                  cv2.rectangle(image, (0,0), (235,80), (249,186,30), -1)           #230,20,150
                  
                  # Rep data
                  cv2.putText(image, 'REPS', (15,12), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                  cv2.putText(image, str(counter), (10,60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                  
                  # Stage data
                  cv2.putText(image, 'STAGE', (75,12), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                  cv2.putText(image, stage, 
                              (60,60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)


                  # Render Detection
                  # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                  #                         mp_drawing.DrawingSpec(color=(82,49,111), thickness=2, circle_radius=2),
                  #                         mp_drawing.DrawingSpec(color=(184,29,95), thickness=2, circle_radius=2)
                  #                         )

                                          

                  cv2.imshow('video',image)

                  if cv2.waitKey(10) & 0xFF == 27: 
                        video.release()
                        cv2.destroyAllWindows()
                        break

def main():
      global counter
      counter = 0
      video = cv2.VideoCapture(0)  
# left bicep curl
# show_excercise(video,1,mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.LEFT_ELBOW,mp_pose.PoseLandmark.LEFT_WRIST)
# right bicep curl
# show_excercise(video,mp_pose.PoseLandmark.RIGHT_SHOULDER,mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.RIGHT_WRIST)
# both bicep 

      show_excercise(video,
                  2,
                  .5,
                  .5,
                  mp_pose.PoseLandmark.LEFT_SHOULDER,
                  mp_pose.PoseLandmark.LEFT_ELBOW,
                  mp_pose.PoseLandmark.LEFT_WRIST,
                  mp_pose.PoseLandmark.RIGHT_SHOULDER,
                  mp_pose.PoseLandmark.RIGHT_ELBOW,
                  mp_pose.PoseLandmark.RIGHT_WRIST)
