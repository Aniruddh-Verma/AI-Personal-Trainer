import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Curl Counter Vaariables 
counter = 0 
stage = None
      
def calculate_squats_angle(a,b,c):
      a=np.array(a) #First
      b=np.array(b) # mid
      c=np.array(c) #End

      radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
      angle = np.abs(radians*180.0/ np.pi) 

      if angle > 180.0:
            angle = 360-angle
      return angle

# Double Arm Curling
def calculate_both_leg_squats(left=[],right=[]):
      left_angle = calculate_squats_angle(left[0],left[1],left[2])
      right_angle = calculate_squats_angle(right[0],right[1],right[2])
      return left_angle,right_angle

# Getting Coordinates of joints 
def get_coords(landmarks,point):
      return  [landmarks[point.value].x,landmarks[point.value].y]

# Visualizing Text
def render_squats(image,points,win_size=[640,480]):
      white = (255,255,255)
      cv2.putText(image,f'{points["Knee"]:.1f}',tuple(np.multiply(points["Knee"],win_size).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5,white,2, cv2.LINE_AA)
      cv2.circle(image,tuple(np.multiply(points["Hip"],[640,480]).astype(int)),5,white,cv2.FILLED)
      cv2.circle(image,tuple(np.multiply(points["Knee"],[640,480]).astype(int)),5,white,cv2.FILLED)
      cv2.circle(image,tuple(np.multiply(points["Ankle"],[640,480]).astype(int)),5,white,cv2.FILLED)
      cv2.line(image,tuple(np.multiply(points["Hip"],[640,480]).astype(int)),tuple(np.multiply(points["Knee"],[640,480]).astype(int)),white,2)
      cv2.line(image,tuple(np.multiply(points["Ankle"],[640,480]).astype(int)),tuple(np.multiply(points["Knee"],[640,480]).astype(int)),white,2)

     

def show_excercise(video, ex=3, min_detection_confidence=0.5, min_tracking_confidence=0.5, **points[]):
      global counter,stage
      with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:    #importing Pose model as a variable pose
            while video.isOpened():       
                  img = video.read()  
                  
                  # RECOLOR IMAGE
                  image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                  image.flags.writeable = False

                  # Make Detection
                  results = pose.process(image)

                  # recoloring image back to BGR 
                  image.flags.writeable = True
                  image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

                  # Extracting Landmarks 
                  try:


                        landmarks=results.pose_landmarks.landmark
                        if landmarks != None:
                              # Getting Coordinates
                              joints = {}
                              if len(points) == 3 and ex==1:
                                    for name, point in points.items():
                                          joints[name] = get_coords(landmarks,point)
                                    
                                    angle = calculate_squats_angle(joints['Hip'],joints['Knee'],joints['Ankle'])
                                    
                                    #Visualize Text 
                                    render_squats(img,results,points)

                        
                                    # Curl Counter Logic 
                                    if angle > 160:
                                          stage= "Down" 
                                    if angle < 55 and stage == "Down":
                                          stage= "Up"
                                          counter += 1
                                          print(counter)
                              

                              if len(points) == 6 and ex==2:
                                    left_hip = [landmarks[points[0].value].x,landmarks[points[0].value].y]
                                    left_knee = [landmarks[points[1].value].x,landmarks[points[1].value].y]
                                    left_ankle= [landmarks[points[2].value].x,landmarks[points[2].value].y]
                                    right_hip = [landmarks[points[3].value].x,landmarks[points[3].value].y]
                                    right_knee = [landmarks[points[4].value].x,landmarks[points[4].value].y]
                                    right_ankle = [landmarks[points[5].value].x,landmarks[points[5].value].y]

                              # Calculating Angles
                              left_angle,right_angle = calculate_both_leg_squats([left_ankle,left_knee,left_hip],[right_ankle,right_knee,right_hip])
                              cv2.putText(image,f'{left_angle:.1f}',
                                          tuple(np.multiply(left_knee,[640,480]).astype(int)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2, cv2.LINE_AA)
                              cv2.putText(image,f'{right_angle:.1f}',
                                          tuple(np.multiply(right_knee,[640,480]).astype(int)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2, cv2.LINE_AA)
                              cv2.circle(image,tuple(np.multiply(left_knee,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.circle(image,tuple(np.multiply(left_ankle,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.circle(image,tuple(np.multiply(left_hip,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.line(image,tuple(np.multiply(left_knee,[640,480]).astype(int)),tuple(np.multiply(left_hip,[640,480]).astype(int)),(255,255,255),2)
                              cv2.line(image,tuple(np.multiply(left_ankle,[640,480]).astype(int)),tuple(np.multiply(left_knee,[640,480]).astype(int)),(255,255,255),2)
                              cv2.circle(image,tuple(np.multiply(right_knee,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.circle(image,tuple(np.multiply(right_ankle,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.circle(image,tuple(np.multiply(right_hip,[640,480]).astype(int)),5,(255,255,255),cv2.FILLED)
                              cv2.line(image,tuple(np.multiply(right_knee,[640,480]).astype(int)),tuple(np.multiply(right_ankle,[640,480]).astype(int)),(255,255,255),2)
                              cv2.line(image,tuple(np.multiply(right_hip,[640,480]).astype(int)),tuple(np.multiply(right_knee,[640,480]).astype(int)),(255,255,255),2)

                              if left_angle > 160 and right_angle > 160:
                                    stage= "Down"
                              if left_angle < 55 and right_angle < 55 and stage == "Down":
                                    stage= "Up"
                                    counter += 1
                                    print(counter)  


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

