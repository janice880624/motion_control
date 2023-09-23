import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    min_detection_confidence=0.5,
    enable_segmentation=True,    
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, img = cap.read()

        if not ret:
            print("Cannot receive frame")
            break

        img = cv2.flip(img, 1)
        # img1 = cv2.resize(img,(680,480))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img2)

        try:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            
            if results.pose_landmarks:
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
                
                if left_hip == 0 or right_hip == 0:
                    display_text = "stand far away"
                    box_color = (255, 215, 0)
                elif abs(left_hip - right_hip) < 0.005: 
                    display_text = "same height"
                    box_color = (34, 139, 34)
                else:
                    display_text = "difference height"
                    box_color = (250, 128, 114)
                
                cv2.rectangle(img, (10, 10), (350, 60), box_color, -1) 
                cv2.putText(img, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass

        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('output', img)
        if cv2.waitKey(5) == ord('q'):
            break     
cap.release()
cv2.destroyAllWindows()
