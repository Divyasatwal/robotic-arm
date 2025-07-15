import cv2
import sys
import numpy as np
print(" Starting ")

s = 0    # Open the laptop camera 

# InCase of external cam 
if len(sys.argv) > 1:
    s = int(sys.argv[1])

cap = cv2.VideoCapture(s)

if not cap.isOpened():
    print(" Cannot open camera")
    exit()
print(" Camera opened successfully")


# Reading frame from the webcam
try:
   while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
# Red color ranges
    lower_red1 = (0, 100, 70)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 100, 100)
    upper_red2 = (180, 255, 255)

# Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    


#   morphology 
    kernel = np.ones((5,5), np.uint8)
    mask_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# Show the red-only mask
    # cv2.imshow('Red Mask', red_mask)
    # cv2.imshow('mask_cleaned', red_mask)

# highlight  red region inoriginal image
    red_part = cv2.bitwise_and(frame, frame, mask=mask_cleaned)
    cv2.imshow('Red Objects Only', red_part)
    
    # Find contours in red mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
     area = cv2.contourArea(cnt)
     if area > 500:  
        x, y, w, h = cv2.boundingRect(cnt)
        #drawing a dot at the center
        cx = x + w // 2
        cy = y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)  # blue dot
        
       
        # Optional// For Arduino communication
        print(f"Red Object Center: ({cx}, {cy})")

# cropping remove later
        # cropped = frame[y:y+h, x:x+w]
        # cv2.imshow("Cropped Red Object", cropped)
    win_name = 'Live Camera '
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, frame)

 # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Clean up

finally:
 cap.release()
 cv2.destroyAllWindows()


