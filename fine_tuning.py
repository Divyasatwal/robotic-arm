# might delete later
import cv2
import numpy as np

def nothing(x):
    pass

# Open camera
cap = cv2.VideoCapture(0)

# Create window and trackbars
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 400, 300)

# First red range (lower red: 0–10 hue)
cv2.createTrackbar("LH1", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("UH1", "Trackbars", 10, 180, nothing)

# Second red range (upper red: 160–180 hue)
cv2.createTrackbar("LH2", "Trackbars", 160, 180, nothing)
cv2.createTrackbar("UH2", "Trackbars", 180, 180, nothing)

# Saturation and Value ranges
cv2.createTrackbar("LS", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 150, 255, nothing)
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)

# Add Trackbars to Adjust Kernel Size Live--morphology
cv2.createTrackbar("Kernel Width", "Trackbars", 5, 20, nothing)
cv2.createTrackbar("Kernel Height", "Trackbars", 5, 20, nothing)


# trcakbar if i wanted to chnage the size of rectangele
cv2.createTrackbar("Min Area", "Trackbars", 500, 10000, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get values from trackbars
    lh1 = cv2.getTrackbarPos("LH1", "Trackbars")
    uh1 = cv2.getTrackbarPos("UH1", "Trackbars")
    lh2 = cv2.getTrackbarPos("LH2", "Trackbars")
    uh2 = cv2.getTrackbarPos("UH2", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    # Two red masks (because red wraps around hue circle)
    lower_red1 = np.array([lh1, ls, lv])
    upper_red1 = np.array([uh1, us, uv])
    lower_red2 = np.array([lh2, ls, lv])
    upper_red2 = np.array([uh2, us, uv])


    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    result = cv2.bitwise_and(frame, frame, mask=red_mask)




 # Get kernel sizes
    k_w = cv2.getTrackbarPos("Kernel Width", "Trackbars")
    k_h = cv2.getTrackbarPos("Kernel Height", "Trackbars")

# Ensure kernel is odd and ≥1
    if k_w % 2 == 0:
     k_w += 1
    if k_h % 2 == 0:
     k_h += 1
    k_w = max(1, k_w)
    k_h = max(1, k_h)

    kernel = np.ones((k_h, k_w), np.uint8)

    mask_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(frame, frame, mask=mask_cleaned)


 # ➕ Show kernel size on frame
    cv2.putText(result, f"Kernel: ({k_w}, {k_h})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

         # Find contours from cleaned mask
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
        if area > min_area :  # Filter small noise contours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result, f"Area: {int(area)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show output
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Detected Red Area", result)

    
    if cv2.waitKey(1) & 0xFF == ord('s'):  # press 's' to save
    
     print("Final HSV Values:")
     print(f"Lower Red 1: ({lh1}, {ls}, {lv})")
     print(f"Upper Red 1: ({uh1}, {us}, {uv})")
     print(f"Lower Red 2: ({lh2}, {ls}, {lv})")
     print(f"Upper Red 2: ({uh2}, {us}, {uv})")
     
    
     print(f"Kernel Size: ({k_w}, {k_h})")

     break
   
cap.release()
cv2.destroyAllWindows()
