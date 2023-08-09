import numpy as np
import cv2 as cv

cap = cv.VideoCapture(2)

# Change this to correct conversion factor
pixel_to_realworld_ratio = 0.001233 * 20.51*1.28

# Set a minimum area for contours
min_contour_area = 5000  # Adjust this value accordingly

# List to hold the sizes
sizes = []

# Current real size
current_real_size = 0

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Check if frame is not empty
    if frame is None or frame.size == 0:
        print("Empty frame. Skipping...")
        continue

    # Crop the frame
    cropped_frame = frame[40:450, 2:620]

    # Check if cropped frame is not empty
    if cropped_frame.size == 0:
        print("Cropped frame is empty. Skipping...")
        continue

    # Convert color image to grayscale
    gray = cv.cvtColor(cropped_frame, cv.COLOR_BGR2GRAY)

    # Apply thresholding
    ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

    # Detect contours in the thresholded image
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get the pixel area of the contour
        pixel_area = cv.contourArea(contour)

        # Ignore small contours
        if pixel_area < min_contour_area:
            continue

        # Convert to real world area and round to 1 decimal place
        real_size = round(pixel_area * pixel_to_realworld_ratio, 1)
        current_real_size = real_size

        # You can also draw the contour on the image for visualization
        cv.drawContours(cropped_frame, [contour], -1, (0,255,0), 3)

    # Get the pressed key
    key = cv.waitKey(1)

    # If 'r' is pressed, record the current size
    if key == ord('r'):
        sizes.append(current_real_size)
        print("Size recorded: ", current_real_size)

    # If spacebar is pressed, reset the list
    if key == 32:  # ASCII value for spacebar
        sizes = []
        print("Sizes reset")

    # If total size is over 1000, give a notification and reset
    #if sum(sizes) >= 1000:
     #   print("Total size is over 1000!")
      #  print("Total size: ", round(sum(sizes), 1))
       # sizes = []  # Reset the list to start over

    # Display total size and current size on every frame
    cv.putText(cropped_frame, f"Total size: {round(sum(sizes)/10000, 1)}", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv.putText(cropped_frame, f"Current size: {current_real_size}", (30, 70), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv.imshow('frame', cropped_frame)

    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
