from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('./Videos/fire.mp4')  # Use webcam (use a video file path instead if needed)
cap.set(3, 1280)  # Set frame width
cap.set(4, 720)  # Set frame height

# Load the YOLO model
model = YOLO('./Yolo-Weights/yolov8l.pt')  # Adjust the path to your YOLO model
model2 = YOLO('./Yolo-Weights/fire.pt')
classnames = ['fire']
while True:
    success, img = cap.read()  # Capture a frame

    # Run object detection on the frame
    results = model(img, stream=True)
    results2 = model2(img, stream=True)
    person_count = 0  # Initialize person counter for this frame

    for info in results2:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = int((confidence * 100))
            Class = int(box.cls[0])
            if confidence > 50:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),5)
                cv2.putText(img, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    for result in results:
        boxes = result.boxes  # Get detected bounding boxes
        for box in boxes:
            # Extract class ID
            cls_id = int(box.cls[0])  # Class ID (integer)

            # Check if the detected object is a person
            if model.names[cls_id] == "person":
                person_count += 1  # Increment the person counter

                # Draw bounding box for the person
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # cv2.circle(img, 2, 5, (255, 0, 255), 3)


                # Display label for the person
                label = f"Person"
                cv2.putText(img, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the Project Details
    cv2.putText(img, "Disaster Management Drone Project by Musab Jaffer and Rayyan (NSAKCET)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # Display the person count on the frame
    cv2.putText(img, f"People Count: {person_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Show the processed video feed
    cv2.imshow("", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
















