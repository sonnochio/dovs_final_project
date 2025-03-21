import cv2
import numpy as np

def perspective_transform(file_name='assets/raw/large.jpg'):
    file_name=file_name
    # Read the image
    img = cv2.imread(file_name)
    if img is None:
        print("Error: Could not read the image.")
        return

    # Create a copy of the image
    img_copy = img.copy()
    
    # List to store the four points
    points = []
    
    # Mouse callback function to select points
    def select_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                points.append([x, y])
                
                # Display current position
                print(f"Point {len(points)} selected: ({x}, {y})")
                
                # If 4 points are selected, calculate the transform
                if len(points) == 4:
                    print("All points selected. Press any key to apply the transform.")
            
            # Redraw the image with the points
            img_display = img_copy.copy()
            if len(points) > 1:
                for i in range(len(points)-1):
                    cv2.line(img_display, tuple(points[i]), tuple(points[i+1]), (255, 0, 0), 2)
                if len(points) == 4:
                    cv2.line(img_display, tuple(points[3]), tuple(points[0]), (255, 0, 0), 2)
            
            cv2.imshow('Select four points', img_display)
    
    # Display the image and set up the callback
    cv2.imshow('Select four points', img_copy)
    cv2.setMouseCallback('Select four points', select_points)
    
    # Wait for a key press
    cv2.waitKey(0)
    
    # If we have 4 points, perform the perspective transform
    if len(points) == 4:
        # Convert to numpy array
        points = np.array(points, dtype=np.float32)
        
        # Calculate the width and height of the new image
        # We'll find the maximum width and height of the quadrilateral
        width_top = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))
        width_bottom = np.sqrt(((points[2][0] - points[3][0]) ** 2) + ((points[2][1] - points[3][1]) ** 2))
        width = max(int(width_top), int(width_bottom))
        
        height_left = np.sqrt(((points[3][0] - points[0][0]) ** 2) + ((points[3][1] - points[0][1]) ** 2))
        height_right = np.sqrt(((points[2][0] - points[1][0]) ** 2) + ((points[2][1] - points[1][1]) ** 2))
        height = max(int(height_left), int(height_right))
        
        # Create destination points
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(points, dst)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(img, matrix, (width, height))
        
        # Display the warped image and save it
        cv2.imshow('Warped Image', warped)
        cv2.imwrite(f'assets/warped/{file_name[11:]}', warped)
        print("Transformed image saved as 'warped_image.jpg'")
    
        
        # Wait for a key press and then save the image
        key = cv2.waitKey(0)

            
    # Clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    perspective_transform()