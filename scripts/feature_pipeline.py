import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_parking_bays(image_path='assets/warped/large.jpg'):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Apply Sobel filter for edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
    
    # Step 3: Apply morphological operations to clean up
    kernel = np.ones((4, 4), np.uint8)
    # have to use larger kernal here
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 4: Apply Watershed algorithm
    # Distance transform
    dist_transform = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find unknown region
    sure_bg = cv2.dilate(255 - binary, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(img, markers)
    
    # Create a copy of original image
    img_result = img.copy()
    img_result[markers == -1] = [0, 0, 255]  # Mark boundaries in red
    
    # Count regions (excluding background and boundaries)
    num_bays = len(np.unique(markers)) - 2  # -1 (boundaries) and 1 (background)
    
    # Step 5: Estimate parking bay organization (rows and columns)
    # Find contours of parking bays
    markers_copy = markers.copy()
    markers_copy[markers_copy <= 1] = 0  # Remove background and boundaries
    markers_copy = markers_copy.astype(np.uint8)
    contours, _ = cv2.findContours(markers_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    min_area = 100
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Get bounding boxes for each parking bay
    bboxes = [cv2.boundingRect(c) for c in valid_contours]
    
    # Extract all centerpoints
    centers = [(int(x + w/2), int(y + h/2)) for (x, y, w, h) in bboxes]
    
    # Draw centers on the image
    for center in centers:
        cv2.circle(img_result, center, 5, (0, 255, 0), -1)
    
    # Estimate rows and columns using K-means clustering if we have enough points
    if len(centers) > 2:
        # Convert centers to numpy array
        centers_array = np.array(centers, dtype=np.float32)
        
        # Sort centers by y-coordinate (row)
        centers_sorted_y = sorted(centers, key=lambda c: c[1])
        
        # Calculate average height of a bay
        y_coords = [c[1] for c in centers_sorted_y]
        y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        avg_height = np.mean(y_diffs) if y_diffs else 0

#============================================================
        # Change to 30% instead, can play around with this 
        row_group_factor = 0.9  # e.g., 0.3 = 30%
        threshold = row_group_factor * avg_height
#============================================================

        # Group into rows
        row_clusters = []
        current_row = [centers_sorted_y[0]]
        
        for i in range(1, len(centers_sorted_y)):
            if abs(centers_sorted_y[i][1] - current_row[0][1]) <= threshold:
                current_row.append(centers_sorted_y[i])
            else:
                row_clusters.append(current_row)
                current_row = [centers_sorted_y[i]]
        
        if current_row:
            row_clusters.append(current_row)
        
        # Count columns in each row
        columns = [len(row) for row in row_clusters]
        avg_columns = int(np.mean(columns)) if columns else 0
        num_rows = len(row_clusters)
        
        # Create a simple visualization of the parking lot structure
        structure_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        
        for i, row in enumerate(row_clusters):
            for j, center in enumerate(sorted(row, key=lambda c: c[0])):
                cv2.circle(structure_img, center, 10, (0, 255, 0), -1)
                cv2.putText(structure_img, f"{i+1},{j+1}", 
                           (center[0]-20, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
    else:
        num_rows = 0
        avg_columns = 0
        structure_img = np.zeros_like(img)

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(231), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(232), plt.imshow(gray, cmap='gray'), plt.title('Grayscale')
    plt.subplot(233), plt.imshow(sobel, cmap='gray'), plt.title('Sobel Filter')
    plt.subplot(234), plt.imshow(binary, cmap='gray'), plt.title('Binary')
    plt.subplot(235), plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)), plt.title(f'Detected Bays: {num_bays}')
    plt.subplot(236), plt.imshow(cv2.cvtColor(structure_img, cv2.COLOR_BGR2RGB)), plt.title(f'Structure: ~{num_rows} rows x ~{avg_columns} columns')
    
    plt.tight_layout()
    plt.show()
    
    return num_rows, avg_columns, num_bays

#Structure: ~{num_rows} rows x ~{avg_columns} columns -> to plug in to digital twin


if __name__ == "__main__":
    rows, cols, bays = detect_parking_bays('assets/warped/large.jpg')
    print(f"Detected approximately {rows} rows and {cols} columns of parking bays.")
    print(f"Total parking bays detected: {bays}")