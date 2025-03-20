import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_filter_results(image_path='assets/warped/large.jpg'):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Apply Sobel filter
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply Unsharp mask
    gaussian = cv2.GaussianBlur(gray, (5, 5), 2.0)
    unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    
    # Apply Hough Lines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    hough_result = img_rgb.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Apply Watershed
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the unknown region with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    watershed_result = img_rgb.copy()
    markers = cv2.watershed(cv2.cvtColor(watershed_result, cv2.COLOR_RGB2BGR), markers)
    watershed_result[markers == -1] = [255, 0, 0]  # boundary in red

    # Display results
    plt.figure(figsize=(20, 10))
    
    plt.subplot(231), plt.imshow(img_rgb), plt.title('Original')
    plt.subplot(232), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
    plt.subplot(233), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel')
    plt.subplot(234), plt.imshow(unsharp_mask, cmap='gray'), plt.title('Unsharp Mask')
    plt.subplot(235), plt.imshow(hough_result), plt.title(f'Hough Lines ({0 if lines is None else len(lines)})')
    plt.subplot(236), plt.imshow(watershed_result), plt.title('Watershed')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_filter_results('assets/warped/large.jpg')