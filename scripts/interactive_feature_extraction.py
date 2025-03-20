import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class ImageProcessor:
    def __init__(self, file_path='assets/warped/large.jpg'):
        self.original_img = cv2.imread(file_path)
        if self.original_img is None:
            raise FileNotFoundError(f"Could not read image: {file_path}")
        self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.working_img = self.original_img.copy()
        self.gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2GRAY)
        
    def reset_image(self):
        """Reset working image to original"""
        self.working_img = self.original_img.copy()
        self.gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_RGB2GRAY)
        return self.working_img
    
    def apply_laplacian(self, kernel_size=3, scale=1.0, delta=0):
        """Apply Laplacian filter for edge detection"""
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_img, (3, 3), 0)
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta)
        
        # Convert back to uint8
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Create colored output for visualization
        result = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2RGB)
        
        return result, laplacian
    
    def apply_sobel(self, ksize=3, scale=1, delta=0, dx=1, dy=1):
        """Apply Sobel filter for edge detection"""
        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize += 1
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(self.gray_img, (3, 3), 0)
        
        # Apply Sobel
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, dx, 0, ksize=ksize, scale=scale, delta=delta)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, dy, ksize=ksize, scale=scale, delta=delta)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Create colored output for visualization
        result = cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)
        
        return result, magnitude
    
    def apply_unsharp_mask(self, kernel_size=5, sigma=1.0, amount=1.0, threshold=0):
        """Apply unsharp mask filter to sharpen the image"""
        # Convert to float for processing
        image_float = self.gray_img.astype(np.float32) / 255.0
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), sigma)
        
        # Calculate unsharp mask
        unsharp_mask = image_float - blurred
        
        # Apply amount and threshold
        sharpened = image_float + amount * unsharp_mask
        sharpened = np.clip(sharpened, 0, 1)
        
        # Apply threshold
        mask = np.abs(unsharp_mask) > threshold
        sharpened = np.where(mask, sharpened, image_float)
        
        # Convert back to uint8
        sharpened = (sharpened * 255).astype(np.uint8)
        
        # Create colored output for visualization
        result = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        
        return result, sharpened
    
    def apply_hough_lines(self, rho=1, theta=np.pi/180, threshold=100, min_line_length=50, max_line_gap=10):
        """Apply Hough transform for line detection"""
        # Make a copy of the original image
        result = self.original_img.copy()
        
        # Apply edge detection (Canny)
        edges = cv2.Canny(self.gray_img, 50, 150, apertureSize=3)
        
        # Apply Hough Lines transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=rho, 
            theta=theta, 
            threshold=threshold, 
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        # Draw detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Count the number of lines
        line_count = 0 if lines is None else len(lines)
        
        return result, line_count
    
    def apply_watershed(self, dist_threshold=0.7, kernel_size=3):
        """Apply watershed algorithm with distance transform for segmentation"""
        # Apply thresholding to isolate potential parking spaces
        _, binary = cv2.threshold(self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal with morphological operations
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area with distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that background is not 0, but 1
        markers = markers + 1
        
        # Mark the unknown region with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(cv2.cvtColor(self.original_img, cv2.COLOR_RGB2BGR), markers)
        
        # Create a copy of the original image for visualization
        result = self.original_img.copy()
        
        # Highlight boundaries and count regions
        # Exclude background (1) and watershed boundaries (-1)
        region_count = len(np.unique(markers)) - 2  # Excluding -1 (boundaries) and 1 (background)
        result[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        
        # Create a colored visualization of the segments
        markers_display = np.zeros_like(self.original_img)
        unique_markers = np.unique(markers)
        unique_markers = unique_markers[unique_markers > 1]  # Skip background and boundaries
        
        # Assign random colors to each segment
        for marker in unique_markers:
            markers_display[markers == marker] = np.random.randint(0, 255, 3)
        
        # Mark boundaries
        markers_display[markers == -1] = [255, 0, 0]
        
        # Blend with original image for better visualization
        blended = cv2.addWeighted(result, 0.7, markers_display, 0.3, 0)
        
        return blended, region_count
    
    def count_parking_bays(self, area_threshold=500):
        """Count potential parking bays using contour detection"""
        # Apply thresholding
        _, binary = cv2.threshold(self.gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]
        
        # Create a copy of the original image for visualization
        result = self.original_img.copy()
        
        # Draw contours and count
        cv2.drawContours(result, valid_contours, -1, (0, 255, 0), 2)
        
        # Add text with count
        cv2.putText(result, f"Parking Bays: {len(valid_contours)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result, len(valid_contours)
    
    def create_interactive_display(self):
        """Create an interactive display for parameter adjustment"""
        plt.figure(figsize=(12, 8))
        
        # Function selection
        functions = {
            'Original': lambda: (self.original_img, "Original Image"),
            'Laplacian': lambda: self.apply_laplacian(),
            'Sobel': lambda: self.apply_sobel(),
            'Unsharp Mask': lambda: self.apply_unsharp_mask(),
            'Hough Lines': lambda: self.apply_hough_lines(),
            'Watershed': lambda: self.apply_watershed(),
            'Count Parking Bays': lambda: self.count_parking_bays()
        }
        
        current_function = 'Original'
        
        # Get initial image and title
        img, title = functions[current_function]()
        
        # Display image
        ax_img = plt.subplot(111)
        img_plot = ax_img.imshow(img)
        plt.title(title)
        
        # Create parameter sliders based on current function
        sliders = {}
        
        def update_sliders(function_name):
            # Clear existing sliders
            for slider in sliders.values():
                slider.ax.remove()
            sliders.clear()
            
            # Define parameters for each function
            params = {
                'Laplacian': [
                    ('Kernel Size', 1, 31, 2, 3),
                    ('Scale', 0.1, 5.0, 0.1, 1.0),
                    ('Delta', 0, 255, 1, 0)
                ],
                'Sobel': [
                    ('Kernel Size', 1, 31, 2, 3),
                    ('Scale', 0.1, 5.0, 0.1, 1.0),
                    ('Delta', 0, 255, 1, 0),
                    ('dx', 0, 1, 1, 1),
                    ('dy', 0, 1, 1, 1)
                ],
                'Unsharp Mask': [
                    ('Kernel Size', 3, 15, 2, 5),
                    ('Sigma', 0.1, 5.0, 0.1, 1.0),
                    ('Amount', 0.1, 5.0, 0.1, 1.0),
                    ('Threshold', 0, 0.5, 0.01, 0)
                ],
                'Hough Lines': [
                    ('Rho', 0.1, 10.0, 0.1, 1.0),
                    ('Theta (degrees)', 0.1, 5.0, 0.1, 1.0),
                    ('Threshold', 10, 300, 1, 100),
                    ('Min Line Length', 10, 300, 1, 50),
                    ('Max Line Gap', 1, 100, 1, 10)
                ],
                'Watershed': [
                    ('Distance Threshold', 0.1, 1.0, 0.05, 0.7),
                    ('Kernel Size', 1, 11, 2, 3)
                ],
                'Count Parking Bays': [
                    ('Area Threshold', 50, 5000, 50, 500)
                ]
            }
            
            # Create sliders if needed
            if function_name in params:
                for i, (name, min_val, max_val, step, default) in enumerate(params[function_name]):
                    ax = plt.axes([0.25, 0.02 + i * 0.03, 0.65, 0.02])
                    sliders[name] = Slider(ax, name, min_val, max_val, valinit=default, valstep=step)
                
                # Create update function for sliders
                def update(val):
                    # Get parameter values
                    kwargs = {name.lower().replace(' ', '_'): slider.val for name, slider in sliders.items()}
                    
                    # Apply function
                    if function_name == 'Laplacian':
                        img, _ = self.apply_laplacian(**kwargs)
                    elif function_name == 'Sobel':
                        img, _ = self.apply_sobel(**kwargs)
                    elif function_name == 'Unsharp Mask':
                        img, _ = self.apply_unsharp_mask(**kwargs)
                    elif function_name == 'Hough Lines':
                        # Convert theta from degrees to radians
                        if 'theta' in kwargs:
                            kwargs['theta'] = kwargs['theta_degrees'] * np.pi / 180
                            del kwargs['theta_degrees']
                        img, line_count = self.apply_hough_lines(**kwargs)
                        plt.title(f"Hough Lines: {line_count} lines detected")
                    elif function_name == 'Watershed':
                        img, region_count = self.apply_watershed(**kwargs)
                        plt.title(f"Watershed: {region_count} regions detected")
                    elif function_name == 'Count Parking Bays':
                        img, bay_count = self.count_parking_bays(**kwargs)
                        plt.title(f"Parking Bays: {bay_count} bays detected")
                    
                    # Update image
                    img_plot.set_data(img)
                    plt.draw()
                
                # Connect sliders to update function
                for slider in sliders.values():
                    slider.on_changed(update)
        
        # Create function selection buttons
        ax_buttons = plt.axes([0.01, 0.02, 0.15, 0.3])
        buttons = {}
        
        for i, name in enumerate(functions.keys()):
            button_ax = plt.axes([0.01, 0.32 - i * 0.03, 0.15, 0.02])
            buttons[name] = Button(button_ax, name)
            
        # Function to handle button clicks
        def on_button_click(event):
            nonlocal current_function
            for name, button in buttons.items():
                if event.inaxes == button.ax:
                    current_function = name
                    break
            
            # Get new image and title
            if current_function == 'Original':
                img, title = self.original_img, "Original Image"
            else:
                img, extra = functions[current_function]()
                if current_function == 'Hough Lines':
                    title = f"{current_function}: {extra} lines detected"
                elif current_function == 'Watershed':
                    title = f"{current_function}: {extra} regions detected"
                elif current_function == 'Count Parking Bays':
                    title = f"{current_function}: {extra} bays detected"
                else:
                    title = current_function
            
            # Update image and title
            img_plot.set_data(img)
            plt.title(title)
            
            # Update sliders
            update_sliders(current_function)
            
            plt.draw()
        
        # Connect buttons to click handler
        for button in buttons.values():
            button.on_clicked(on_button_click)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)
        plt.show()

# Create and run the application
if __name__ == "__main__":
    file_name = 'assets/warped/large.jpg'
    try:
        processor = ImageProcessor(file_name)
        processor.create_interactive_display()
    except FileNotFoundError as e:
        print(e)
        print("Please check that the file path is correct: ", file_name)