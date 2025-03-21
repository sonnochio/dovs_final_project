# Personal Contributions

I was responsible for scoping and task 1-3 of the project, which mainly involves the python parts of the project. Specifically, I contributed to:

## Defining the Project Scope
I helped define the project’s aim of producing a digital twin of the car park to assist in downstream planning. This involved identifying the key steps (homography transformation, image pre-processing, and segmentation) to bridge camera footage with the planning algorithm.
## Task 1: Homography Transformation:
I developed the perspective transformation tool, allowing the selection of four corner points to correct the camera’s perspective. This step was crucial for aligning the input image correctly for later processing.
## Task 2: Image Pre-processing with Edge Detection:
I integrated and experimented with various image processing tools in the interactive_feature_extraction.py module. Through these experiments, I determined that the Sobel operator produced a cleaner, more robust image for subsequent processing than alternatives like the Laplacian operator. Although I initially tried using Hough transformations to detect lines, I found they did not contribute to an effective segmentation strategy.
## Task 3: Segmentation and Digital Twin Creation:
I implemented the watershed algorithm to segment the parking bays. This segmentation allowed me to extract and organize the centers of the bays, from which I derived the number of rows and columns, and ultimately built the digital twin of the car park.

# Reflection and Learning

Through this project, I gained valuable insights into:

## Interactive Image Processing
Building an interactive tool to test various pre-processing methods taught me which techniques yield the cleanest images for watershed segmentation. This experimentation was key to understanding the strengths and limitations of different edge detection operators.
## Pipeline Integration
I learned the importance of sequencing in a multi-stage image processing pipeline, especially when dealing with real-world images that contain noise and uneven features.
## Algorithm Selection
The process helped me appreciate why certain algorithms, such as the Sobel operator, perform better under noisy conditions compared to others like the Laplacian. The experiments also clarified why the Hough transformation, despite its capability to draw lines, was not suitable for segmenting parking bays as I originally envisioned. I also looked at the shortcomings of opensource models such as YOLO.

# Design Decisions
## Use of the Sobel Operator
I opted for the Sobel operator because it produces clearer edge detection results and is more resistant to noise compared to the Laplacian. This was crucial for obtaining a clean image input for the watershed segmentation.
Switching from Hough Transformation to Watershed:
Although the Hough transformation initially showed promise by drawing the lines in the image, it failed to segment the parking bays effectively or contribute to the digital twin’s generation. Therefore, I transitioned to the watershed algorithm, which better delineated the parking bays and facilitated subsequent digital twin construction.
Challenges, Mistakes, and Future Improvements

## Counting Rows and Columns
One of the main challenges was accurately counting the rows and columns of parking bays. Initially, I grouped points into rows based on whether the y-coordinate of a subsequent point was within half the average height of the first point in the current row. However, due to uneven distribution of the parking bays, this method proved unreliable. I then experimented with using a percentage threshold (approximately 3% of the image height) to group the centers more effectively.

# Future Improvements
If I were to approach this project again, I would:
## Directly Segment Before Counting
Segment the image first and overlay a grid on the segmented image to visually verify the grouping of parking bays. This could streamline the creation of a digital twin.
## Integrate with Downstream Tasks
Develop the pipeline further so that the digital twin seamlessly integrates with the cellular automata planning algorithm, potentially by pre-defining grid cells that match the physical layout of the parking bays.