%% For image detection

image = imread('assets/raw/carParkImg.png');

detector = yolov4ObjectDetector("csp-darknet53-coco");

[bboxes, scores] = detect(detector, image);

figure;
imshow(image);
hold on;

for i = 1:size(bboxes,1)
    rectangle('Position', bboxes(i,:), 'EdgeColor', 'r', 'LineWidth', 2);
end

title('Car Detection Result');
hold off;
%% For video detection

videoFile = 'assets/raw/carPark.mp4'; 
vidReader = VideoReader(videoFile);

outputVideo = VideoWriter('output_video.avi', 'Uncompressed AVI');  
open(outputVideo);

detector = yolov4ObjectDetector("csp-darknet53-coco");

while hasFrame(vidReader)

    frame = readFrame(vidReader);
    
    [bboxes, scores] = detect(detector, frame);
    
    frameWithBoxes = frame;
    for i = 1:size(bboxes,1)
        frameWithBoxes = insertShape(frameWithBoxes, 'Rectangle', bboxes(i,:), 'Color', 'red', 'LineWidth', 2);
    end
    
    imshow(frameWithBoxes);
    writeVideo(outputVideo, frameWithBoxes);  
    
end

close(outputVideo);


