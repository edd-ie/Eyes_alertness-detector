# Eyes_alertness-detector

Project uses the feature mappers called – Haar cascades
Mapper - algorithm for exploration, analysis and visualization of data
Language: Python
Library: OpenCV

Haar cascades are stored in OpenCV as:
-	haarcascade_frontalface_default.xml
-	haarcascade_eye_tree_eyeglasses.xml files

Project develops an understanding of the system:
-	Drowsiness detection
-	Eye blink locks
-	Eye detection
-	Face detection

*Haar Cascades*
An effective object detection method proposed in the paper, “Rapid Object Detection using a Boosted Cascade of Simple Features_2001”
A machine learning approach where a Cascade function is trained from lots of positive and negative images
	Cascade function – function that tests conditions until True
	Positive images - are the samples which contain the target object
	Negative images - are the samples which don’t contain the target object





*Operation*
Extrude features from the input image with haar features
 
Each feature is a single value got by:
	(Sum of pixels in white rectange) – (Sum of pixels in black rectangle)
 


False features:
-	From features calculated most are irrelevant.
-	To remove false feature AdaBoost is used
-	AdaBoost - Adaptive Boosting is a Machine learning algorithm which was used for this sole task.




Algorithm:
-	The frame is captured and converted to grayscale.
-	Bilateral Filtering is applied to remove impurities.
-	Face is detected with the haar cascade.
-	The ROI (Region Of Image) of Face is fed to eye detection part of algorithm.
-	Eyes are detected and resulting list is passed to if-else construct.
-	If the length of list is more than two, means that the eyes are there.
-	Else the program is marked to be eye blinked and restarted.
