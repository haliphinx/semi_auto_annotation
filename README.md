# semi_auto_annotation
A script to annotate the video based object detection(bounding box).
1. Label the first and the last frames.
2. Select the tracker (support ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'SIAMRPN'])
3. The script would track the object forward and backward to compare the miou of two tracked trajctory.
4. If the miou is lower than the pre-defined threshold, the middle frame need to be manually labeled, and the original frame set is divided into two sets (first to middle, middle to end).
5. Redo the step 3&4 until there is no frame need to be manually labeled.
