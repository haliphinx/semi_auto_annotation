import cv2
import sys
import time
from tqdm import tqdm
from siamrpn import TrackerSiamRPN

def opencvTracker(frame_list: list, init_bbox: list, tracker_type: int or str = 0, is_inverse: bool = False) -> list:
    '''
    The function to use opencv supported trackers for tracking
    Input:
        tracker_type: the type of trackers to be used. The input shold be the name or the index of the following list
                      ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'] 
        frame_list: a list of path to the sequence of frames to track
        is_inverse: whether tracking the frames inversely or not.
        init_bbox: the initial bbox in the first frame. [xtl, ytl, xbr, ybr]
    Output:
        bbox_list: the tracked bbox in each frame. [[xtl, ytl, xbr, ybr],...,[xtl, ytl, xbr, ybr]]
    '''

    # Select the tracker
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'SIAMRPN']
    if isinstance(tracker_type, int) and tracker_type >= 0 and tracker_type <= 8:
        tracker_type = tracker_types[tracker_type]
    elif isinstance(tracker_type, str) and tracker_type.upper() in tracker_types:
        tracker_type = tracker_type.upper()
    else:
        assert False, "The tracker type is not supported"

    # Setup the trackers
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    if tracker_type == 'SIAMRPN':
        net_path = 'pretrained/siamrpn/model.pth'
        tracker = TrackerSiamRPN(net_path=net_path)

    # Generate the loop list
    frame_length = len(frame_list)
    if is_inverse:
        loop = [x for x in range(frame_length - 1, -1, -1)]
    else:
        loop = [x for x in range(0, frame_length)]

    # Generate the initial bbox and round the number
    xtl, ytl, xbr, ybr = init_bbox

    width = int(xbr - xtl + 0.5)
    height = int(ybr - ytl + 0.5)
    xtl = int(xtl + 0.5)
    ytl = int(ytl + 0.5)
    bbox = (xtl, ytl, width, height)

    bbox_list = [(xtl, ytl, int(xbr), int(ybr))]
    # Initial the tracker with the first frame
    init_frame = cv2.imread(frame_list[loop[0]])
    f_height, f_width, _ = init_frame.shape
    ok = tracker.init(init_frame, bbox)

    fps_total = 0

    f_tracked = 1

    for idx in range(1, frame_length):
        # print(idx, loop[idx])
        cur_frame = cv2.imread(frame_list[loop[idx]])

        # Check if the image is loaded and the traker is initialized
        if (cur_frame is None) or (not ok):
            # print("track failure", ok, cur_frame is not None, frame_list[loop[idx]])
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(cur_frame)

        # Calculate Frames per second (FPS)
        fps_total += cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Add the tracking result to the list
        if ok:
            # print(idx, bbox)
            bbox_list.append((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            f_tracked += 1
            if bbox[2] > f_width or bbox[3] > f_height:
                ok = False
                # print(f"The tracking bbox is larger than the image frame.")


    fps_average = fps_total/(f_tracked)
    # print(f"The average tracking fps is {fps_average}. There are {f_tracked} frames are tracked including the inital frame.")
    
    bbox_list[0] = init_bbox

    return bbox_list

        
        


    

if __name__ == '__main__' :
    # Set up tracker.
    # Instead of MIL, you can also use

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT', 'SIAMRPN']
    tracker_type = tracker_types[8]

    if False:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        if tracker_type == 'SIAMRPN':
            net_path = 'pretrained/siamrpn/model.pth'
            tracker = TrackerSiamRPN(net_path=net_path)

    # Read video
    video = cv2.VideoCapture("/media/xhu/Study/data/coin_dataset/data/58_ReplaceBatteryOnKeyToCar/121/42GHJCP0knI_55_60.mp4")

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Define an initial bounding box
    bbox = (120, 23, 140, 150)

    # Uncomment the line below to select a different bounding box manually
    # bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        
        if not ok:
            break
        
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        # ok, bbox = tracker.update(frame)
        bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            # print(type(bbox))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        time.sleep(0.2)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break