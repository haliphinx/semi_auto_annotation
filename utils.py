import os
from cvat_gt_converter import GTdata
from tracker import *

def frame_sort(elem:str) -> int:
    '''
    The key function used to sort the frames by name
    Input:
        elem: the file name of the frame
    Output:
        num: the number used for sorting
    '''
    num = elem[-10:-4]
    return int(num)

def frame_to_vid(frame_path: str, save_path: str, img_format: str = "PNG", vid_name: str = "test", fps: str = 30) -> None:
    '''
    The function to convert sequence of frames into a video. The function is built on ffmpeg.
    Input:
        frame_path: the path to the folder with frames
        save_path: the path to save the generated video
        img_format: the format of the image frame
        vid_name: the name of the generated video
        fps: the fps for the video
    '''
    vid_path = os.path.join(save_path, vid_name+".mp4")
    cmd = f"/usr/bin/ffmpeg -r {fps} -pattern_type glob -i '{frame_path}/*.{img_format}' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r {fps} -crf 25 -c:v libx264 -pix_fmt yuv420p -movflags +faststart {vid_path}"
    os.system(cmd)

def frame_list_gen(frame_path: str, img_format: str = "PNG", start: int = 0, end: int = -1, check_num:int = -1, is_full_path:bool = True) -> list:
    '''
    Generate a list for the frame sequence path
    Input:
        frame_path: the folder contains all frames
        img_format: the format of the image frame
        start: the frame id to start
        end: the frame id to end, - represents to select from the end.
        check_num: if set to a positive number then check the number of frames under the folder equals to the number or not
        is_full_path: whether to store the full path or the relevant path to the frame_path
    Output:
        frame_list: a list of path for the image frames
    '''
    frames = os.listdir(frame_path)
    format_len = len(img_format)
    frame_list = [x for x in frames if x[-format_len:]==img_format]
    frame_list.sort(key = frame_sort)
    # print(frame_list)
    if check_num>0:
        assert check_num == len(frame_list), "The number of frames is not equal to the asked number"

    # Convert to the full path is needed
    if is_full_path:
        frame_list = [os.path.join(frame_path, x) for x in frame_list]

    if end < 0:
        end = len(frame_list) + end +1
    if end >= len(frame_list)-1:
        return frame_list[start:]
    else:
        return frame_list[start:end+1]

def iou_cal(gt_bbox: tuple, est_bbox: tuple) -> float:
    '''
    Calculate the iou between the gt and the tracking estimations.
    Input:
        gt_bbox: (xtl, ytl, xbr, ybr), the bbox for the gt
        est_bbox: (xtl, ytl, xbr, ybr), the bbox for the estimation
    Output:
        miou: the calculated iou over the bbox
    '''
    # Verify if the bboxs are legel
    # print(gt_bbox, est_bbox)
    assert gt_bbox[2] > 0 and gt_bbox[3] > 0 and est_bbox[2] > 0 and est_bbox[3] > 0, "The bbox is illegal."
    
    # Calculate the intersection
    i_xtl = max(gt_bbox[0], est_bbox[0])
    i_ytl = max(gt_bbox[1], est_bbox[1])
    i_xbr = min(gt_bbox[2], est_bbox[2])
    i_ybr = min(gt_bbox[3], est_bbox[3])

    # If there is no intersection
    if i_xtl >= i_xbr or i_ytl >= i_ybr:
        return 0.0

    i_area = (i_xbr - i_xtl) * (i_ybr - i_ytl)

    # Calculate the union. Union = gt_bbox + est_bbox - intersection
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    est_area = (est_bbox[2] - est_bbox[0]) * (est_bbox[3] - est_bbox[1])

    u_area = float(gt_area + est_area - i_area)
    iou = i_area / u_area

    # Check if the iou is legal
    assert iou >= 0.0 and iou <= 1.0, "IOU is out of range."

    return iou

def volume_iou(gt_bboxes:list, est_bboxes:list, is_inverse:bool = False) -> (float, int):
    '''
    Calculate the volume iou over the frame sequences
    Input:
        gt_bboxes: the list of gt bboxes. [(xtl, ytl, xbr, ybr)]
        est_bboxes: the list of estimation bboxes. [(xtl, ytl, xbr, ybr)]
        is_inverse: whether the estimation is in the inverse order
    Output:
        viou: volume iou is the mean iou over the whole video
        f_tracked: the number of frame that is tracked
    '''

    # Check if the estimated bboxes is less than the gt bboxes
    assert len(gt_bboxes) >= len(est_bboxes) and len(est_bboxes) > 0, "The number of estimated bboxes is over than the gt."

    f_tracked = 0
    iou_sum = 0

    for idx in range(len(est_bboxes)):

        gt_bbox = gt_bboxes[-(idx+1)] if is_inverse else gt_bboxes[idx]
        est_bbox = est_bboxes[idx]

        # There is bbox in the current frame in gt
        if len(gt_bbox) == 4:
            f_tracked += 1
            iou = iou_cal(gt_bbox, est_bbox)
            iou_sum += iou

    viou = iou_sum / float(f_tracked)
    return viou, f_tracked

def viou_gt(gt_bboxes:list, est_bboxes:dict) -> float:
    '''
    Calculate the viou between the estimation and the gt
    Input:
        gt_bboxes: the list of gt bbox
        est_bboxes: the dict of the estimated bbox {id:bbox}
    '''
    iou_sum = 0
    f_tracked = 0
    for idx in range(0, len(gt_bboxes)):
        if len(gt_bboxes[idx]) != 4:
            continue
        if idx not in est_bboxes:
            continue
        f_tracked += 1
        gt_bbox = gt_bboxes[idx]
        est_bbox = est_bboxes[idx]
        iou_sum += iou_cal(gt_bbox, est_bbox)
    if f_tracked == 0:
        return 0
    else:
        return iou_sum / f_tracked

def tracker_eval(gt:object, frame_list:list, start:int, end:int, track_type:int, obj_id:int, gt_comp:bool = True) -> tuple:
    '''
    Evaluate the tracking method on a given frame sequences with the volume iou
    Input:
        gt: the gt object generated from the xml file
        frame_list: the frame sequences for tracking
        start & end: the frame sequence id for tracking
        track_type: the tracker used for tracking
        obj_id: the id of the object to be tracked
        gt_comp: if compare the tracker result with the annotated gt in the interval
    Output:
        viou: the volume iou between forward tracking and backward tracking
        ftrack_bbox: the bbox trajectory from forward tracking
        btrack_bbox: the bbox trajectory from backward tracking
        gt_iou: the iou calculated with gt ksyframes
    '''

    init_bbox_start = gt.get_bbox(obj_id = obj_id, frame_id = start)

    if end == len(frame_list) - 1:
        frame_list = frame_list[start:]
    else:
        frame_list = frame_list[start:end + 1]

    init_bbox_end = gt.get_bbox(obj_id = obj_id, frame_id = end)

    # print(init_bbox_start, init_bbox_end)

    assert len(init_bbox_start) == 4 and len(init_bbox_end) == 4

    # print(f"Tracking with the {track_type} method.")
    # print(start, init_bbox_start, end, init_bbox_end)

    ftrack_bbox = opencvTracker(frame_list, init_bbox_start, track_type)

    # The backward tracking, the result is in the inverse order
    btrack_bbox = opencvTracker(frame_list, init_bbox_end, track_type, is_inverse = True)

    # print(len(btrack_bbox), len(ftrack_bbox))

    if len(btrack_bbox) != len(frame_list) or len(ftrack_bbox) != len(frame_list):
        print(f"method {track_type} can't tracking successfully")
        return 0.0, ftrack_bbox, btrack_bbox, 0.0

    viou = volume_iou(ftrack_bbox, btrack_bbox)


    # if using the pre-labeled frames for evaluation
    gt_iou_list = []
    gt_iou = 1.0

    if gt_comp and (end-start) > 1:
        # loop through all frames to see if there is any pre-labeled keyframes in the interval
        for idx in range(start+1, end):
            
            gt_bbox = gt.get_bbox(obj_id = obj_id, frame_id = idx)
            # if a keyframe is detected
            if len(gt_bbox) == 4:
                # The forward tracking bbox
                f_bbox = ftrack_bbox[idx - start]

                # The backward tracking bbox
                b_bbox = btrack_bbox[-(idx-start+1)]

                # Add the iou into the list
                gt_iou_list.append(iou_cal(gt_bbox, f_bbox))
                gt_iou_list.append(iou_cal(gt_bbox, b_bbox))
        if len(gt_iou_list) > 0:
            gt_iou = sum(gt_iou_list) / len(gt_iou_list)

    print(f"The {track_type} method tracks successfully, the viou info is {viou[0]}, the gt_iou info is {gt_iou}, total gt num is {len(gt_iou_list)//2}")

    return viou[0], ftrack_bbox, btrack_bbox, gt_iou

def draw_result(frame_list:list, bboxes:dict, save_path:str, add_gt:bool = False, gt:list = None, is_vid:bool = False, keyframe:list = None) -> None:
    '''
    Draw the bboxes into the frame and generate a video
    Input:
        frame_list: the list of path to all the original frame image
        bboxes: the dictionary to store all bboxes
        save_path: the path to save the result
        add_gt: add the ground truth into the frame or not. If yes, the following para need to be specified
        gt: the gt bbox list
        is_vid: whether convert the result into a video or not
        keyframe: the list of frame which belongs to the keyframe
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok = True)

    for idx in range(0,len(frame_list)):
        frame = cv2.imread(frame_list[idx])
        img_name = frame_list[idx].split("/")[-1]
        img_save_path = os.path.join(save_path, img_name)

        if idx in bboxes:
            bbox = bboxes[idx]
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        if add_gt:
            bbox = gt[idx]
            if len(bbox) == 4:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
            
        if keyframe is not None and idx in keyframe:
            cv2.putText(frame, "Manually labeled", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        cv2.imwrite(img_save_path, frame)

    if is_vid:
        frame_to_vid(save_path, save_path)

def for_back_interpolation(ftrack: list, btrack: list) -> dict:
    '''
    Interpolate the forward tracking result with the backward tracking result
    Input:
        ftrack: the forward tracking result
        btrack: the backward tracking result
    Output:
        itrack: the interpolated tracking result
    '''
    frame_num = len(ftrack)
    btrack.reverse() 
    # print(frame_id)
    itrack = {}
    for idx in range(0, frame_num):
        f_weight = (frame_num-idx) / frame_num
        b_weight = idx / frame_num
        f_bbox = ftrack[idx]
        b_bbox = btrack[idx]
        itrack[idx] = (f_bbox[0]*f_weight + b_bbox[0]*b_weight,
                        f_bbox[1]*f_weight + b_bbox[1]*b_weight,
                        (f_bbox[2]*f_weight + b_bbox[2]*b_weight),
                        (f_bbox[3]*f_weight + b_bbox[3]*b_weight))
    return itrack
    

def add_keyframe(gt:object, frame_list:list, obj_id:int, frame_ids:list) -> None:
    '''
    The function to add keyframes and annotations
    Input:
        gt: the ground truth data 
        frame_list: the list of frame path
        obj_id: the object id to be labeled
        frame_ids: the list of frame_id to be labeled
    '''
    obj_name = gt.data["labels"][obj_id]
    for frame_id in frame_ids:
        cur_frame = cv2.imread(frame_list[frame_id])
        resize_ratio = 2
        resize_dim = (cur_frame.shape[1] * resize_ratio, cur_frame.shape[0] * resize_ratio)
        resized_frame = cv2.resize(cur_frame, resize_dim, interpolation = cv2.INTER_AREA)
        bbox = cv2.selectROI(f"frame: {frame_id} -- object: {obj_name}", resized_frame, showCrosshair = False)
        new_bbox = {}
        new_bbox[frame_id] = (bbox[0]/resize_ratio, bbox[1]/resize_ratio, (bbox[0] + bbox[2])/resize_ratio, (bbox[1] + bbox[3])/resize_ratio)
        gt.update_xml(obj_id, new_bbox, is_save = True)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    xml_path = "/home/xhu/Code/auto_annotation/data/annotations.xml"
    # print(GTdata(xml_path))

    save_root_path = "/home/xhu/Code/auto_annotation/src"

    extracted_fps = 30

    img_path = "/home/xhu/Code/auto_annotation/data/images"
    vid_path = "test.mp4"
    # frame_to_vid(img_path, save_root_path)
    frame_list_gen(img_path)