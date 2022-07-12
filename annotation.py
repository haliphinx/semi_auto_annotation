from utils import *
import json

if __name__ == "__main__":

    json_path = "/home/xhu/Code/auto_annotation/src/config.json"
    with open(json_path, 'r') as f:
        cfg = json.load(f)

    gt = GTdata(cfg["xml_path"])

    interval = cfg["intervals"]

    assert len(interval) > 0, "No valid interval."

    frame_list = frame_list_gen(cfg["img_path"])

    # print(frame_list)

    final_interval = []

    false_interval = []

    viou_thresh = cfg["viou_thresh"]

    # f_bbox_traj = {}
    # b_bbox_traj = {}

    # Total number of tracker to be tried
    track_num = len(cfg["track_type"])
    i_bbox_traj = {}

    kf_require = set()

    while(len(interval)>0):
        cur_interval = interval[0]
        interval = interval[1:]

        f_bbox_traj = {}
        b_bbox_traj = {} 

        bbox_0 = gt.get_bbox(cfg["obj_id"], cur_interval[0])
        bbox_1 = gt.get_bbox(cfg["obj_id"], cur_interval[1])
        if len(bbox_0) == 0:
            kf_require.add(cur_interval[0])
        
        if len(bbox_1) == 0:
            kf_require.add(cur_interval[1])

        if len(bbox_0) == 0 or len(bbox_1) == 0:
            false_interval.append(cur_interval)
            continue

        print(cur_interval)
        # if the current interval contains less than 3 frames, then stop tracking, 
        # Since the middle frame can be labeled by linear interpolation
        if cur_interval[1] - cur_interval[0] <2 :
            false_interval.append(cur_interval)
            # Linear Interpolation
            length = cur_interval[1] - cur_interval[0]
            

            for idx in range(cur_interval[0], cur_interval[1] + 1):
                weight = (idx - cur_interval[0])/length
                bbox_interpolate = (bbox_0[0]*(1 - weight) + bbox_1[0]*weight,
                        bbox_0[1]*(1 - weight) + bbox_1[1]*weight,
                        (bbox_0[2]*(1 - weight) + bbox_1[2]*weight),
                        (bbox_0[3]*(1 - weight) + bbox_1[3]*weight))

                i_bbox_traj[idx] = bbox_interpolate
            print("The result is manually labeled.")
            continue


        viou_max = 0
        tracker_tried = 0
        
        # Track the interval by all selected trackers
        for tracker in cfg["track_type"]:
            viou, ftrack, btrack = tracker_eval(gt, frame_list, cur_interval[0], cur_interval[1], tracker, cfg["obj_id"])
            tracker_tried += 1
            if viou >= viou_thresh and viou > viou_max:
                # add the current interval into final interval list if hasn't done before
                if viou_max == 0:
                    final_interval.append(cur_interval)

                length = cur_interval[1] - cur_interval[0] + 1

                # Calculate the interpolated trajectory between forward and backward tracking
                i_bboxes = for_back_interpolation(ftrack, btrack)

                for idx in range(0, length):
                    i_bbox_traj[idx + cur_interval[0]] = i_bboxes[idx]

                viou_max = viou
            
            else:
                # If all methods are tried and no one tracked successfully
                if (tracker_tried == track_num) and (viou_max == 0):
                    mid = (cur_interval[1]+cur_interval[0])//2
                    interval.append([cur_interval[0], mid])
                    interval.append([mid, cur_interval[1]])
    

    # If there is no required kf
    if len(kf_require) == 0:
        # Generate a list to record all manually labeled frame id
        manual_label = len(final_interval) + len(false_interval)+1
        print(f"There are {manual_label} frames need to be manually labeled.")
        all_interval = final_interval + false_interval

        keyframe = set()

        for interval in all_interval:
            keyframe.add(interval[0])
            keyframe.add(interval[1])

        gt_bbox_traj = gt.get_bboxes(cfg["obj_id"])
        
        # Draw the bbox to the frame
        draw_result(frame_list, i_bbox_traj, cfg["save_path"], False, None, True, keyframe)

    else:
        if len(cfg["intervals"]) == 1:
            print("initial keyframe selection.",init_keyframe_select(cfg["intervals"][0][0], cfg["intervals"][0][1]))
        obj_id = cfg["obj_id"]
        print(f"The gt for {obj_id} in frame {kf_require} need to be labeled.")
        cfg["intervals"] = false_interval
        with open(json_path, 'w') as f:
            json.dump(cfg, f)

# [[0,143], [263, 359]]
# [[0,124]]
# [[0,329]]