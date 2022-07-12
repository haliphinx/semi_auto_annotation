import xml.dom.minidom

class GTdata:
    def __init__(self, xml_path: str) -> None:
        '''
        Initialize the ground truth data from the xml file.
        Input:
            xml_path: the path to the gt file
        '''
        self.data = self.xml_reader(xml_path)

    def get_bbox(self, obj_id: int = 0, frame_id: int = 0) -> tuple:
        '''
        Get the location of the bbox for the obj_id in the frame_id
        Input:
            frame_id: the id of the frame
            obj_id: the id of the object
        Output:
            bbox: the location of the bbox, tuple (xtl, ytl, xbr, ybr). Return empty tuple if the required bbox is not exist
        '''
        cur = self.data

        # Check if the annotation exists
        if "annotations" in cur:
            cur = cur["annotations"]
            
            # Check if the object exists
            if obj_id in cur:
                cur = cur[obj_id]
                
                # Check if the frame exists
                if frame_id in cur:
                    cur = cur[frame_id]
                    if (not cur["occluded"]) and cur["keyframe"]:
                        bbox = (cur["xtl"], cur["ytl"], cur["xbr"], cur["ybr"])
                        return bbox
        
        return ()

    def get_bboxes(self, obj_id: int = 0, start: int = 0, end: int = -1) -> list:
        '''
        Return the bbox trajectory in the whole video.
        Input:
            obj_id: the object id to be tracked
            start: the frame id to start
            end: the frame id to end, - represent to select from the end side. end is not included
        Output:
            bboxes: list[(xtl, ytl, xbr, ybr)], the bbox trajectory
        '''
        bboxes = []
        frame_num = self.data["frame_num"]

        assert start < frame_num and abs(end) <= frame_num, "Index out of the range."

        if end < 0:
            end = frame_num + end +1

        # print(start, end)
        for idx in range(start, end):
            bbox = self.get_bbox(obj_id, idx)
            bboxes.append(bbox)

        return bboxes


    def xml_reader(self, xml_path: str) -> dict:
        '''
        Read the data from an xml file and generate a dictionary
        Input:
            xml_path: the path to the xml file
        Output:
            data: the generated dictionary data

        data:
        --task_id: int, the task id in the CVAT
        --vid_name: str, the name of the video clip
        --frame_num: int, the total number of frames for the video clip
        --labels: list[str], the name of all labels
        --frame_size: (width, height), the size of the image frame
        --annotations: dict(label_id(str):gt_traj(dict)), the dict to store all annotation gt
        ----gt_traj: dict(frame_id(str): bbox(dict)), the dict to store the bbox for the object in all frames
        ------bbox: dict, store the attributes for each bbox
        --------outside: bool, if the object is outside of the frame
        --------occluded: bool, if the object is occluded
        --------keyframe: bool, if the current frame is keyframe
        --------xtl: float, the x value for the top left corner of the bbox
        --------ytl: float, the y value for the top left corner of the bbox
        --------xbr: float, the x value for the bottom right corner of the bbox
        --------ybr: float, the y value for the bottom right corner of the bbox
        --------z_order: int, not sure but keep it.
        '''
        # Read the xml file as a tree

        dom = xml.dom.minidom.parse(xml_path)

        # Get the root node of the tree ('annotations' in gt file)
        root = dom.documentElement
        
        # The dict to store all the infos
        data = {}

        # Get the root node for the task
        task_node = root.getElementsByTagName('meta')[0].getElementsByTagName('task')[0]

        # Get the task id
        task_id = int(task_node.getElementsByTagName('id')[0].childNodes[0].nodeValue)
        data["task_id"] = task_id

        # Get the video name
        vid_name = task_node.getElementsByTagName('name')[0].childNodes[0].nodeValue
        data["vid_name"] = vid_name

        # Get the size of video (number of frames)
        frame_num = int(task_node.getElementsByTagName('size')[0].childNodes[0].nodeValue)
        data["frame_num"] = frame_num

        # Get the labels name
        labels = []
        labels_node = task_node.getElementsByTagName('labels')[0].getElementsByTagName('label')

        for label in labels_node:
            label_name = label.getElementsByTagName('name')[0].childNodes[0].nodeValue
            labels.append(label_name)

        data["labels"] = labels

        # Get the frame size
        size_node = task_node.getElementsByTagName('original_size')[0]
        f_width = float(size_node.getElementsByTagName('width')[0].childNodes[0].nodeValue)
        f_height = float(size_node.getElementsByTagName('height')[0].childNodes[0].nodeValue)

        # Get the annotation data
        data["annotations"] = {}

        track_nodes = root.getElementsByTagName('track')
        for track in track_nodes:
            # Get the label of the tracker
            t_label = track.getAttribute("label")
            t_id = int(track.getAttribute("id"))

            # Store the trajectory data of the tracker
            t_data_dict = {}

            t_data = track.getElementsByTagName('box')
            for item in t_data:
                frame_id = int(item.getAttribute("frame"))
                outside=False if int(item.getAttribute("outside")) == 0 else True
                occluded=False if int(item.getAttribute("occluded")) == 0 else True 
                keyframe=False if int(item.getAttribute("keyframe")) == 0 else True 
                xtl=float(item.getAttribute("xtl")) 
                ytl=float(item.getAttribute("ytl")) 
                xbr=float(item.getAttribute("xbr"))  
                ybr=float(item.getAttribute("ybr"))
                xtl = max(0.0, min(xtl, f_width))
                ytl = max(0.0, min(ytl, f_height))
                xbr = max(0.0, min(xbr, f_width))
                ybr = max(0.0, min(ybr, f_height))


                z_order=int(item.getAttribute("z_order"))

                t_data_dict[frame_id] = {"outside":outside,
                                        "occluded":occluded,
                                        "keyframe":keyframe,
                                        "xtl":xtl,
                                        "ytl":ytl,
                                        "xbr":xbr,
                                        "ybr":ybr,
                                        "z_order":z_order}

            data["annotations"][t_id] = t_data_dict

        return data


if __name__ == "__main__":
    xml_path = "/home/xhu/Code/auto_annotation/data/annotations.xml"
    gt = GTdata(xml_path)
    print(gt.get_bbox(frame_id = 93))
