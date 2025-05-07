import xml.etree.ElementTree as ET
import os
from typing import Dict, List, Tuple
from collections import defaultdict

def parse_xml(file_path: str) -> Tuple[List[Dict], int]:
    """
    Parse an XML file and extract bounding box information and the width of the image.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    boxes = []
    width = int(root.find('size').find('width').text)

    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    return boxes, width

def merge_boxes(boxes_list: List[Tuple[List[Dict], int]]) -> List[Dict]:
    """
    Merge bounding boxes from multiple lists based on the width of the previous image.
    """
    merged_boxes = []
    offset = 0

    for boxes, width in boxes_list:
        for box in boxes:
            box['xmin'] += offset
            box['xmax'] += offset
            merged_boxes.append(box)
        offset += width

    return merged_boxes

def process_xml_folder(folder_path: str) -> Dict[str, List[Dict]]:
    """
    Process all XML files in a folder and merge boxes based on timestamp.
    """
    files = os.listdir(folder_path)
    timestamp_boxes = defaultdict(list)

    for file in sorted(files):
        if file.endswith('.xml'):
            timestamp = '-'.join(file.split('-')[:6])
            boxes, width = parse_xml(os.path.join(folder_path, file))
            timestamp_boxes[timestamp].append((boxes, width))

    for timestamp in timestamp_boxes:
        timestamp_boxes[timestamp] = merge_boxes(timestamp_boxes[timestamp])

    return dict(timestamp_boxes)

