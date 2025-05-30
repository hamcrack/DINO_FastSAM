import socket
import pickle
import struct
import cv2
import os
import supervision as sv
import torch
import numpy as np
from ultralytics import FastSAM

from typing import List

live = False
save_images = False

image_cnt = 0
folder_path = '4_double-packages'
output_folder = 'out'

def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def get_random_color():
    return [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)
print("supervision:", sv.__version__)
print("torch:", torch.__version__)

HOME = os.getcwd() + "/.."
print("HOME:", HOME)
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))

from groundingdino.util.inference import Model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)

fast_sam_model = FastSAM("FastSAM-s.pt")

if torch.cuda.is_available():
    fast_sam_model.to('cuda')
    print("FastSAM model loaded onto GPU.")
else:
    print("CUDA not available, using CPU.")

def segment(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    all_masks = []
    for box in xyxy:
        torch_box = torch.tensor([int(val) for val in box], device=fast_sam_model.device).unsqueeze(0)
        results = fast_sam_model(image, bboxes=torch_box)

        if results and len(results) > 0:  # Check if results list is not empty
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                masks_np = results[0].masks.data.cpu().numpy()
                all_masks.extend(masks_np)
            else:
                print("No masks found in results[0].")
        else:
            print("No detection results for box:", box)

    return np.array(all_masks)

TEXT_PROMPT = "package"
CLASSES = ['package']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# Get screen dimensions
screen_width, screen_height = 1920, 900 #  set to a default.  You can use  `get_monitors()` from the screeninfo library if needed

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created output folder: {output_folder}")

# Get a list of all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('172.25.0.1', 9999))
print("Connected to server")

data = b""
payload_size = struct.calcsize("Q")

if live and save_images:
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (640, 480))

while True:
    if live:
        while len(data) < payload_size:
            data += client_socket.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data)

    else:
        # Construct the full path to the image
        image_path = os.path.join(folder_path, image_files[image_cnt])
        output_path = os.path.join(output_folder, image_files[image_cnt]) #save with original name

            # Read the image using OpenCV
        frame = cv2.imread(image_path)
        frame = cv2.resize(frame, (640, 480))


    detections = grounding_dino_model.predict_with_classes(
        image=frame,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = []
    filtered_detections_list = []

    if detections.confidence is not None and detections.class_id is not None:
        for i in range(len(detections.confidence)):
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]
            xyxy = detections.xyxy[i]
            tracker_id = detections.tracker_id[i] if detections.tracker_id is not None else None

            low_confidence_threshold = 0.3  # Example threshold

            if class_id is not None:  # Check if class_id is not None
                class_id_int = int(class_id)
                if class_id_int < len(CLASSES):
                    if confidence >= low_confidence_threshold:
                        labels.append(f"{CLASSES[class_id_int]} {confidence:0.2f}")
                        filtered_detections_list.append(np.concatenate((xyxy, [confidence, class_id_int], [tracker_id] if tracker_id is not None else []), axis=0))
                    else:
                        labels.append(f"[LOW CONF] {CLASSES[class_id_int]} {confidence:0.2f}")
                        filtered_detections_list.append(np.concatenate((xyxy, [confidence, class_id_int], [tracker_id] if tracker_id is not None else []), axis=0))
            else:
                # Handle cases where class_id is None
                labels.append(f"[NO CLASS] {confidence:0.2f}")
                filtered_detections_list.append(np.concatenate((xyxy, [confidence, -1], [tracker_id] if tracker_id is not None else []), axis=0)) # Use -1 or another indicator

    # Convert the list of arrays to a NumPy array
    if filtered_detections_list:
        filtered_detections_array = np.array(filtered_detections_list)
        formatted_detections = sv.Detections(
            xyxy=filtered_detections_array[:, :4],
            confidence=filtered_detections_array[:, 4],
            class_id=filtered_detections_array[:, 5].astype(int),
            tracker_id=filtered_detections_array[:, 6] if filtered_detections_array.shape[1] > 6 else None
        )
    else:
        formatted_detections = sv.Detections(
            xyxy=np.empty((0, 4)),
            confidence=np.empty((0,)),
            class_id=np.empty((0,)),
            tracker_id=None
        )

    detections.mask = segment(
        image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        xyxy=formatted_detections.xyxy
    )

    print("Orig No. of detections: ", len(detections.mask))
    annotated_image_orig = mask_annotator.annotate(scene=frame.copy(), detections=detections)

    new_masks = []
    kernel = np.ones((20, 20), np.uint8)
    edge_lim = 150
    package_no = 1
    for mask in detections.mask:
        mask_uint8 = mask.astype(np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, kernel) 
        analysis = cv2.connectedComponentsWithStats(mask_uint8, 4, cv2.CV_32S) 
        (totalLabels, label_ids, values, centroid) = analysis 
        # print("     Total labels:", totalLabels)
        # print("     centroids : ", centroid)  
        for i in range(1, totalLabels): 
            center_near_edge = False

            if len(mask[0]) - edge_lim < centroid[i][0] or centroid[i][0] < edge_lim or len(mask) - edge_lim < centroid[i][1] or centroid[i][1] < edge_lim:
                center_near_edge = True
            area = values[i, cv2.CC_STAT_AREA]   

            if area < 10000 or center_near_edge: 
                componentMask = (label_ids != i).astype("uint8") * 255
                mask_uint8 = cv2.bitwise_and(mask_uint8, componentMask)
            else:
                colour = get_random_color()
                cv2.putText(frame, f"Package {package_no}", (int(centroid[i][0]),  int(centroid[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2) 
                package_no+=1

        if np.all(mask_uint8 == 0):
            print("Removed a mask")
        else:
            new_masks.extend([mask_uint8]) 
    print("New No. of detections: ", len(new_masks))
    detections.mask = np.array(new_masks)
    
    annotated_image_filt = mask_annotator.annotate(scene=frame.copy(), detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    if live and save_images:
        out.write(annotated_image_filt)

    cv2.imshow("Orig", annotated_image_orig)
    cv2.imshow("Filtered", annotated_image_filt)
    if cv2.waitKey(1) == ord('q'):
        if live:
            break
        else:
            image_cnt += 1
            if save_images:
                cv2.imwrite(output_path, annotated_image_filt)
            if image_cnt > 11:
                break

if live and save_images:
    out.release()
client_socket.close()
cv2.destroyAllWindows()
