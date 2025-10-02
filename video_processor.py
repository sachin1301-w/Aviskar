# video_processor.py

import cv2
import numpy as np
import os
import uuid # For generating unique filenames

# ===============================================================
# >> CAR PRICE DATA DICTIONARY <<
# ===============================================================
car_prices_data = {
    "HONDA": {
        "City": {"Bonnet": 15000, "Bumper": 10000, "Dickey": 8000, "Door": 20000, "Fender": 5000, "Light": 3000, "Windshield": 8000},
        "Amaze": {"Bonnet": 12000, "Bumper": 8000, "Dickey": 6000, "Door": 18000, "Fender": 4000, "Light": 2500, "Windshield": 7000},
        "WR-V": {"Bonnet": 16000, "Bumper": 11000, "Dickey": 9000, "Door": 22000, "Fender": 6000, "Light": 3500, "Windshield": 9000},
        "Jazz": {"Bonnet": 14000, "Bumper": 9000, "Dickey": 7000, "Door": 19000, "Fender": 4500, "Light": 2800, "Windshield": 8000},
        "HR-V": {"Bonnet": 18000, "Bumper": 12000, "Dickey": 10000, "Door": 24000, "Fender": 7000, "Light": 4000, "Windshield": 10000},
        "Pilot": {"Bonnet": 22000, "Bumper": 15000, "Dickey": 13000, "Door": 28000, "Fender": 8000, "Light": 5000, "Windshield": 12000},
        "CR-V": {"Bonnet": 20000, "Bumper": 13000, "Dickey": 11000, "Door": 26000, "Fender": 7500, "Light": 4500, "Windshield": 11000},
        "Accord": {"Bonnet": 22000, "Bumper": 15000, "Dickey": 13000, "Door": 28000, "Fender": 8000, "Light": 5000, "Windshield": 12000},
        "Civic": {"Bonnet": 18000, "Bumper": 12000, "Dickey": 10000, "Door": 24000, "Fender": 7000, "Light": 4000, "Windshield": 10000}
    },
    "MARUTI SUZUKI": {
        "Swift": {"Bonnet": 10000, "Bumper": 7000, "Dickey": 5000, "Door": 15000, "Fender": 3000, "Light": 2000, "Windshield": 6000},
        "Baleno": {"Bonnet": 12000, "Bumper": 8000, "Dickey": 6000, "Door": 18000, "Fender": 4000, "Light": 2500, "Windshield": 7000},
        "Vitara Brezza": {"Bonnet": 14000, "Bumper": 9000, "Dickey": 7000, "Door": 20000, "Fender": 4500, "Light": 2800, "Windshield": 8000},
        "Wagon R": {"Bonnet": 12000, "Bumper": 8000, "Dickey": 6000, "Door": 18000, "Fender": 4000, "Light": 2500, "Windshield": 7000},
        "Ertiga": {"Bonnet": 16000, "Bumper": 11000, "Dickey": 9000, "Door": 22000, "Fender": 6000, "Light": 3500, "Windshield": 9000},
        "Grand Vitara": {"Bonnet": 18000, "Bumper": 12000, "Dickey": 10000, "Door": 24000, "Fender": 7000, "Light": 4000, "Windshield": 10000}
    },
    "TOYOTA": {
        "Corolla": {"Bonnet": 20000, "Bumper": 13000, "Dickey": 11000, "Door": 26000, "Fender": 7500, "Light": 4500, "Windshield": 11000},
        "Camry": {"Bonnet": 22000, "Bumper": 15000, "Dickey": 13000, "Door": 28000, "Fender": 8000, "Light": 5000, "Windshield": 12000},
        "Fortuner": {"Bonnet": 25000, "Bumper": 17000, "Dickey": 15000, "Door": 30000, "Fender": 9000, "Light": 6000, "Windshield": 14000},
        "Innova": {"Bonnet": 23000, "Bumper": 16000, "Dickey": 14000, "Door": 29000, "Fender": 8500, "Light": 5500, "Windshield": 13000},
        "Yaris": {"Bonnet": 18000, "Bumper": 12000, "Dickey": 10000, "Door": 24000, "Fender": 7000, "Light": 4000, "Windshield": 10000}
    },
    "HYUNDAI": {
        "i20": {"Bonnet": 15000, "Bumper": 10000, "Dickey": 8000, "Door": 20000, "Fender": 5000, "Light": 3000, "Windshield": 8000},
        "Creta": {"Bonnet": 18000, "Bumper": 12000, "Dickey": 10000, "Door": 24000, "Fender": 7000, "Light": 4000, "Windshield": 10000},
        "Verna": {"Bonnet": 16000, "Bumper": 11000, "Dickey": 9000, "Door": 22000, "Fender": 6000, "Light": 3500, "Windshield": 9000},
        "Venue": {"Bonnet": 17000, "Bumper": 11500, "Dickey": 9500, "Door": 23000, "Fender": 6500, "Light": 3750, "Windshield": 9500},
        "Tucson": {"Bonnet": 20000, "Bumper": 13000, "Dickey": 11000, "Door": 26000, "Fender": 7500, "Light": 4500, "Windshield": 11000}
    },
    "NISSAN": {
        "Altima": {"Bonnet": 18000, "Bumper": 13000, "Dickey": 11000, "Door": 24000, "Fender": 7000, "Light": 4000, "Windshield": 10000},
        "Rogue": {"Bonnet": 20000, "Bumper": 14000, "Dickey": 12000, "Door": 26000, "Fender": 7500, "Light": 4500, "Windshield": 11000},
        "Sentra": {"Bonnet": 17000, "Bumper": 12000, "Dickey": 10000, "Door": 22000, "Fender": 6500, "Light": 3750, "Windshield": 9500},
        "Pathfinder": {"Bonnet": 18000, "Bumper": 13000, "Dickey": 11000, "Door": 24000, "Fender": 7000, "Light": 4000, "Windshield": 10000},
        "Titan": {"Bonnet": 20000, "Bumper": 14000, "Dickey": 12000, "Door": 26000, "Fender": 7500, "Light": 4500, "Windshield": 11000}
    },
    "SKODA": {
        "Octavia": {"Bonnet": 20000, "Bumper": 14000, "Dickey": 12000, "Door": 26000, "Fender": 7500, "Light": 4500, "Windshield": 11000},
        "Superb": {"Bonnet": 22000, "Bumper": 15000, "Dickey": 13000, "Door": 28000, "Fender": 8000, "Light": 5000, "Windshield": 12000},
        "Rapid": {"Bonnet": 18000, "Bumper": 12000, "Dickey": 10000, "Door": 24000, "Fender": 7000, "Light": 4000, "Windshield": 10000},
        "Kodiaq": {"Bonnet": 22000, "Bumper": 15000, "Dickey": 13000, "Door": 28000, "Fender": 8000, "Light": 5000, "Windshield": 12000},
        "Karoq": {"Bonnet": 19000, "Bumper": 13500, "Dickey": 11500, "Door": 25000, "Fender": 7250, "Light": 4250, "Windshield": 10500}
    }
}


def get_part_name_from_id(class_id):
    """Maps a YOLO class ID to a part name string."""
    class_names = ['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield']
    if 0 <= class_id < len(class_names):
        return class_names[int(class_id)]
    return None


def get_predictions_from_frame(frame, model, user_car_details):
    """
    Runs YOLO prediction on a single frame and calculates prices for detected parts.
    Returns:
        tuple: (list of detected predictions, image with bounding boxes drawn)
    """
    if not all(k in user_car_details for k in ['car_brand', 'model']):
        print("Error: User car details are incomplete.")
        return [], frame

    car_brand = user_car_details['car_brand']
    car_model = user_car_details['model']
    
    results = model(frame, verbose=False) # Run YOLO prediction
    
    # Draw bounding boxes and labels on a copy of the frame
    annotated_frame = results[0].plot()

    detected_objects = results[0].boxes
    predictions = []
    
    for box in detected_objects:
        class_id = box.cls.item()
        part_name = get_part_name_from_id(class_id)
        
        if part_name:
            try:
                price_per_part = car_prices_data[car_brand.upper()][car_model][part_name]
                predictions.append({
                    'part_name': part_name,
                    'price': price_per_part
                })
            except KeyError:
                print(f"Price not found for: {car_brand}, {car_model}, {part_name}")
                continue
                
    return predictions, annotated_frame


def process_video_for_repair_estimate(video_path, model, user_car_details):
    """
    Main function to process video, run predictions on frames, and aggregate results.
    Returns:
        dict: Contains total_price, itemized_report, and a list of paths to detected images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Define a folder to save detected frames within static
    STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    DETECTED_VIDEO_FRAMES_DIR = os.path.join(STATIC_DIR, 'detected_video_frames')
    os.makedirs(DETECTED_VIDEO_FRAMES_DIR, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Default fps if not available

    all_predictions = []
    detected_image_paths = [] # To store paths of frames with detections
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process one frame per second
        if frame_count % int(fps) == 0:
            print(f"Analyzing frame {frame_count}...")
            predictions_from_frame, annotated_frame = get_predictions_from_frame(frame, model, user_car_details)
            
            if predictions_from_frame:
                all_predictions.extend(predictions_from_frame)
                
                # Save the annotated frame
                unique_filename = f"detected_frame_{uuid.uuid4().hex}.jpg"
                save_path = os.path.join(DETECTED_VIDEO_FRAMES_DIR, unique_filename)
                cv2.imwrite(save_path, annotated_frame)
                
                # Store the relative path for web display
                detected_image_paths.append(f"detected_video_frames/{unique_filename}")
        frame_count += 1
    
    cap.release()

    # --- Aggregation Logic ---
    if not all_predictions:
        return {
            "total_price": 0,
            "itemized_report": [],
            "detected_images": []
        }

    final_report = {}
    for pred in all_predictions:
        part_name = pred['part_name']
        price = pred['price']
        
        # Aggregate by part name, storing the price for each unique part
        if part_name not in final_report:
            final_report[part_name] = {'price': price}

    itemized_report = []
    total_price = 0
    for part_name, data in final_report.items():
        itemized_report.append({
            'part_name': part_name,
            'price': data['price']
        })
        total_price += data['price']

    return {
        "total_price": round(total_price, 2),
        "itemized_report": itemized_report,
        "detected_images": list(set(detected_image_paths)) # Return only unique image paths
    }