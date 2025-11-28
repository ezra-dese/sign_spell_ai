import cv2
import csv
import os
import time
from hand_detector import HandDetector
from feature_extractor import FeatureExtractor

def collect_data():
    detector = HandDetector(detection_con=0.8)
    extractor = FeatureExtractor()
    
    # Define signs to record
    # Skipping J and Z for now as they are dynamic
    signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    data_dir = "../data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    dataset_path = os.path.join(data_dir, "dataset.csv")
    
    # Check if header needs to be written
    write_header = not os.path.exists(dataset_path)
    
    cap = cv2.VideoCapture(0)
    
    print("Sign Language Data Collector")
    print("----------------------------")
    print(f"Signs to collect: {signs}")
    print("Press 'r' to start recording for the current sign.")
    print("Press 'n' to skip to next sign.")
    print("Press 'q' to quit.")
    
    current_sign_idx = 0
    samples_per_sign = 50
    
    while current_sign_idx < len(signs):
        target_sign = signs[current_sign_idx]
        print(f"\n>>> Target Sign: {target_sign} <<<")
        
        recording = False
        sample_count = 0
        
        while True:
            success, img = cap.read()
            if not success:
                break
                
            img = detector.find_hands(img)
            lm_list = detector.find_position(img, draw=False)
            
            # UI Overlay
            cv2.putText(img, f"Target: {target_sign}", (10, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
            
            if recording:
                cv2.putText(img, f"Recording: {sample_count}/{samples_per_sign}", (10, 100), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                
                if len(lm_list) != 0:
                    features = extractor.extract_features(lm_list)
                    if features:
                        # Append label and features to CSV
                        with open(dataset_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if write_header:
                                # Create header: label, f0, f1, ...
                                header = ['label'] + [f'f{i}' for i in range(len(features))]
                                writer.writerow(header)
                                write_header = False
                            
                            writer.writerow([target_sign] + features)
                            
                        sample_count += 1
                        if sample_count >= samples_per_sign:
                            recording = False
                            print(f"Finished recording {target_sign}")
                            current_sign_idx += 1
                            break # Break inner loop to move to next sign
            else:
                cv2.putText(img, "Press 'r' to Record", (10, 100), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            cv2.imshow("Data Collector", img)
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                recording = True
                sample_count = 0
            elif key == ord('n'):
                current_sign_idx += 1
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete!")

if __name__ == "__main__":
    collect_data()
