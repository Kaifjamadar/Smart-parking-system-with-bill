import cv2
import pytesseract
import sqlite3
import os
from datetime import datetime
import numpy as np

# Set the path to the Tesseract OCR executable
# Update this path based on your Tesseract installation
# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Update the path below if your Tesseract is installed in a different location
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Initialize the Haar Cascade classifier for license plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('parking.db')
cursor = conn.cursor()

# Create a table to store vehicle records
cursor.execute('''
    CREATE TABLE IF NOT EXISTS vehicles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT NOT NULL,
        entry_time TEXT,
        exit_time TEXT,
        amount REAL
    )
''')
conn.commit()

def recognize_plate(image_path):
    """
    Detects and recognizes the license plate number from the given image.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image at {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        plate_img = img[y:y+h, x:x+w]
        # Preprocess the plate image for better OCR results
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, plate_thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(plate_thresh, config='--psm 8')
        # Clean the OCR result
        plate_number = ''.join(filter(str.isalnum, text))
        if plate_number:
            return plate_number
    return None

def process_entry(image_path):
    """
    Processes the entry of a vehicle.
    """
    plate_number = recognize_plate(image_path)
    if plate_number:
        entry_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('INSERT INTO vehicles (plate, entry_time) VALUES (?, ?)', (plate_number, entry_time))
        conn.commit()
        print(f"Vehicle {plate_number} entered at {entry_time}")
    else:
        print(f"License plate not detected in {image_path}")

def process_exit(image_path):
    """
    Processes the exit of a vehicle and generates the bill.
    """
    plate_number = recognize_plate(image_path)
    if plate_number:
        cursor.execute('SELECT id, entry_time FROM vehicles WHERE plate = ? AND exit_time IS NULL', (plate_number,))
        record = cursor.fetchone()
        if record:
            vehicle_id, entry_time_str = record
            entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
            exit_time = datetime.now()
            duration = exit_time - entry_time
            hours = duration.total_seconds() / 3600
            amount = round(hours * 100, 2)  # ₹100 per hour
            cursor.execute('UPDATE vehicles SET exit_time = ?, amount = ? WHERE id = ?', (exit_time.strftime('%Y-%m-%d %H:%M:%S'), amount, vehicle_id))
            conn.commit()
            print(f"Vehicle {plate_number} exited at {exit_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Duration: {hours:.2f} hours, Amount: ₹{amount:.2f}")
            generate_bill(plate_number, entry_time_str, exit_time.strftime('%Y-%m-%d %H:%M:%S'), amount)
        else:
            print(f"No entry record found for vehicle {plate_number}")
    else:
        print(f"License plate not detected in {image_path}")

def generate_bill(plate_number, entry_time, exit_time, amount):
    """
    Generates a bill for the vehicle.
    """
    if not os.path.exists('bills'):
        os.makedirs('bills')
    filename = f'bills/{plate_number}_{datetime.now().strftime("%Y%m%d%H%M%S")}.txt'
    with open(filename, 'w') as f:
        f.write('--- Parking Bill ---\n')
        f.write(f'Plate Number: {plate_number}\n')
        f.write(f'Entry Time: {entry_time}\n')
        f.write(f'Exit Time: {exit_time}\n')
        f.write(f'Total Amount: ₹{amount:.2f}\n')
    print(f"Bill generated: {filename}")

def process_images(directory, process_function):
    """
    Processes all images in the given directory using the specified function.
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            process_function(image_path)

if __name__ == '__main__':
    # Process entry images
    print("Processing entry images...")
    process_images('entry_images', process_entry)

    # Process exit images
    print("\nProcessing exit images...")
    process_images('exit_images', process_exit)

    # Close the database connection
    conn.close()
