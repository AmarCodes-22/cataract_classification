import os
import csv
import json
import argparse
import requests
from PIL import Image
from time import sleep
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def create_directory(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)

def download_image(image_url, save_path, retries=3):
    for attempt in range(retries):
        try:
            with requests.get(image_url, stream=True) as response:
                response.raise_for_status()  
                content_length = response.headers.get('Content-Length')
                if not os.path.exists(save_path):
                    with open(save_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  
                                f.write(chunk)
                                downloaded += len(chunk)
                    
                
                if content_length and int(content_length) == downloaded:
                    print(f"Downloaded image {save_path} successfully.")
                    return save_path
                else:
                    print(f"Download failed, retry {attempt + 1}/{retries}.")
                    os.remove(save_path)  
        except requests.RequestException as e:
            print(f"Error downloading {image_url}: {str(e)}")

    print(f"Failed to download {image_url} after {retries} attempts.")
    return None

def convert_payload_to_form_data(payload):
    form_data = {}
    for key, value in payload.items():
        if isinstance(value, dict) or isinstance(value, list):
            value = json.dumps(value)
        form_data[key] = (None, '' if value is None else str(value))
    return form_data

def process_image(image_url, payload, api_url):
    payload['image_url'] = image_url.strip()  
    form_data = convert_payload_to_form_data(payload)
    print(f"Sending payload: {form_data} to {api_url}")
    response = requests.post(api_url, files=form_data)
    print("received output from API")
    if response.status_code == 200:
        return response.json()
    else:
        print(f'Error processing {image_url}: {response.status_code} - {response.text}')
        return None

def image_already_processed(image_name, *directories):
    paths = [os.path.join(dir, image_name + '.jpg') for dir in directories]
    return all(os.path.exists(path) for path in paths)

def save_output_image(image_url, output_dir, image_name):
    image_path = os.path.join(output_dir, image_name + '.jpg')
    response = requests.get(image_url)
    if response.status_code == 200:
        if not os.path.exists(image_path):
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f'Saved processed image to {image_path}')
    else:
        print(f'Failed to download {image_url}')

def compare_images(input_path, manual_path):
    try:
        with Image.open(input_path) as img1, Image.open(manual_path) as img2:
            return img1.size == img2.size and img1.mode == img2.mode
    except IOError as e:
        print(f"Error opening image files: {str(e)}")
        return False

def convert_null_to_none(obj):
    if isinstance(obj, dict):
        return {k: convert_null_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_null_to_none(v) for v in obj]
    elif obj is None:
        return None
    return obj

def correct_force_process_format(payload):
    if 'force_process' in payload and isinstance(payload['force_process'], dict):
        convert_null_to_none(payload['force_process']) 
    return payload

def clean_json_string(json_str):
    try:
        json_str = json_str.replace("'", '"')  
        json_str = json_str.replace('None', 'null')  
        json_str = json_str.replace('True', 'true').replace('False', 'false')  

        json_obj = json.loads(json_str)
        return json.dumps(json_obj)
    except json.JSONDecodeError as e:
        print(f"Error cleaning JSON string: {e}")
        return None

def process_entry(row, api_url, payload_template, directories):
    try:
        image_category = row.get('image_category')
        image_url = row.get('input_image_hres_url')
        image_name = row.get('image_id')

        if not image_category or not image_url or not image_name:
            print(f"Missing required data in row: {row}")
            return

        payload = json.loads(json.dumps(payload_template))
        payload = convert_null_to_none(payload)

        if image_category == 'fail':
            print(f"Processing fail case for image {image_name}")
            bg_id = row.get('BG_ID')
            save_params = row.get('save_params')
            manual_url = row.get('manual_edited_hres_url')

            if not bg_id or not save_params or not manual_url:
                print(f"Missing BG_ID, save_params, or manual URL for fail case: {row}")
                return

            clean_save_params = clean_json_string(save_params)
            if clean_save_params is None:
                print(f"Skipping row due to invalid save_params: {row}")
                return

            try:
                payload['bg_id'] = bg_id
                payload['save_params'] = json.loads(clean_save_params)
                payload['force_process'] = {"auto-segmentation-interior_removebg": "backtrace"}
                payload = correct_force_process_format(payload)
            except json.JSONDecodeError as e:
                print(f"Error decoding save_params JSON: {e}")
                return

            print(f"Payload for fail case: {payload}")
            image_path = os.path.join(directories['fail_images_dir'], image_name + '.jpg')
            manual_path = os.path.join(directories['manual_dir'], image_name + '.jpg')
            ai_dash_path = os.path.join(directories['ai_dash_dir'], image_name + '.jpg')

            if not image_already_processed(image_name, directories['fail_images_dir'], directories['ai_dash_dir'], directories['manual_dir']):
                if not os.path.exists(image_path):
                    download_result = download_image(image_url, image_path)
                    if download_result is None:
                        print(f"Failed to download fail image {image_url}")
                        return

                if not os.path.exists(manual_path):
                    download_result = download_image(manual_url, manual_path)
                    if download_result is None:
                        print(f"Failed to download manual image {manual_url}")
                        return

                if os.path.exists(image_path) and os.path.exists(manual_path):
                    if compare_images(image_path, manual_path):
                        os.system(f"cp {image_path} {ai_dash_path}")
                        print(f"Copied {image_name} to AI dashboard directory as they are identical.")
                    else:
                        processed_data = process_image(image_url, payload, api_url)
                        if processed_data and 'url' in processed_data:
                            save_output_image(processed_data['url'], directories['ai_dash_dir'], image_name)
                        else:
                            print(f"Failed to process image {image_name} in fail case")
    

        elif image_category == 'pass':
            print(f"Processing pass case for image {image_name}")
            backtrace_exists = os.path.exists(os.path.join(directories['backtrace_dir'], image_name + '.jpg'))
            without_backtrace_exists = os.path.exists(os.path.join(directories['without_backtrace_dir'], image_name + '.jpg'))

            image_path = os.path.join(directories['pass_images_dir'], image_name + '.jpg')
            if not os.path.exists(image_path):
                download_image(image_url, image_path)

            if not backtrace_exists or not without_backtrace_exists:
                if not backtrace_exists:
                    backtrace_payload = payload_template.copy()
                    backtrace_payload.update({
                        'force_process': {"auto-segmentation-interior_removebg": "backtrace"}
                    })
                    backtrace_payload = correct_force_process_format(backtrace_payload)
                    backtrace_data = process_image(image_url, backtrace_payload, api_url)
                    if backtrace_data and 'url' in backtrace_data:
                        save_output_image(backtrace_data['url'], directories['backtrace_dir'], image_name)

                if not without_backtrace_exists:
                    without_backtrace_payload = payload_template.copy()
                    without_backtrace_payload.update({
                        'force_process': None
                    })
                    without_backtrace_data = process_image(image_url, without_backtrace_payload, api_url)
                    if without_backtrace_data and 'url' in without_backtrace_data:
                        save_output_image(without_backtrace_data['url'], directories['without_backtrace_dir'], image_name)
    except Exception as e:
        print("Error:",e)

def main(api_url, csv_path, payload_file, directories):
    create_directory(directories['backtrace_dir'])
    create_directory(directories['without_backtrace_dir'])
    create_directory(directories['ai_dash_dir'])
    create_directory(directories['manual_dir'])
    create_directory(directories['pass_images_dir'])
    create_directory(directories['fail_images_dir'])

    with open(payload_file) as f:
        payload_template = json.load(f)
        payload_template = convert_null_to_none(payload_template)

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_entry, row, api_url, payload_template, directories) for row in reader]
            for future in tqdm(futures, total=len(futures)):
                future.result()

    with open("bulk_complete.flag", "w") as f:
        f.write("Processing completed successfully.")
        print("Flag file created to signal completion.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('api_url', type=str, help='API URL')
    parser.add_argument('csv_path', type=str, help='CSV file path')
    parser.add_argument('payload_file', type=str, help='Payload JSON file')
    parser.add_argument('--backtrace_dir', type=str, default='backtrace_output')
    parser.add_argument('--without_backtrace_dir', type=str, default='without_backtrace_output')
    parser.add_argument('--ai_dash_dir', type=str, default='ai_dashboard_output')
    parser.add_argument('--manual_dir', type=str, default='manual_images')
    parser.add_argument('--pass_images_dir', type=str, default='pass_images')
    parser.add_argument('--fail_images_dir', type=str, default='fail_images')
    args = parser.parse_args()

    directories = {
        'backtrace_dir': args.backtrace_dir,
        'without_backtrace_dir': args.without_backtrace_dir,
        'ai_dash_dir': args.ai_dash_dir,
        'manual_dir': args.manual_dir,
        'pass_images_dir': args.pass_images_dir,
        'fail_images_dir': args.fail_images_dir
    }

    main(args.api_url, args.csv_path, args.payload_file, directories)
