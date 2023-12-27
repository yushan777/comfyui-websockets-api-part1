import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import random
from PIL import Image
import io
from sys import stdout

# server_address = "127.0.0.1:8188"
# server_address = "192.168.0.37:8188"
server_address = "192.168.86.218:8188"
client_id = str(uuid.uuid4())

# Sends a given prompt to the server and waits for a response
def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

# Gets an image from the server based on specified parameters
def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

# Gets the history of a prompt's execution based on its ID
def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

# Main function to send a prompt, wait for execution, and retrieve images
def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
            elif message['type'] == 'progress':
                data = message['data']
                print_progress(data['value'], data['max'])
                # When progress is complete
                if data['value'] == data['max']:
                    print("\nPrompt complete!")
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


def print_progress(value, max_value):
    bar_length = 50  # Length of the progress bar
    percent = float(value) / max_value
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    # Update the progress bar in place using stdout instead of print
    stdout.write("\rProgress: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    stdout.flush()


def get_node_by_title(prompt_workflow, title):
    # find node by title (case-insensitive)
    lower_title = title.lower()

    for id in prompt_workflow:
        node = prompt_workflow[id]
        node_title = node["_meta"]["title"].lower()
        if node_title == lower_title:
            return node

    print(f"Warning: No node found with the title '{title}'.")
    return None  # Return None if no matching node is found


# create new websocket object
ws = websocket.WebSocket()
# connect to the websocket server
ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
print(f"client_id = {client_id}")


# read workflow api data from file and convert it into dictionary 
# assign to var prompt_workflow
prompt_workflow = json.load(open('workflow_api.json'))

# create a list of prompts
prompt_list = []
prompt_list.append("photo of a man sitting in a cafe")
prompt_list.append("photo of a woman standing in the middle of a busy street")
prompt_list.append("drawing of a cat sitting in a tree")
prompt_list.append("beautiful scenery nature glass bottle landscape, purple galaxy bottle")

# give some easy-to-remember names to the nodes
chkpoint_loader_node = get_node_by_title(prompt_workflow, 'Checkpoint Loader')
prompt_pos_node = get_node_by_title(prompt_workflow, 'Pos Prompt')
empty_latent_img_node = get_node_by_title(prompt_workflow, 'Empty Latent Image')
ksampler_node = get_node_by_title(prompt_workflow, 'KSampler')
save_image_node = get_node_by_title(prompt_workflow, 'Image Saver')

# load the checkpoint
# make sure the path is correct to avoid 'HTTP Error 400: Bad Request' errors
chkpoint_loader_node["inputs"]["ckpt_name"] = "SD1-5/sd_v1-5_vae.ckpt"

# set image dimensions and batch size in EmptyLatentImage node
empty_latent_img_node["inputs"]["width"] = 512
empty_latent_img_node["inputs"]["height"] = 640
# each prompt will produce a batch of 4 images
empty_latent_img_node["inputs"]["batch_size"] = 4

# for every prompt in prompt_list...
for index, prompt in enumerate(prompt_list):

    # set the text prompt for positive CLIPTextEncode node
    prompt_pos_node["inputs"]["text"] = prompt

    # set a random seed in KSampler node
    ksampler_node["inputs"]["seed"] = random.randint(1, 18446744073709551614)

    # if it is the last prompt
    if index == 3:
        # set latent image height to 768
        empty_latent_img_node["inputs"]["height"] = 768

    # set filename prefix to be the same as prompt
    # (truncate to first 100 chars if necessary)
    fileprefix = prompt
    if len(fileprefix) > 100:
        fileprefix = fileprefix[:100]

    save_image_node["inputs"]["filename_prefix"] = fileprefix

    # get images, passing the ws and workflow
    # returns a dictionary of output_node_id (key) and list of images (value)
    images = get_images(ws, prompt_workflow)

    for node_id, image_list in images.items():

        # Collect all images from current image list
        imgs = [Image.open(io.BytesIO(image_data)) for image_data in image_list]

        # Calculate the required width and height
        total_width = sum(img.width for img in imgs)
        max_height = max(img.height for img in imgs)

        # Create a new image with the new dimensions
        merged_image = Image.new('RGB', (total_width, max_height))

        # Paste each image into the new image
        x_offset = 0
        for img in imgs:
          merged_image.paste(img, (x_offset, 0))
          x_offset += img.width

        # Show the final merged image
        merged_image.show()
