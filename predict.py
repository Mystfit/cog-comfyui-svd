import subprocess
import threading
import time
from cog import BasePredictor, Input, Path
# from typing import List
import os
import torch
import shutil
import uuid
import json
import urllib
import io
import websocket
from PIL import Image
import requests
from urllib.error import URLError
import random
import numpy as np
from equilib import equi2pers


def upload_image(url, image: Image, image_name: str):
    #image_format = get_image_format(image_path)

    #if not image_format:
    #    print(f"Error: Unsupported image format for {image_path}")
    #    return

    # Image needs to be file-like before upload
    in_mem_file = io.BytesIO()
    in_mem_file.name = image_name
    image.save(in_mem_file, format=os.path.splitext(image_name)[1][1:])
    in_mem_file.seek(0)
    files = {'image': in_mem_file}

    # Send the POST request using the requests library
    response = requests.post(url, files=files)

    # Print the server's response
    print(f"Received response: {response}")


class Predictor(BasePredictor):
    def setup(self):
        # start server
        self.server_address = "127.0.0.1:8188"
        self.start_server()

    def start_server(self):
        server_thread = threading.Thread(target=self.run_server)
        server_thread.start()

        while not self.is_server_running():
            time.sleep(1)  # Wait for 1 second before checking again

        print("Server is up and running!")

    def run_server(self):
        command = "python ./ComfyUI/main.py"
        server_process = subprocess.Popen(command, shell=True)
        server_process.wait()

    # hacky solution, will fix later
    def is_server_running(self):
        try:
            with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, "123")) as response:
                return response.status == 200
        except URLError:
            return False
    
    def queue_prompt(self, prompt, client_id):
        p = {"prompt": prompt, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req =  urllib.request.Request("http://{}/prompt".format(self.server_address), data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = requests.get(f"http://{self.server_address}/view", params=params, stream=True)
        return response.raw

        #with urllib.request.urlopen("http://{}/view?{}".format(self.server_address, url_values)) as response:
            #return response.read()

    def get_images(self, ws, prompt, client_id):
        prompt_id = self.queue_prompt(prompt, client_id)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break #Execution is done
            else:
                continue #previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history['outputs']:
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                print("node output: ", node_output)

                images_output = []
                if 'gifs' in node_output:
                    for image in node_output['gifs']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        images_output.append((image['filename'], image_data))
                output_images[node_id] = images_output

        return output_images

    def get_history(self, prompt_id):
        with urllib.request.urlopen("http://{}/history/{}".format(self.server_address, prompt_id)) as response:
            return json.loads(response.read())
    
    # TODO: add dynamic fields based on the workflow selected
    def predict(
        self,
        image: Path = Input(
            description="Inital image",
        ),
        azimuth_divisions: int = Input(
            description="Number of horizontal divisions to split the panorama into",
            default=8
        ),
        azimuth_width: int = Input(
            description="Width of each starting image",
            default=1024
        ),
        altitude_divisions: int = Input(
            description="Number of vertical divisions to split the panorama into",
            default=1
        ),
        altitude_height: int = Input(
            description="Height of each starting image",
            default=576
        ),
        steps: int = Input(
            description="Steps",
            default=20
        ),
        seed: int = Input(
            description="Sampling seed, leave Empty for Random", 
            default=None),
        width: int = Input(
            description="Output image width",
            default=1024
        ),
        height: int = Input(
            description="Output image height",
            default=576
        ),
        video_frames: int = Input(
            description="Number of frames to generate",
            default=25
        ),
        motion_bucket_id: int = Input(
            description="Higher values add more movement to the output. Values higher than 200 will hallucinate",
            default=127
        ),
        fps: int = Input(
            description="Framerate of the output",
            default=6
        )
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        # Input equirectangular image
        equi_img = Image.open(image)
        equi_img = np.asarray(equi_img)
        equi_img = np.transpose(equi_img, (2, 0, 1))

        # Calculate how many iterations we will need to correctly calculate how many images we will need
        outputs = []
        azimuth_inc = (np.pi * 2) / azimuth_divisions
        altitude_inc = (np.pi) / altitude_divisions

        for azimuth_idx in range(azimuth_divisions):
            for altitude_idx in range(altitude_divisions):

                # rotations
                rots = {
                    'roll': 0.,
                    'pitch': 0,  # rotate vertical
                    'yaw': azimuth_inc * azimuth_idx,  # rotate horizontal
                }

                # Run equi2pers
                pers_img = equi2pers(
                    equi=equi_img,
                    rots=rots,
                    height=altitude_height,
                    width=azimuth_width,
                    fov_x=90.0,
                    mode="bilinear",
                )
                pers_img = np.transpose(pers_img, (1, 2, 0))
                pers_img = Image.fromarray(pers_img)

                image_format = "png"

                # queue prompt
                img_output_path = self.get_workflow_output(
                    init_image = pers_img,
                    image_name = f"{image.stem}_{azimuth_idx}_{altitude_idx}.{image_format}",
                    steps = steps,
                    seed = seed,
                    width = width,
                    height = height,
                    video_frames = video_frames,
                    motion_bucket_id = motion_bucket_id,
                    fps = fps
                )
                outputs.append(Path(img_output_path))

        return outputs


    def get_workflow_output(
        self, 
        init_image: Image, 
        image_name: str,
        steps: int, 
        seed: int,
        width: int,
        height: int,
        video_frames: int,
        motion_bucket_id: int,
        fps: int
    ):
        # load config
        prompt = None
        workflow_config = "./custom_workflows/svd_api.json"
        with open(workflow_config, 'r') as file:
            prompt = json.load(file)

        if not prompt:
            raise Exception('no workflow config found')

        # set input variables
        prompt["3"]["inputs"]["seed"] = seed
        prompt["3"]["inputs"]["steps"] = steps
        prompt["12"]["inputs"]["width"] = width
        prompt["12"]["inputs"]["height"] = height
        prompt["12"]["inputs"]["video_frames"] = video_frames
        prompt["12"]["inputs"]["motion_bucket_id"] = motion_bucket_id
        prompt["12"]["inputs"]["fps"] = fps
        prompt["23"]["inputs"]["image"] = image_name
        prompt["26"]["inputs"]["frame_rate"] = fps
        prompt["26"]["inputs"]["filename_prefix"] = image_name

        # Upload init image
        upload_image("http://{}/upload/image".format(self.server_address), init_image, image_name)

        # start the process
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(self.server_address, client_id))
        images = self.get_images(ws, prompt, client_id)

        for node_id in images:
            for image_filename, image_data in images[node_id]:
                print(f"About to write {image_filename}")
                with open(image_filename, 'wb') as out_file:
                    shutil.copyfileobj(image_data, out_file)
                    return Path(image_filename)
