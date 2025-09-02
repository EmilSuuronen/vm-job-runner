import os
import boto3
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# AWS credentials (IAM user with S3 read/write access)
AWS_ACCESS_KEY = ""
AWS_SECRET_KEY = ""
AWS_REGION = ""
BUCKET_NAME = ""

# Job parameters (hardcoded for test input)
user_id = ""
image_id = ""

# S3 keys
input_key = f"user-images/{user_id}/{image_id}.png"
output_key = f"user-images/{user_id}/{image_id}.glb"

# Local paths
input_path = "/tmp/input.png"
output_path = "/tmp/output.glb"

# Configure S3 client
s3 = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

print("Downloading input image from S3...")
s3.download_file(BUCKET_NAME, input_key, input_path)

print("Running TRELLIS...")
img = Image.open(input_path)
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
output = pipeline.run(img, seed=1)

print("Generating .glb file...")
glb = postprocessing_utils.to_glb(
    output['gaussian'][0],
    output['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)

glb.export(output_path)

print("Uploading .glb to S3...")
s3.upload_file(output_path, BUCKET_NAME, output_key)

print("Job complete.")

