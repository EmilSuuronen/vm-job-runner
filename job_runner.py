import os
import sys
import traceback
import boto3
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
os.environ['SPCONV_ALGO'] = 'native'

# --- Configuration ---
AWS_ACCESS_KEY = ""
AWS_SECRET_KEY = ""
AWS_REGION = ""
BUCKET_NAME = ""

# Get user_id and image_id from environment (injected by backend)
user_id = os.environ.get("USER_ID")
image_id = os.environ.get("IMAGE_ID")

if not user_id or not image_id:
    print("ERROR: USER_ID or IMAGE_ID environment variable missing.")
    sys.exit(1)

input_key = f"user-images/{user_id}/{image_id}.png"
output_key = f"user-images/{user_id}/{image_id}.glb"
input_path = "/tmp/input.png"
output_path = "/tmp/output.glb"

# --- S3 Setup ---
s3 = boto3.client(
    's3',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

try:
    print(f"Downloading input image from s3://{BUCKET_NAME}/{input_key}")
    s3.download_file(BUCKET_NAME, input_key, input_path)

    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    print("Running generation...")
    image = Image.open(input_path)
    result = pipeline.run(
        image,
        seed=1,
        sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7,
        },
        slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3
        }
    )

    print("Exporting .glb...")
    glb = postprocessing_utils.to_glb(
        result['gaussian'][0],
        result['mesh'][0],
        simplify=0.95,
        texture_size=1024
    )
    glb.export(output_path)

    print(f"Uploading .glb to s3://{BUCKET_NAME}/{output_key}")
    s3.upload_file(output_path, BUCKET_NAME, output_key)

    print("Job completed successfully.")

except Exception as e:
    print("Job failed:")
    traceback.print_exc()

    sys.exit(1)
