source /opt/conda/etc/profile.d/conda.sh
conda activate trellis

# Export backend for Trellis
export ATTN_BACKEND=xformers

# Get metadata
USER_ID=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/USER_ID -H "Metadata-Flavor: Google")
IMAGE_ID=$(curl -s http://metadata.google.internal/computeMetadata/v1/instance/attributes/IMAGE_ID -H "Metadata-Flavor: Google")

# Run job
python3 /home/trellis/jobrunner.py "$USER_ID" "$IMAGE_ID"

# Shutdown the VM after completion
sudo shutdown -h now