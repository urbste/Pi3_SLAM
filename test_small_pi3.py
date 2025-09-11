# create pi3 with vits
# create a random image
import torch
import numpy as np
import time
from pi3.models.pi3 import Pi3

model = Pi3(decoder_size="small", use_fp8_ffn=True, use_fp8_attention=True, global_merging=True, merging=0.7)

print(model)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

num_images = 1024
# create a random batch of images
images = torch.randn(1, num_images, 1, 308, 546).to(device)


start_time = time.time()
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        results = model(images)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(f"FPS: {1 / (end_time - start_time) * num_images}") # normalize FPS by number of images
