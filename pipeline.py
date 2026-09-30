import os
import json
import asyncio
import re
import shutil
import edge_tts
import google.generativeai as genai
from mega import Mega
from diffusers import StableDiffusionPipeline
import torch

# ==========================================
# 1. SETUP & API KEYS
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MEGA_EMAIL = os.environ.get("MEGA_EMAIL")
MEGA_PASSWORD = os.environ.get("MEGA_PASSWORD")

genai.configure(api_key=GEMINI_API_KEY)

# Temp folders banana
os.makedirs("output/audio", exist_ok=True)
os.makedirs("output/images", exist_ok=True)

# ==========================================
# 2. ASK AI TO THINK OF A TOPIC & SCRIPT
# ==========================================
print("🧠 Asking AI to generate a viral script...")

prompt = """
You are an automated YouTube video engine for a doodle animation channel. 
Pick a RANDOM, highly engaging, unheard-of topic in human history, psychology, or evolution.
Format: Calm 2nd-person narration. No jargon.
Visuals: Hand-drawn 2D doodle cartoon, flat colors, bold outlines, NO gradients/shadows.

Return ONLY a valid JSON object. No markdown. Structure:
{
  "video_title": "[Viral title under 70 characters]",
  "seo_tags": "tag1, tag2, tag3",
  "description": "[3-sentence hook, CTA, 15 hashtags]",
  "script_sections": [
    {
      "text": "[Narration text max 2 sentences]",
      "image_prompt": "Hand-drawn 2D doodle cartoon animation, [SCENE DESC], flat colors, bold black outlines, educational doodle style"
    }
  ]
}
Make sure to generate at least 25 script_sections for a good length video.
"""

model = genai.GenerativeModel('gemini-1.5-pro')
response = model.generate_content(prompt)

# Clean JSON output (remove ```json tags if AI adds them)
raw_text = response.text
clean_json = re.sub(r'```json|```', '', raw_text).strip()
data = json.loads(clean_json)

print(f"🎬 Topic Selected: {data['video_title']}")

# ==========================================
# 3. GENERATE MEDIA (AUDIO + IMAGES)
# ==========================================
print("🎨 Loading Image Generator (Stable Diffusion)...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda") # Ensure GPU is used

async def generate_assets():
    for i, section in enumerate(data['script_sections']):
        scene_num = str(i).zfill(3)
        print(f"Generating Scene {scene_num}...")
        
        # Audio (Edge-TTS)
        audio_path = f"output/audio/scene_{scene_num}.mp3"
        communicate = edge_tts.Communicate(section['text'], "en-US-ChristopherNeural")
        await communicate.save(audio_path)
        
        # Image (Stable Diffusion)
        image_path = f"output/images/scene_{scene_num}.png"
        image = pipe(section['image_prompt'], num_inference_steps=25).images[0]
        image.save(image_path)

asyncio.run(generate_assets())

# Write Metadata text file
with open('output/metadata.txt', 'w', encoding='utf-8') as f:
    f.write(f"TITLE: {data['video_title']}\n\nDESC: {data['description']}\n\nTAGS: {data['seo_tags']}")

# ==========================================
# 4. UPLOAD TO MEGA DRIVE
# ==========================================
print("☁️ Connecting to Mega...")
mega = Mega()
m = mega.login(MEGA_EMAIL, MEGA_PASSWORD)

folder_name = data['video_title'].replace('/', '-').replace(':', '')[:40] # Safe folder name
folder = m.create_folder(folder_name)
folder_id = folder[folder_name]

print("🚀 Uploading files to Mega...")
m.upload('output/metadata.txt', folder_id)

for file in os.listdir('output/audio'):
    m.upload(f'output/audio/{file}', folder_id)

for file in os.listdir('output/images'):
    m.upload(f'output/images/{file}', folder_id)

print("✅ Upload Complete!")

# ==========================================
# 5. CLEANUP (To save PC Storage)
# ==========================================
print("🧹 Cleaning up local files...")
shutil.rmtree('output')
print("🎉 All Done! See you tomorrow.")
