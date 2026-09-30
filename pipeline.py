import os
import json
import asyncio
import urllib.parse
import requests
import edge_tts
from google import genai
from mega import Mega
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# ==========================================
# 1. SETUP & API KEYS
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MEGA_EMAIL = os.environ.get("MEGA_EMAIL")
MEGA_PASSWORD = os.environ.get("MEGA_PASSWORD")

# Naya SDK Initialisation
client = genai.Client(api_key=GEMINI_API_KEY)

# Temp folders setup
os.makedirs("output/audio", exist_ok=True)
os.makedirs("output/images", exist_ok=True)

# ==========================================
# 2. ASK AI TO GENERATE SCRIPT & METADATA
# ==========================================
print("🧠 Asking AI to generate a viral script...")

prompt = """
You are an automated YouTube video engine for a doodle animation channel. 
Pick a RANDOM, highly engaging topic in human history, psychology, or evolution.
Format: Calm 2nd-person narration. Short sentences.
Visuals: Hand-drawn 2D doodle cartoon, flat colors, bold outlines, NO gradients/shadows.

Return ONLY a valid JSON object starting with { and ending with }. No markdown, no explanations. 
Structure:
{
  "video_title": "[Viral title under 70 characters]",
  "seo_tags": "tag1, tag2, tag3",
  "description": "[3-sentence hook, CTA, 15 hashtags]",
  "thumbnail_prompt": "Hand-drawn 2D doodle cartoon, very catchy and dramatic scene representing the title, bold black outlines, flat colors, white background, YouTube thumbnail style",
  "script_sections": [
    {
      "text": "[Narration text max 2 sentences]",
      "image_prompt": "Hand-drawn 2D doodle cartoon animation, [SCENE DESC], flat colors, bold black outlines"
    }
  ]
}
Generate exactly 15 script_sections.
"""

response = client.models.generate_content(
    model='gemini-1.5-pro',
    contents=prompt
)

# Bulletproof JSON extraction
raw_text = response.text
start_idx = raw_text.find('{')
end_idx = raw_text.rfind('}') + 1
clean_json = raw_text[start_idx:end_idx]

data = json.loads(clean_json)
print(f"🎬 Topic Selected: {data['video_title']}")

# ==========================================
# 3. GENERATE AUDIO, IMAGES & THUMBNAIL
# ==========================================
async def generate_assets():
    # 1. Generate Thumbnail
    print("🖼️ Generating Thumbnail...")
    safe_thumb_prompt = urllib.parse.quote(data['thumbnail_prompt'])
    url = f"https://image.pollinations.ai/prompt/{safe_thumb_prompt}?width=1280&height=720&nologo=true"
    with open("output/thumbnail.png", 'wb') as handler:
        handler.write(requests.get(url).content)

    # 2. Generate Scene Assets
    for i, section in enumerate(data['script_sections']):
        scene_num = str(i).zfill(3)
        print(f"Generating Audio & Image for Scene {scene_num}...")
        
        # Audio
        audio_path = f"output/audio/scene_{scene_num}.mp3"
        communicate = edge_tts.Communicate(section['text'], "en-US-ChristopherNeural")
        await communicate.save(audio_path)
        
        # Image
        image_path = f"output/images/scene_{scene_num}.png"
        safe_prompt = urllib.parse.quote(section['image_prompt'])
        url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=1280&height=720&nologo=true"
        with open(image_path, 'wb') as handler:
            handler.write(requests.get(url).content)

asyncio.run(generate_assets())

# ==========================================
# 4. CREATE SEO TEXT FILE
# ==========================================
print("📝 Creating SEO file...")
with open('output/seo_and_description.txt', 'w', encoding='utf-8') as f:
    f.write(f"TITLE:\n{data['video_title']}\n\n")
    f.write(f"DESCRIPTION:\n{data['description']}\n\n")
    f.write(f"TAGS:\n{data['seo_tags']}")

# ==========================================
# 5. STITCH AUDIO & IMAGES INTO VIDEO (.MP4)
# ==========================================
print("🎞️ Assembling Final Video...")
clips = []
for i in range(len(data['script_sections'])):
    scene_num = str(i).zfill(3)
    audio_path = f"output/audio/scene_{scene_num}.mp3"
    image_path = f"output/images/scene_{scene_num}.png"
    
    # Load audio and image, match image duration to audio
    audio_clip = AudioFileClip(audio_path)
    img_clip = ImageClip(image_path).set_duration(audio_clip.duration).set_audio(audio_clip)
    clips.append(img_clip)

# Concatenate all clips and save as video.mp4
final_video = concatenate_videoclips(clips, method="compose")
final_video.write_videofile("output/video.mp4", fps=24, logger=None)

# Close clips to free memory
for clip in clips:
    clip.close()
final_video.close()

# ==========================================
# 6. UPLOAD TO MEGA & CLEANUP OLD FILES
# ==========================================
print("☁️ Connecting to Mega...")
mega = Mega()
m = mega.login(MEGA_EMAIL, MEGA_PASSWORD)

folder_name = "Latest_YouTube_Video"

# Safe deletion of old folder (Crash Protection)
print("🧹 Checking for old video files to delete...")
try:
    old_folder = m.find(folder_name)
    if old_folder:
        m.destroy(old_folder[0])
        print("🗑️ Old files deleted.")
except Exception as e:
    print(f"No old folder found or skipped deletion: {e}")

# Create fresh folder
folder = m.create_folder(folder_name)
folder_id = folder[folder_name]

print("🚀 Uploading new Video, Thumbnail, and SEO TXT to Mega...")
m.upload('output/video.mp4', folder_id)
m.upload('output/thumbnail.png', folder_id)
m.upload('output/seo_and_description.txt', folder_id)

print("✅ Workflow Complete! Aapka naya video aur assets Mega par aa chuke hain.")
