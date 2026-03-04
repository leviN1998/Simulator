"""

Modified example from https://github.com/neuromorphicsystems/IEBCS/

Dowload video from YouTube and save it as a .mp4 (hopefully) file.

"""
import yt_dlp

URL = 'https://www.youtube.com/watch?v=RtUQ_pz5wlo'

ydl_opts = {
    # bevorzugt AVC (H.264) Video + m4a Audio, sonst bestes MP4
    "format": "bv*[vcodec^=avc1][ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
    "outtmpl": "../data/videos/hummingbird_video.%(ext)s",
    "merge_output_format": "mp4"
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([URL])