from .utils import ffmpeg_path, merge_filter_args, ENCODE_ARGS
from .load_video_nodes import LoadVideoUpload, LoadVideoPath
from concurrent.futures import ThreadPoolExecutor
from comfy.utils import ProgressBar
import folder_paths
import numpy as np
import subprocess
import tempfile
import uuid
import sys
import os

if 'VHS_video_formats' not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ((),{".json"})
if len(folder_paths.folder_names_and_paths['VHS_video_formats'][1]) == 0:
    folder_paths.folder_names_and_paths["VHS_video_formats"][1].add(".json")
audio_extensions = ['mp3', 'mp4', 'wav', 'ogg']

def tensor_to_bytes(tensor):
    tensor = tensor.cpu().numpy() * (2**8-1) + 0.5
    return np.clip(tensor, 0, (2**8-1)).astype(np.uint8)

class FfmpegProcess:
    def __init__(self, args, file_path, env):
        self.proc = subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                                      stdin=subprocess.PIPE, env=env)
        self.total_frames_output = 0

    def write_frame(self, frame_data):
        try:
            self.proc.stdin.write(frame_data)
            self.total_frames_output += 1
        except BrokenPipeError:
            res = self.proc.stderr.read()
            raise Exception("An error occurred in the ffmpeg subprocess:\n"
                             + res.decode(*ENCODE_ARGS))

    def close(self):
        self.proc.stdin.flush()
        self.proc.stdin.close()
        res = self.proc.stderr.read()
        self.proc.wait()
        if len(res) > 0:
            print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)
        return self.total_frames_output

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "step": 1},
                ),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["video/h264-mp4"],),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        images=None,
        filename_prefix="AnimateDiff",
        format="video/h264-mp4",
        audio=None,
        **kwargs
    ):
        num_frames = len(images)
        pbar = ProgressBar(num_frames)

        first_image = images[0]
        images = iter(images)

        # get output information
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        counter = str(uuid.uuid4())

        file = f"{filename}_{counter}.mp4"
        file_path = os.path.join(full_output_folder, file)
        env = os.environ.copy()

        # If audio is present, convert it to a temp WAV file up front so it
        # can be fed to ffmpeg as a second -i input in the SAME pass as the
        # video encode, instead of remuxing in a separate subprocess later.
        audio_temp_path = None
        if audio is not None:
            channels = audio['waveform'].size(1)
            audio_data = audio['waveform'].squeeze(0).transpose(0, 1).numpy().tobytes()

            audio_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_temp_path = audio_temp.name
            audio_temp.close()

            wav_args = [ffmpeg_path, "-v", "error", "-y", "-f", "f32le",
                "-ar", str(audio['sample_rate']), "-ac", str(channels),
                "-i", "-", audio_temp_path]
            try:
                subprocess.run(wav_args, input=audio_data, env=env,
                                check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                raise Exception("An error occurred converting audio:\n"
                                 + e.stderr.decode(*ENCODE_ARGS))

        args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", 'rgb24',
                "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-s", f"{first_image.shape[1]}x{first_image.shape[0]}", "-r", str(frame_rate), "-i", "-"]

        if audio_temp_path:
            args += ["-i", audio_temp_path, "-map", "0:v", "-map", "1:a", "-shortest"]

        args += [
            "-n", "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "19",
            "-preset", "ultrafast",
            "-vf", "scale=out_color_matrix=bt709",
            "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"
        ]
        if audio_temp_path:
            args += ["-c:a", "aac", "-movflags", "use_metadata_tags"]

        merge_filter_args(args)

        output_process = FfmpegProcess(args, file_path, env)

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            images = executor.map(lambda x: tensor_to_bytes(x).tobytes(), images, chunksize=8)

            for image in images:
                pbar.update(1)
                output_process.write_frame(image)

        output_process.close()

        if audio_temp_path:
            os.remove(audio_temp_path)

        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
            "format": format,
            "frame_rate": frame_rate,
            "workflow": '',
            "fullpath": file_path,
        }
        return {"ui": {"gifs": [preview]}, "result": ((True, [file_path]),)}

class VideoInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "video_info": ("VHS_VIDEOINFO",),
                    }
                }

    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"

    RETURN_TYPES = ("FLOAT","INT", "FLOAT", "INT", "INT", "FLOAT","INT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "source_fps🟨",
        "source_frame_count🟨",
        "source_duration🟨",
        "source_width🟨",
        "source_height🟨",
        "loaded_fps🟦",
        "loaded_frame_count🟦",
        "loaded_duration🟦",
        "loaded_width🟦",
        "loaded_height🟦",
    )
    FUNCTION = "get_video_info"

    def get_video_info(self, video_info):
        keys = ["fps", "frame_count", "duration", "width", "height"]

        source_info = []
        loaded_info = []

        for key in keys:
            source_info.append(video_info[f"source_{key}"])
            loaded_info.append(video_info[f"loaded_{key}"])

        return (*source_info, *loaded_info)

NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideoUpload,
    "VHS_LoadVideoPath": LoadVideoPath,
    "VHS_VideoInfo": VideoInfo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoCombine": "Video Combine 🎥🅥🅗🅢",
    "VHS_LoadVideo": "Load Video (Upload) 🎥🅥🅗🅢",
    "VHS_LoadVideoPath": "Load Video (Path) 🎥🅥🅗🅢",
    "VHS_VideoInfo": "Video Info 🎥🅥🅗🅢",
}
