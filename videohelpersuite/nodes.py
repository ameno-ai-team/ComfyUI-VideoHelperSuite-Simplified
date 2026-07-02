from .utils import ffmpeg_path
from .load_video_nodes import LoadVideoUpload
import folder_paths
import subprocess
import tempfile
import torch
import uuid
import os

def tensor_to_bytes_gpu(tensor_batch):
    scaled = torch.clamp(tensor_batch * 255.0 + 0.5, 0, 255).to(torch.uint8)
    return scaled.cpu().numpy()

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
        audio=None
    ):
        first_image = images[0]

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
                raise Exception("An error occurred converting audio")

        args = [
            ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", 'rgb24',
            "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-s", f"{first_image.shape[1]}x{first_image.shape[0]}", "-r", str(frame_rate), "-i", "-"
        ]

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

        output_process = subprocess.Popen(
            args + [file_path],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        byte_batch = tensor_to_bytes_gpu(images)

        try:
            for frame in byte_batch:
                output_process.stdin.write(frame.tobytes())
            output_process.stdin.flush()
        except BrokenPipeError:
            pass  # ffmpeg died early, real error is in stderr below

        output_process.stdin.close()
        output_process.wait()

        if output_process.returncode != 0:
            stderr = output_process.stderr.read().decode(errors="replace")
            raise Exception(f"ffmpeg failed: {stderr}")

        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
            "format": "video/h264-mp4",
            "frame_rate": frame_rate,
            "workflow": '',
            "fullpath": file_path,
        }
        return {"ui": {"gifs": [preview]}, "result": ((True, [file_path]),)}

NODE_CLASS_MAPPINGS = {
    "VHS_VideoCombine": VideoCombine,
    "VHS_LoadVideo": LoadVideoUpload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VHS_VideoCombine": "Video Combine 🎥🅥🅗🅢",
    "VHS_LoadVideo": "Load Video (Upload) 🎥🅥🅗🅢",
}
