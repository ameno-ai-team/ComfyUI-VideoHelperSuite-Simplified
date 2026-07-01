import os
import sys
import json
import subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import uuid
from string import Template

import folder_paths
from .logger import logger
from .load_video_nodes import LoadVideoUpload, LoadVideoPath
from .utils import ffmpeg_path, requeue_workflow, merge_filter_args, ENCODE_ARGS, cached, ContainsAll
from comfy.utils import ProgressBar

if 'VHS_video_formats' not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ((),{".json"})
if len(folder_paths.folder_names_and_paths['VHS_video_formats'][1]) == 0:
    folder_paths.folder_names_and_paths["VHS_video_formats"][1].add(".json")
audio_extensions = ['mp3', 'mp4', 'wav', 'ogg']

def flatten_list(l):
    ret = []
    for e in l:
        if isinstance(e, list):
            ret.extend(e)
        else:
            ret.append(e)
    return ret

def iterate_format(video_format, for_widgets=True):
    """Provides an iterator over widgets, or arguments"""
    def indirector(cont, index):
        if isinstance(cont[index], list) and (not for_widgets
          or len(cont[index])> 1 and not isinstance(cont[index][1], dict)):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield
    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)

base_formats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_formats")
@cached(5)
def get_video_formats():
    format_files = {}
    for format_name in folder_paths.get_filename_list("VHS_video_formats"):
        format_files[format_name] = folder_paths.get_full_path("VHS_video_formats", format_name)
    for item in os.scandir(base_formats_dir):
        if not item.is_file() or not item.name.endswith('.json'):
            continue
        format_files[item.name[:-5]] = item.path
    formats = []
    format_widgets = {}
    for format_name, path in format_files.items():
        with open(path, 'r') as stream:
            video_format = json.load(stream)
        widgets = list(iterate_format(video_format))
        formats.append("video/" + format_name)
        if (len(widgets) > 0):
            format_widgets["video/"+ format_name] = widgets
    return formats, format_widgets

def apply_format_widgets(format_name, kwargs):
    if os.path.exists(os.path.join(base_formats_dir, format_name + ".json")):
        video_format_path = os.path.join(base_formats_dir, format_name + ".json")
    else:
        video_format_path = folder_paths.get_full_path("VHS_video_formats", format_name)
    with open(video_format_path, 'r') as stream:
        video_format = json.load(stream)
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and 'default' in w[2]:
                default = w[2]['default']
            else:
                if type(w[1]) is list:
                    default = w[1][0]
                else:
                    #NOTE: This doesn't respect max/min, but should be good enough as a fallback to a fallback to a fallback
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
            logger.warn(f"Missing input for {w[0]} has been set to {default}")
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                #TODO: mapping=kwargs should be safer, but results in key errors, investigate why
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format

def tensor_to_int(tensor, bits):
    tensor = tensor.cpu().numpy() * (2**bits-1) + 0.5
    return np.clip(tensor, 0, (2**bits-1))
def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)
def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

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
        if len(res) > 0:
            print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)
        return self.total_frames_output

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": (
                    "INT",
                    {"default": 8, "min": 1, "step": 1},
                ),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {'formats': format_widgets}),
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
        format="image/gif",
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
        ) = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory()
        )
        output_files = []

        counter = str(uuid.uuid4())

        _, format_ext = format.split("/")

        has_alpha = first_image.shape[-1] == 4
        kwargs["has_alpha"] = has_alpha
        video_format = apply_format_widgets(format_ext, kwargs)

        if has_alpha:
            i_pix_fmt = 'rgba'
        else:
            i_pix_fmt = 'rgb24'

        file = f"{filename}_{counter}.{video_format['extension']}"
        file_path = os.path.join(full_output_folder, file)

        args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                "-s", f"{first_image.shape[1]}x{first_image.shape[0]}", "-r", str(frame_rate), "-i", "-"]

        args += video_format['main_pass']
        merge_filter_args(args)
        env = os.environ.copy()
        output_process = FfmpegProcess(args, file_path, env)
            
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            images = executor.map(lambda x: tensor_to_bytes(x).tobytes(), images, chunksize=8)

            for image in images:
                pbar.update(1)
                output_process.write_frame(image)

        output_process.close()
        output_files.append(file_path)

        if audio is not None:
            # Create audio file if input was provided
            output_file_with_audio = f"{filename}_{counter}-audio.{video_format['extension']}"
            output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
            channels = audio['waveform'].size(1)

            mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                        "-ar", str(audio['sample_rate']), "-ac", str(channels),
                        "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                        + video_format["audio_pass"] \
                        + ["-shortest", output_file_with_audio_path]

            audio_data = audio['waveform'].squeeze(0).transpose(0, 1).numpy().tobytes()
            merge_filter_args(mux_args, '-af')
            try:
                subprocess.run(mux_args, input=audio_data,
                                env=env, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                raise Exception("An error occured in the ffmpeg subprocess:\n"
                                 + e.stderr.decode(*ENCODE_ARGS))

            output_files.append(output_file_with_audio_path)
            # Return this file with audio to the webui.
            # It will be muted unless opened or saved with right click
            file = output_file_with_audio

        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
            "format": format,
            "frame_rate": frame_rate,
            "workflow": '',
            "fullpath": output_files[-1],
        }
        return {"ui": {"gifs": [preview]}, "result": ((True, output_files),)}

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
