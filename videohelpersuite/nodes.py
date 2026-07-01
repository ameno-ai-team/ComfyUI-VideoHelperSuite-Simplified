import os
import sys
import json
import subprocess
import numpy as np
import re
import datetime
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
from string import Template
import itertools

import folder_paths
from .logger import logger
from .load_video_nodes import LoadVideoUpload, LoadVideoPath
from .utils import ffmpeg_path, requeue_workflow, \
        gifski_path, \
        imageOrLatent, merge_filter_args, ENCODE_ARGS, floatOrInt, cached, \
        ContainsAll
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
        if "gifski_pass" in video_format and gifski_path is None:
            #Skip format
            continue
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

def ffmpeg_process(args, video_format, video_metadata, file_path, env):

    res = None
    frame_data = yield
    total_frames_output = 0
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        #metadata from file should  escape = ; # \ and newline
        def escape_ffmpeg_metadata(key, value):
            value = str(value)
            value = value.replace("\\","\\\\")
            value = value.replace(";","\\;")
            value = value.replace("#","\\#")
            value = value.replace("=","\\=")
            value = value.replace("\n","\\\n")
            return f"{key}={value}"

        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            if "prompt" in video_metadata:
                f.write(escape_ffmpeg_metadata("prompt", json.dumps(video_metadata["prompt"])) + "\n")
            if "workflow" in video_metadata:
                f.write(escape_ffmpeg_metadata("workflow", json.dumps(video_metadata["workflow"])) + "\n")
            for k, v in video_metadata.items():
                if k not in ["prompt", "workflow"]:
                    f.write(escape_ffmpeg_metadata(k, json.dumps(v)) + "\n")

        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now", "-movflags", "use_metadata_tags"]
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    #TODO: skip flush for increased speed
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                #Check if output file exists. If it does, the re-execution
                #will also fail. This obscures the cause of the error
                #and seems to never occur concurrent to the metadata issue
                if os.path.exists(file_path):
                    raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                            + err.decode(*ENCODE_ARGS))
                #Res was not set
                print(err.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                logger.warn("An error occurred when saving with metadata")
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output+=1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                raise Exception("An error occurred in the ffmpeg subprocess:\n" \
                        + res.decode(*ENCODE_ARGS))
    yield total_frames_output
    if len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)

def gifski_process(args, dimensions, frame_rate, video_format, file_path, env):
    frame_data = yield
    with subprocess.Popen(args + video_format['main_pass'] + ['-f', 'yuv4mpegpipe', '-'],
                          stderr=subprocess.PIPE, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, env=env) as procff:
        with subprocess.Popen([gifski_path] + video_format['gifski_pass']
                              + ['-W', f'{dimensions[0]}', '-H', f'{dimensions[1]}']
                              + ['-r', f'{frame_rate}']
                              + ['-q', '-o', file_path, '-'], stderr=subprocess.PIPE,
                              stdin=procff.stdout, stdout=subprocess.PIPE,
                              env=env) as procgs:
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)
                    frame_data = yield
                procff.stdin.flush()
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                outgs = procgs.stdout.read()
            except BrokenPipeError as e:
                procff.stdin.close()
                resff = procff.stderr.read()
                resgs = procgs.stderr.read()
                raise Exception("An error occurred while creating gifski output\n" \
                        + "Make sure you are using gifski --version >=1.32.0\nffmpeg: " \
                        + resff.decode(*ENCODE_ARGS) + '\ngifski: ' + resgs.decode(*ENCODE_ARGS))

class VideoCombine:
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    floatOrInt,
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {'formats': format_widgets}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        **kwargs
    ):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []),)

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
       
        first_image = images[0]
        images = iter(images)
        
        # get output information
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        (
            full_output_folder,
            filename,
            _,
            subfolder,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)
        output_files = []

        metadata = PngInfo()
        video_metadata = {}
        extra_options = {}
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            for existing_file in os.listdir(full_output_folder):
                # Check if the file matches the expected format
                match = matcher.fullmatch(existing_file)
                if match:
                    # Extract the numeric portion of the filename
                    file_counter = int(match.group(1))
                    # Update the maximum counter value if necessary
                    if file_counter > max_counter:
                        max_counter = file_counter

            # Increment the counter by 1 to get the next available value
            counter = max_counter + 1
            output_process = None

        # save first frame as png to keep metadata
        first_image_file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, first_image_file)
        if extra_options.get('VHS_MetadataImage', True) != False:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                file_path,
                pnginfo=metadata,
                compress_level=4,
            )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")

        has_alpha = first_image.shape[-1] == 4
        kwargs["has_alpha"] = has_alpha
        video_format = apply_format_widgets(format_ext, kwargs)
        dim_alignment = video_format.get("dim_alignment", 2)
        if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
            #output frames must be padded
            to_pad = (-first_image.shape[1] % dim_alignment,
                        -first_image.shape[0] % dim_alignment)
            padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                        to_pad[1]//2, to_pad[1] - to_pad[1]//2)
            padfunc = torch.nn.ReplicationPad2d(padding)
            def pad(image):
                image = image.permute((2,0,1))#HWC to CHW
                padded = padfunc(image.to(dtype=torch.float32))
                return padded.permute((1,2,0))
            images = map(pad, images)
            dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                            -first_image.shape[0] % dim_alignment + first_image.shape[0])
            logger.warn("Output images were not of valid resolution and have had padding applied")
        else:
            dimensions = (first_image.shape[1], first_image.shape[0])

        loop_args = []
        # if video_format.get('input_color_depth', '8bit') == '16bit':
        #     images = map(tensor_to_shorts, images)
        #     if has_alpha:
        #         i_pix_fmt = 'rgba64'
        #     else:
        #         i_pix_fmt = 'rgb48'
        # else:
        images = map(tensor_to_bytes, images)
        if has_alpha:
            i_pix_fmt = 'rgba'
        else:
            i_pix_fmt = 'rgb24'
                
        file = f"{filename}_{counter:05}.{video_format['extension']}"
        file_path = os.path.join(full_output_folder, file)
        
        args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                # The image data is in an undefined generic RGB color space, which in practice means sRGB.
                # sRGB has the same primaries and matrix as BT.709, but a different transfer function (gamma),
                # called by the sRGB standard name IEC 61966-2-1. However, video hosting platforms like YouTube
                # standardize on full BT.709 and will convert the colors accordingly. This last minute change
                # in colors can be confusing to users. We can counter it by lying about the transfer function
                # on a per format basis, i.e. for video we will lie to FFmpeg that it is already BT.709. Also,
                # because the input data is in RGB (not YUV) it is more efficient (fewer scale filter invocations)
                # to specify the input color space as RGB and then later, if the format actually wants YUV,
                # to convert it to BT.709 YUV via FFmpeg's -vf "scale=out_color_matrix=bt709".
                "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-"] \
                + loop_args

        images = map(lambda x: x.tobytes(), images)
        env=os.environ.copy()

        if "inputs_main_pass" in video_format:
            in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
            args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

        if output_process is None:
            args += video_format['main_pass']
            merge_filter_args(args)
            output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
            
            #Proceed to first yield
            output_process.send(None)
            if meta_batch is not None:
                meta_batch.outputs[unique_id] = (counter, output_process)

        for image in images:
            pbar.update(1)
            output_process.send(image)
        if meta_batch is not None:
            requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
        if meta_batch is None or meta_batch.has_closed_inputs:
            #Close pipe and wait for termination.
            try:
                total_frames_output = output_process.send(None)
                output_process.send(None)
            except StopIteration:
                pass
            if meta_batch is not None:
                meta_batch.outputs.pop(unique_id)
                if len(meta_batch.outputs) == 0:
                    meta_batch.reset()
        else:
            #batch is unfinished
            #TODO: Check if empty output breaks other custom nodes
            return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

        output_files.append(file_path)

        a_waveform = None
        if audio is not None:
            try:
                #safely check if audio produced by VHS_LoadVideo actually exists
                a_waveform = audio['waveform']
            except:
                pass
        if a_waveform is not None:
            # Create audio file if input was provided
            output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
            output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
            if "audio_pass" not in video_format:
                logger.warn("Selected video format does not have explicit audio support")
                video_format["audio_pass"] = ["-c:a", "libopus"]


            # FFmpeg command with audio re-encoding
            #TODO: expose audio quality options if format widgets makes it in
            #Reconsider forcing apad/shortest
            channels = audio['waveform'].size(1)
            min_audio_dur = total_frames_output / frame_rate + 1
            if video_format.get('trim_to_audio', 'False') != 'False':
                apad = []
            else:
                apad = ["-af", "apad=whole_dur="+str(min_audio_dur)]
            mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                        "-ar", str(audio['sample_rate']), "-ac", str(channels),
                        "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                        + video_format["audio_pass"] \
                        + apad + ["-shortest", output_file_with_audio_path]

            audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                    .numpy().tobytes()
            merge_filter_args(mux_args, '-af')
            try:
                res = subprocess.run(mux_args, input=audio_data,
                                        env=env, capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                raise Exception("An error occured in the ffmpeg subprocess:\n" \
                        + e.stderr.decode(*ENCODE_ARGS))
            if res.stderr:
                print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
            output_files.append(output_file_with_audio_path)
            #Return this file with audio to the webui.
            #It will be muted unless opened or saved with right click
            file = output_file_with_audio
            
        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output" if save_output else "temp",
            "format": format,
            "frame_rate": frame_rate,
            "workflow": first_image_file,
            "fullpath": output_files[-1],
        }
        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}

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
