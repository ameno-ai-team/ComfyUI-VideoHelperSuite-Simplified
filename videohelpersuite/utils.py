import hashlib
import os
from typing import Iterable
import shutil
import subprocess
import re
from collections.abc import Mapping
import torch

from .logger import logger
import folder_paths

BIGMIN = -(2**53-1)
BIGMAX = (2**53-1)

DIMMAX = 8192

ENCODE_ARGS = ("utf-8", 'backslashreplace')

def ffmpeg_suitability(path):
    try:
        version = subprocess.run([path, "-version"], check=True,
                                 capture_output=True).stdout.decode(*ENCODE_ARGS)
    except:
        return 0
    score = 0
    #rough layout of the importance of various features
    simple_criterion = [("libvpx", 20),("264",10), ("265",3),
                        ("svtav1",5),("libopus", 1)]
    for criterion in simple_criterion:
        if version.find(criterion[0]) >= 0:
            score += criterion[1]
    #obtain rough compile year from copyright information
    copyright_index = version.find('2000-2')
    if copyright_index >= 0:
        copyright_year = version[copyright_index+6:copyright_index+9]
        if copyright_year.isnumeric():
            score += int(copyright_year)
    return score

if "VHS_FORCE_FFMPEG_PATH" in os.environ:
    ffmpeg_path = os.environ.get("VHS_FORCE_FFMPEG_PATH")
else:
    ffmpeg_paths = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        imageio_ffmpeg_path = get_ffmpeg_exe()
        ffmpeg_paths.append(imageio_ffmpeg_path)
    except:
        if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
            raise
        logger.warn("Failed to import imageio_ffmpeg")
    if "VHS_USE_IMAGEIO_FFMPEG" in os.environ:
        ffmpeg_path = imageio_ffmpeg_path
    else:
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is not None:
            ffmpeg_paths.append(system_ffmpeg)
        if os.path.isfile("ffmpeg"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg"))
        if os.path.isfile("ffmpeg.exe"):
            ffmpeg_paths.append(os.path.abspath("ffmpeg.exe"))
        if len(ffmpeg_paths) == 0:
            logger.error("No valid ffmpeg found.")
            ffmpeg_path = None
        elif len(ffmpeg_paths) == 1:
            #Evaluation of suitability isn't required, can take sole option
            #to reduce startup time
            ffmpeg_path = ffmpeg_paths[0]
        else:
            ffmpeg_path = max(ffmpeg_paths, key=ffmpeg_suitability)

ytdl_path = os.environ.get("VHS_YTDL", None) or shutil.which('yt-dlp') \
        or shutil.which('youtube-dl')
download_history = {}
def try_download_video(url):
    if ytdl_path is None:
        return None
    if url in download_history:
        return download_history[url]
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    #Format information could be added to only download audio for Load Audio,
    #but this gets hairy if same url is also used for video.
    #Best to just always keep defaults
    #dl_format = ['-f', 'ba'] if is_audio else []
    try:
        res = subprocess.run([ytdl_path, "--print", "after_move:filepath",
                              "-P", folder_paths.get_temp_directory(), url],
                             capture_output=True, check=True)
        #strip newline
        file = res.stdout.decode(*ENCODE_ARGS)[:-1]
    except subprocess.CalledProcessError as e:
        raise Exception("An error occurred in the yt-dl process:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
        file = None
    download_history[url] = file
    return file

def is_safe_path(path, strict=False):
    if "VHS_STRICT_PATHS" not in os.environ and not strict:
        return True
    basedir = os.path.abspath('.')
    try:
        common_path = os.path.commonpath([basedir, path])
    except:
        #Different drive on windows
        return False
    return common_path == basedir

def get_sorted_dir_files_from_directory(directory: str, skip_first_images: int=0, select_every_nth: int=1, extensions: Iterable=None):
    directory = strip_path(directory)
    dir_files = os.listdir(directory)
    dir_files = sorted(dir_files)
    dir_files = [os.path.join(directory, x) for x in dir_files]
    dir_files = list(filter(lambda filepath: os.path.isfile(filepath), dir_files))
    # filter by extension, if needed
    if extensions is not None:
        extensions = list(extensions)
        new_dir_files = []
        for filepath in dir_files:
            ext = "." + filepath.split(".")[-1]
            if ext.lower() in extensions:
                new_dir_files.append(filepath)
        dir_files = new_dir_files
    # start at skip_first_images
    dir_files = dir_files[skip_first_images:]
    dir_files = dir_files[0::select_every_nth]
    return dir_files


# modified from https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
def calculate_file_hash(filename: str, hash_every_n: int = 1):
    #Larger video files were taking >.5 seconds to hash even when cached,
    #so instead the modified time from the filesystem is used as a hash
    h = hashlib.sha256()
    h.update(filename.encode())
    h.update(str(os.path.getmtime(filename)).encode())
    return h.hexdigest()

requeue_guard = [None, 0, 0, {}]

def get_audio(file, start_time=0, duration=0):
    args = [ffmpeg_path, "-i", file]
    if start_time > 0:
        args += ["-ss", str(start_time)]
    if duration > 0:
        args += ["-t", str(duration)]
    try:
        #TODO: scan for sample rate and maintain
        res =  subprocess.run(args + ["-f", "f32le", "-"],
                              capture_output=True, check=True)
        audio = torch.frombuffer(bytearray(res.stdout), dtype=torch.float32)
        match = re.search(', (\\d+) Hz, (\\w+), ',res.stderr.decode(*ENCODE_ARGS))
    except subprocess.CalledProcessError as e:
        raise Exception(f"VHS failed to extract audio from {file}:\n" \
                + e.stderr.decode(*ENCODE_ARGS))
    if match:
        ar = int(match.group(1))
        #NOTE: Just throwing an error for other channel types right now
        #Will deal with issues if they come
        ac = {"mono": 1, "stereo": 2}[match.group(2)]
    else:
        ar = 44100
        ac = 2
    audio = audio.reshape((-1,ac)).transpose(0,1).unsqueeze(0)
    return {'waveform': audio, 'sample_rate': ar}

class LazyAudioMap(Mapping):
    def __init__(self, file, start_time, duration):
        self.file = file
        self.start_time=start_time
        self.duration=duration
        self._dict=None
    def __getitem__(self, key):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return self._dict[key]
    def __iter__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return iter(self._dict)
    def __len__(self):
        if self._dict is None:
            self._dict = get_audio(self.file, self.start_time, self.duration)
        return len(self._dict)
def lazy_get_audio(file, start_time=0, duration=0, **kwargs):
    return LazyAudioMap(file, start_time, duration)

def is_url(url):
    return url.split("://")[0] in ["http", "https"]

def validate_sequence(path):
    #Check if path is a valid ffmpeg sequence that points to at least one file
    (path, file) = os.path.split(path)
    if not os.path.isdir(path):
        return False
    match = re.search('%0?\\d+d', file)
    if not match:
        return False
    seq = match.group()
    if seq == '%d':
        seq = '\\\\d+'
    else:
        seq = '\\\\d{%s}' % seq[1:-1]
    file_matcher = re.compile(re.sub('%0?\\d+d', seq, file))
    for file in os.listdir(path):
        if file_matcher.fullmatch(file):
            return True
    return False

def strip_path(path):
    #This leaves whitespace inside quotes and only a single "
    #thus ' ""test"' -> '"test'
    #consider path.strip(string.whitespace+"\"")
    #or weightier re.fullmatch("[\\s\"]*(.+?)[\\s\"]*", path).group(1)
    path = path.strip()
    if path.startswith("\""):
        path = path[1:]
    if path.endswith("\""):
        path = path[:-1]
    return path
def hash_path(path):
    if path is None:
        return "input"
    if is_url(path):
        return "url"
    if not os.path.isfile(path):
        return "DNE"
    return calculate_file_hash(strip_path(path))


def validate_path(path, allow_none=False, allow_url=True):
    if path is None:
        return allow_none
    if is_url(path):
        #Probably not feasible to check if url resolves here
        if not allow_url:
            return "URLs are unsupported for this path"
        return is_safe_path(path)
    if not os.path.isfile(strip_path(path)):
        return "Invalid file path: {}".format(path)
    return is_safe_path(path)