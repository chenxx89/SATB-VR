import os

VALID_IMAGE_EXTENSIONS = ('jpg', 'jpeg', 'png', 'webp', 'bmp')
VALID_VIDEO_EXTENSIONS = ('mp4', 'mov', 'avi', 'webm', 'mkv', 'y4m')


def _has_ext(filename, valid_exts):
    return filename.split('.')[-1].lower() in valid_exts


def collect_video_paths(input_dir):
    if os.path.isfile(input_dir):
        if not _has_ext(input_dir, VALID_VIDEO_EXTENSIONS):
            raise ValueError(f"Unsupported video file: {input_dir}")
        return [input_dir]

    validation_paths = []
    vnames = os.listdir(input_dir)
    is_image_dirs = os.path.isdir(os.path.join(input_dir, vnames[0]))

    for vn in vnames:
        path = os.path.join(input_dir, vn)
        if is_image_dirs:
            fnames = os.listdir(path)
            if not _has_ext(fnames[0], VALID_IMAGE_EXTENSIONS):
                continue
        else:
            if not _has_ext(vn, VALID_VIDEO_EXTENSIONS):
                continue
        validation_paths.append(path)

    return validation_paths
