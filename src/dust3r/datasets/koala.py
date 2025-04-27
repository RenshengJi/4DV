import os.path as osp
import os
import numpy as np
import sys
import gc
from contextlib import contextmanager
from PIL import Image
try:
    lanczos = Image.Resampling.LANCZOS
    bicubic = Image.Resampling.BICUBIC
except AttributeError:
    lanczos = Image.LANCZOS
    bicubic = Image.BICUBIC

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.datasets.utils.cropping import ImageList
try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    VideoDecoder = None
    print("torchcodec not installed, please install it if you want to use it as backend.")
    print("pip install torchcodec")
    print("")

try:
    from decord import VideoReader
except ImportError:
    VideoReader = None
    print("decord not installed, please install it if you want to use it as backend.")
    print("pip install decord")
    print("")


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


class Koala_Multi(BaseMultiViewDataset):
    """Dataset of koala-36m"""

    def __init__(self, *args, ROOT, max_interval=4, length_drop_start=0.1, length_drop_end=0.9, backend="torchcodec", **kwargs):
        self.ROOT = ROOT
        self.max_interval = max_interval
        self.video = True
        self.is_metric = False
        self.length_drop_start = length_drop_start
        self.length_drop_end = length_drop_end
        if backend == "torchcodec":
            assert VideoDecoder is not None, "torchcodec is not installed."
        elif backend == "decord":
            assert VideoReader is not None, "decord is not installed."
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.backend = backend
        super().__init__(*args, **kwargs)
        assert self.split is None
        self._load_data()

    def _load_data(self):
        scenes = sorted(
            [
                d
                for d in os.listdir(self.ROOT)
                if os.path.isdir(os.path.join(self.ROOT, d))
            ]
        )
        scenes_offset = np.zeros(len(scenes) + 1, dtype=np.int32)
        for scene_idx, scene in enumerate(scenes):
            scene_dir = osp.join(self.ROOT, scene)
            clip_list = [name for name in os.listdir(scene_dir) if name.endswith(".mp4")]
            scenes_offset[scene_idx + 1] = scenes_offset[scene_idx] + len(clip_list)

        self.scenes = scenes
        self.scenes_offset = scenes_offset

    def __len__(self):
        return self.scenes_offset[-1]

    def get_stats(self):
        return f"{len(self)} groups of views"

    def _get_views(self, idx, resolution, rng, num_views):
        scene_id = np.searchsorted(self.scenes_offset, idx, side="right") - 1
        scene_name = self.scenes[scene_id]
        scene_dir = osp.join(self.ROOT, scene_name)
        clip_list = sorted([name for name in os.listdir(scene_dir) if name.endswith(".mp4")])
        clip_path = osp.join(scene_dir, clip_list[idx - self.scenes_offset[scene_id]])
        ordered_video = True

        views = []
        if self.backend == "torchcodec":
            decoder = VideoDecoder(clip_path, dimension_order="NHWC", device="cpu")
            video_length = decoder.metadata.num_frames
            if video_length < num_views:
                raise ValueError(f"Not Enough Frames in video.")
            start_pos = int(video_length * self.length_drop_start)
            end_pos = int(video_length * self.length_drop_end)  # exclusive

            sample_stride = np.clip((end_pos - start_pos) // (num_views - 1), 1, self.max_interval)
            clip_length = (num_views - 1) * sample_stride + 1
            start_idx = rng.integers(start_pos, end_pos - clip_length)
            end_idx = min(start_idx + clip_length, end_pos) - 1
            batch_index = np.linspace(start_idx, end_idx, num_views, dtype=int)
            try:
                images = decoder.get_frames_at(batch_index).data.numpy()
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. {e}.")
        elif self.backend == "decord":
            with VideoReader_contextmanager(clip_path, num_threads=2) as video_reader:
                video_length = len(video_reader)
                if video_length < num_views:
                    raise ValueError(f"Not Enough Frames in video.")
                start_pos = int(video_length * self.length_drop_start)
                end_pos = int(video_length * self.length_drop_end)  # exclusive

                sample_stride = np.clip((end_pos - start_pos) // (num_views - 1), 1, self.max_interval)
                clip_length = (num_views - 1) * sample_stride + 1
                start_idx = rng.integers(start_pos, end_pos - clip_length)
                end_idx = min(start_idx + clip_length, end_pos) - 1
                batch_index = np.linspace(start_idx, end_idx, num_views, dtype=int)

                try:
                    images = video_reader.get_batch(batch_index).asnumpy()
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. {e}.")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        images = ImageList(list(images))
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        input_resolution = np.array(images.size)  # (W,H)
        target_resolution = np.array(target_resolution)
        scale_final = max(target_resolution / input_resolution) + 1e-8
        target_resolution = np.floor(input_resolution * scale_final).astype(int)
        images = images.resize(
            target_resolution, resample=lanczos if scale_final < 1 else bicubic
        )
        l, t = np.int32(np.round((target_resolution - np.array(resolution)) * 0.5))
        crop_bbox = (l, t, l + resolution[0], t + resolution[1])
        images = images.crop(crop_bbox)
        assert images.size == resolution
        images = images.to_pil()

        for v in range(num_views):
            views.append(
                dict(
                    img=images[v],
                    dataset="Koala",
                    label=osp.relpath(clip_path, self.ROOT),
                    is_metric=self.is_metric,
                    instance=f"frame_{batch_index[v]:05d}.jpg",
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    single_view=False,
                    reset=False,
                )
            )

        return views
