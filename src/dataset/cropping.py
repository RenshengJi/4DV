"""
Cropping utilities for dataset processing.
Extracted from dust3r to avoid dependencies.
"""
import PIL.Image
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from dataset.utils import colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


class ImageList:
    """Convenience class to apply the same operation to a set of images."""

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch("resize", *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch("crop", *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def camera_matrix_of_crop(
    input_camera_matrix,
    input_resolution,
    output_resolution,
    scaling=1,
    offset_factor=0.5,
    offset=None,
):
    """Compute camera matrix after cropping/resizing"""
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def bbox_from_intrinsics_in_out(
    input_camera_matrix, output_camera_matrix, output_resolution
):
    """Get bounding box from camera matrices"""
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l + out_width, t + out_height)
    return crop_bbox


def rescale_image_depthmap(
    image, depthmap, camera_intrinsics, output_resolution, force=True
):
    """Rescale image and depthmap to target resolution"""
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    image = image.resize(
        output_resolution, resample=lanczos if scale_final < 1 else bicubic
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )

    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final
    )

    return image.to_pil(), depthmap, camera_intrinsics


def rescale_image_depthmap_flowmap_segmask(
    image, depthmap, flowmap, seg_mask, camera_intrinsics, output_resolution, force=True
):
    """Rescale image, depthmap, flowmap, and segmentation mask to target resolution"""
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]
    if flowmap is not None:
        assert tuple(flowmap.shape[:2]) == image.size[::-1]
    if seg_mask is not None:
        assert tuple(seg_mask.shape[:2]) == image.size[::-1]

    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:
        return (image.to_pil(), depthmap, flowmap, seg_mask, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    image = image.resize(
        output_resolution, resample=lanczos if scale_final < 1 else bicubic
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap, output_resolution, fx=scale_final, fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )
    if flowmap is not None:
        flowmap = cv2.resize(
            flowmap, output_resolution, fx=scale_final, fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )
    if seg_mask is not None:
        seg_mask = cv2.resize(
            seg_mask, output_resolution, fx=scale_final, fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )

    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final
    )

    return image.to_pil(), depthmap, flowmap, seg_mask, camera_intrinsics


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """Crop image and depthmap"""
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    depthmap = depthmap[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics


def crop_image_depthmap_flowmap_segmask(image, depthmap, flowmap, seg_mask, camera_intrinsics, crop_bbox):
    """Crop image, depthmap, flowmap, and segmentation mask"""
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))
    if depthmap is not None:
        depthmap = depthmap[t:b, l:r]
    if flowmap is not None:
        flowmap = flowmap[t:b, l:r]
    if seg_mask is not None:
        seg_mask = seg_mask[t:b, l:r]

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, flowmap, seg_mask, camera_intrinsics
