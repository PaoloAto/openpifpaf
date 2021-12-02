import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch
import torchvision

from .annrescaler import AnnRescaler
from .. import headmeta
from ..visualizer import Cif as CifVisualizer
from ..utils import create_sink, mask_valid_area

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Cif:
    meta: headmeta.Cif
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 0.1  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[int] = 4
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return CifGenerator(self)(image, anns, meta)


class CifGenerator():
    def __init__(self, config: Cif):
        self.config = config

        self.rescaler = config.rescaler or AnnRescaler(
            config.meta.stride, config.meta.pose)
        self.visualizer = config.visualizer or CifVisualizer(config.meta)

        self.intensities = None
        self.fields_reg = None
        self.fields_bmin = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0

    def __call__(self, image, anns, meta):
        LOG.info(image.shape)
        # LOG.info(anns)
        # LOG.info(meta)
        # meta is the meta info of the image retrieved from the dataloader

        width_height_original = image.shape[2:0:-1]

        keypoint_sets = self.rescaler.keypoint_sets(anns)
        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.side_length - 1) / 2)
        valid_area = self.rescaler.valid_area(meta)
        # LOG.info('valid area: %s, pif side length = %d', valid_area, self.config.side_length)

        n_fields = len(self.config.meta.keypoints)
        self.init_fields(n_fields, bg_mask)
        self.fill(keypoint_sets)
        fields = self.fields(valid_area)

        # LOG.info(f'n_fields: {n_fields}, init_fields: {self.init_fields(n_fields, bg_mask)}, fill: {self.fill(keypoint_sets)}, kp_sets: {len(keypoint_sets)}, fields: {fields.shape}')

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns)

        # LOG.info(f'Confidence Shape: {fields[:, 0].shape}')
        # LOG.info(f'Confidence Fields: {fields[:, 0]}')
        # LOG.info(f'Values: {(fields[:, 0] > 0).nonzero(as_tuple=True)}')

        # self.visualizer._confidences(fields[:, 0])

        # LOG.info(f'Fields: {fields.shape}')

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_bmin = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

        # LOG.info(f'field_w: {field_w}, field_h: {field_h}, n_fields:{n_fields}, p: {p}')
        # LOG.info(f'intensity: {self.intensities.shape}, reg: {self.fields_reg.shape}, bmin: {self.fields_bmin.shape}, scale: {self.fields_scale.shape}, reg_l: {self.fields_reg_l.shape}')

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            self.fill_keypoints(keypoints)

    def fill_keypoints(self, keypoints):
        scale = self.rescaler.scale(keypoints)
        for f, xyv in enumerate(keypoints):
            if xyv[2] <= self.config.v_threshold:
                continue

            joint_scale = (
                scale
                if self.config.meta.sigmas is None
                else scale * self.config.meta.sigmas[f]
            )
            # f = keypoint? , xloc + yloc, v = visibility— v=0: not labeled, v=1: labeled but not visible, and v=2: labeled and visible  


            # LOG.info(f'f: {f}, xyv: {xyv}')
            self.fill_coordinate(f, xyv, joint_scale)
    

    #Notes: For the joint points of the same person, the scale value is the same, which is the square root of the (maxx-minx, maxy-miny) area 
    # which is represented by the currently visible joint points.

    # The main function fill_coordinate is to combine each of the individual joint points which are placed in the pif label's center.

    def fill_coordinate(self, f, xyv, scale):
        ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xyv[:2] - (ij + self.s_offset - self.config.padding)

        # LOG.info(f'Offset: {offset}, ij: {ij}, Min x: {minx}, Min y: {miny}, Max x: {maxx}, Max y: {maxy}')

        offset = offset.reshape(2, 1, 1)

        # LOG.info(f'Offset: {offset}')

        # mask (4,4), sink: (2, 4, 4), sink_reg: (2, 4, 4), sink_l: (4, 4)
        # -> they used numpy.linalg.norm function used to get the sum from a row or column of a matrix
        
        # sink_reg is the area the true joint point is located where in this area, the whole label of intensity is 1. (side length [4,4] of label present in the area)
        sink_reg = self.sink + offset
        # sink_l is the condition that decides the number of self.fields_reg_l is.Like if there are two joint point's sink_reg is crossed (really close/overlap), 
        # so I think the number of the crossed area should be the sink_l value that is less.
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[f, miny:maxy, minx:maxx]
        mask_peak = np.logical_and(mask, sink_l < 0.7)
        self.fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # LOG.info(f'Sink: {self.sink.shape}, sink_reg: {sink_reg.shape}, sink_l: {sink_l.shape}, mask: {mask.shape}, mask_peak: {mask_peak.shape}')
        # LOG.info(f'Fields_reg_l: {self.fields_reg_l[f, miny:maxy, minx:maxx][mask]}, Fields_reg_l Shape: {self.fields_reg_l[f, miny:maxy, minx:maxx][mask].shape}, f: {f}, minx: {minx}, miny: {miny}, maxx: {maxx}, maxy: {maxy}')

        # Intensities set the matrix value in the range of (self.side_length (4), self.side_length) centered at the joint point (x, y) to 1, 
        # corresponding to the last layer of background The same position of the channel is set to 0. The intensities matrix of pif has (n_joints+1) channels, 
        # and the last channel is the background category.

        # update intensity
        self.intensities[f, miny:maxy, minx:maxx][mask] = 1.0   
        self.intensities[f, miny:maxy, minx:maxx][mask_peak] = 1.0
        # LOG.info(f'Intensities: {self.intensities.shape}, Mask: {self.intensities[f, miny:maxy, minx:maxx][mask].shape}, Mask Peak: {self.intensities[f, miny:maxy, minx:maxx][mask_peak].shape}')

        # update regression: so also calculate the x & y offset from the points when multiple overlap the offset value of the point in that range should 
        # be the offset value nearest of the nearest joint   
        patch = self.fields_reg[f, :, miny:maxy, minx:maxx]
        patch[:, mask] = sink_reg[:, mask]
        # LOG.info(f'Patch: {patch.shape}')
        # LOG.info(f'Patch Range: {patch[:, mask].shape}')

        # update bmin => b = 0.1(config) & stride = 16
        bmin = self.config.bmin / self.config.meta.stride
        self.fields_bmin[f, miny:maxy, minx:maxx][mask] = bmin
        # LOG.info(f'bmin: {self.config.bmin} / {self.config.meta.stride}')    

        # update scale 
        assert np.isnan(scale) or 0.0 < scale < 100.0
        self.fields_scale[f, miny:maxy, minx:maxx][mask] = scale
        # LOG.info(f'scale: {self.fields_scale[f, miny:maxy, minx:maxx][mask].shape}')

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg = self.fields_reg[:, :, p:-p, p:-p]
        fields_bmin = self.fields_bmin[:, p:-p, p:-p]
        fields_scale = self.fields_scale[:, p:-p, p:-p]

        # LOG.info(f'padding: {p}, intensities: {intensities}, reg_xy: {fields_reg}, b: {fields_bmin}, scale: {fields_scale}')
        # LOG.info(f'padding: {p}, intensities: {intensities.shape}, reg_xy: {fields_reg.shape}, b: {fields_bmin.shape}, scale: {fields_scale.shape}')
        
        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale, valid_area, fill_value=np.nan)

        #Concatinates the results from the PIF part and passes a tensor of [17, 5, H, W]

        # raise Exception

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg,
            np.expand_dims(fields_bmin, 1),
            np.expand_dims(fields_scale, 1),
        ], axis=1))
