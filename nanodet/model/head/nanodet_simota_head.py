import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nanodet.util import (
    bbox2distance,
    distance2bbox,
    images_to_levels,
    multi_apply,
    overlay_bbox_cv,
)

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss, bbox_overlaps
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from ..module.scale import Scale
from .assigner.sim_ota_assigner import SimOTAAssigner


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.true_divide(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project", torch.linspace(0, self.reg_max, self.reg_max + 1)
        )

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        shape = x.size()
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=-1)
        x = F.linear(x, self.project.type_as(x)).reshape(*shape[:-1], 4)
        return x


class NanoDetSimOTAHead(nn.Module):
    """
    :param num_classes: Number of categories excluding the background category.
    :param loss: Config of all loss functions.
    :param input_channel: Number of channels in the input feature map.
    :param feat_channels: Number of conv layers in cls and reg tower. Default: 4.
    :param stacked_convs: Number of conv layers in cls and reg tower. Default: 4.
    :param octave_base_scale: Scale factor of grid cells.
    :param strides: Down sample strides of all level feature map
    :param conv_cfg: Dictionary to construct and config conv layer. Default: None.
    :param norm_cfg: Dictionary to construct and config norm layer.
    :param reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                    in QFL setting. Default: 16.
    :param kwargs:
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=256,
        stacked_convs=4,
        octave_base_scale=4,
        strides=[8, 16, 32],
        conv_type="DWConv",
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        reg_max=16,
        share_cls_reg=False,
        activation="LeakyReLU",
        **kwargs
    ):
        super(NanoDetSimOTAHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.grid_cell_scale = octave_base_scale
        self.strides = strides
        self.reg_max = reg_max
        self.share_cls_reg = share_cls_reg
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule

        self.loss_cfg = loss
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid = self.loss_cfg.loss_qfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.assigner = SimOTAAssigner(candidate_topk=10)
        self.distribution_project = Integral(self.reg_max)

        self.loss_qfl = QualityFocalLoss(
            use_sigmoid=self.use_sigmoid,
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.share_cls_reg
                    else self.cls_out_channels,
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )
        # TODO: if
        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
                for _ in self.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                        activation=self.activation,
                    )
                )

        return cls_convs, reg_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)
        print("Finish initialize NanoDet Head.")

    def forward(self, feats):
        return multi_apply(
            self.forward_single,
            feats,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
        )

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg):
        cls_feat = x
        reg_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            feat = gfl_cls(cls_feat)
            cls_score, bbox_pred = torch.split(
                feat, [self.cls_out_channels, 4 * (self.reg_max + 1)], dim=1
            )
        else:
            cls_score = gfl_cls(cls_feat)
            bbox_pred = gfl_reg(reg_feat)

        if torch.onnx.is_in_onnx_export():
            cls_score = (
                torch.sigmoid(cls_score)
                .reshape(1, self.num_classes, -1)
                .permute(0, 2, 1)
            )
            bbox_pred = bbox_pred.reshape(1, (self.reg_max + 1) * 4, -1).permute(
                0, 2, 1
            )
        return cls_score, bbox_pred

    def loss(self, preds, gt_meta):
        cls_scores, bbox_preds = preds
        batch_size = cls_scores[0].shape[0]
        device = cls_scores[0].device
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        gt_bboxes_ignore = None

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # get grid cells of one image
        multi_level_grid_cells = [
            self.get_grid_cells(
                featmap_sizes[i],
                self.grid_cell_scale,
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        flatten_grid_cells = (
            torch.cat(multi_level_grid_cells, dim=0)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )

        # pixel cell number of multi-level feature maps
        num_level_cells = [grid_cells.size(0) for grid_cells in multi_level_grid_cells]
        num_level_cells_list = [num_level_cells] * batch_size

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 4 * (self.reg_max + 1)
            )
            for bbox_pred in bbox_preds
        ]
        flatten_bboxes = [
            distance2bbox(
                self.grid_cells_to_center(multi_level_grid_cells[i]),
                self.distribution_project(bbox_pred) * self.strides[i],
            )
            for i, bbox_pred in enumerate(flatten_bbox_preds)
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_bboxes = torch.cat(flatten_bboxes, dim=1)

        (
            foreground_mask,
            all_labels,
            all_label_weights,
            all_label_scores,
            all_bbox_targets,
            bbox_weights,
            all_grid_cell_centers,
            num_pos_per_img,
        ) = multi_apply(
            self.target_assign_single_img,
            flatten_cls_preds.detach(),
            flatten_grid_cells,
            flatten_bboxes.detach(),
            gt_bboxes,
            gt_labels,
        )
        # merge list of targets tensors into one batch then split to multi levels
        mlvl_grid_cell_centers = images_to_levels(
            all_grid_cell_centers, num_level_cells
        )
        mlvl_labels = images_to_levels(all_labels, num_level_cells)
        mlvl_label_weights = images_to_levels(all_label_weights, num_level_cells)
        mlvl_label_scores = images_to_levels(all_label_scores, num_level_cells)
        mlvl_bbox_targets = images_to_levels(all_bbox_targets, num_level_cells)

        num_total_samples = reduce_mean(
            torch.tensor(sum(num_pos_per_img)).to(device)
        ).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_qfl, losses_bbox, losses_dfl, avg_factor = multi_apply(
            self.loss_single,
            mlvl_grid_cell_centers,
            cls_scores,
            bbox_preds,
            mlvl_labels,
            mlvl_label_weights,
            mlvl_label_scores,
            mlvl_bbox_targets,
            self.strides,
            num_total_samples=num_total_samples,
        )

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        if avg_factor <= 0:
            loss_qfl = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(
                device
            )
            loss_bbox = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(
                device
            )
            loss_dfl = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(
                device
            )
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

            loss_qfl = sum(losses_qfl)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss, loss_states

    def loss_single(
        self,
        grid_cell_centers,
        cls_score,
        bbox_pred,
        labels,
        label_weights,
        label_scores,
        bbox_targets,
        stride,
        num_total_samples,
    ):

        grid_cell_centers = grid_cell_centers.reshape(-1, 4)[:, :2]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        label_scores = label_scores.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < bg_class_ind), as_tuple=False
        ).squeeze(1)

        # score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]  # (n, 4 * (reg_max + 1))
            pos_grid_centers = grid_cell_centers[pos_inds]
            pos_grid_cell_centers = pos_grid_centers / stride

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(
                pos_grid_cell_centers, pos_bbox_pred_corners
            )
            pos_decode_bbox_targets = pos_bbox_targets / stride
            # score[pos_inds] = bbox_overlaps(
            #     pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            # )
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(
                pos_grid_cell_centers, pos_decode_bbox_targets, self.reg_max
            ).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0,
            )

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0,
            )
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).to(cls_score.device)

        # qfl loss
        loss_qfl = self.loss_qfl(
            cls_score,
            (labels, label_scores),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        return loss_qfl, loss_bbox, loss_dfl, weight_targets.sum()

    @torch.no_grad()
    def target_assign_single_img(
        self, cls_preds, priors, decoded_bboxes, gt_bboxes, gt_labels
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        device = priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        center_priors = torch.cat(
            [(priors[:, :2] + priors[:, 2:]) * 0.5, priors[:, 2:] - priors[:, :2]],
            dim=-1,
        )
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = torch.zeros_like(priors)
            weight_targets = cls_target.new_zeros((0,))
            labels = priors.new_full((num_priors,), self.num_classes, dtype=torch.long)
            label_weights = priors.new_zeros(num_priors, dtype=torch.float)
            label_scores = label_weights.new_zeros(labels.shape)
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return (
                foreground_mask,
                labels,
                label_weights,
                label_scores,
                bbox_target,
                weight_targets,
                center_priors,
                0,
            )

        assign_result = self.assigner.assign(
            cls_preds.sigmoid(), center_priors, decoded_bboxes, gt_bboxes, gt_labels
        )

        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )
        num_pos_per_img = pos_inds.size(0)
        bbox_targets = torch.zeros_like(priors)
        bbox_weights = torch.zeros_like(priors)
        pos_ious = assign_result.max_overlaps[pos_inds]
        labels = priors.new_full((num_priors,), self.num_classes, dtype=torch.long)
        label_weights = priors.new_zeros(num_priors, dtype=torch.float)
        label_scores = label_weights.new_zeros(labels.shape)

        weight_targets = cls_preds.sigmoid()
        weight_targets = weight_targets.max(dim=-1)[0][pos_inds]

        if len(pos_inds) > 0:
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
                label_scores[pos_inds] = pos_ious

            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        foreground_mask = torch.zeros_like(labels).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return (
            foreground_mask,
            labels,
            label_weights,
            label_scores,
            bbox_targets,
            weight_targets,
            center_priors,
            num_pos_per_img,
        )

    def sample(self, assign_result, gt_bboxes):
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def post_process(self, preds, meta):
        cls_scores, bbox_preds = preds
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
            result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=False):

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        device = cls_scores[0].device

        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = [input_height, input_width]

        result_list = []
        for img_id in range(cls_scores[0].shape[0]):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            scale_factor = 1
            dets = self.get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                input_shape,
                scale_factor,
                device,
                rescale,
            )

            result_list.append(dets)
        return result_list

    def get_bboxes_single(
        self, cls_scores, bbox_preds, img_shape, scale_factor, device, rescale=False
    ):
        """
        Decode output tensors to bboxes on one image.
        :param cls_scores: classification prediction tensors of all stages
        :param bbox_preds: regression prediction tensors of all stages
        :param img_shape: shape of input image
        :param scale_factor: scale factor of boxes
        :param device: device of the tensor
        :return: predict boxes and labels
        """
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred in zip(self.strides, cls_scores, bbox_preds):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            featmap_size = cls_score.size()[-2:]
            y, x = self.get_single_level_center_point(
                featmap_size, stride, cls_score.dtype, device, flatten=True
            )
            center_points = torch.stack([x, y], dim=-1)
            scores = (
                cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            )
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4 * (self.reg_max + 1))
            bbox_pred = self.distribution_project(bbox_pred) * stride

            nms_pre = 1000
            if scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                center_points = center_points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(center_points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)
        # add a dummy background class at the end of all labels
        # same with mmdetection2.0
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr=0.05,
            nms_cfg=dict(type="nms", iou_threshold=0.6),
            max_num=100,
        )
        return det_bboxes, det_labels

    def get_single_level_center_point(
        self, featmap_size, stride, dtype, device, flatten=True
    ):
        """
        Generate pixel centers of a single stage feature map.
        :param featmap_size: height and width of the feature map
        :param stride: down sample stride of the feature map
        :param dtype: data type of the tensors
        :param device: device of the tensors
        :param flatten: flatten the x and y tensors
        :return: y and x of the center points
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_grid_cells(self, featmap_size, scale, stride, dtype, device):
        """
        Generate grid cells of a feature map for target assignment.
        :param featmap_size: Size of a single level feature map.
        :param scale: Grid cell scale.
        :param stride: Down sample stride of the feature map.
        :param dtype: Data type of the tensors.
        :param device: Device of the tensors.
        :return: Grid_cells xyxy position. Size should be [feat_w * feat_h, 4]
        """
        cell_size = stride * scale
        y, x = self.get_single_level_center_point(
            featmap_size, stride, dtype, device, flatten=True
        )
        grid_cells = torch.stack(
            [
                x - 0.5 * cell_size,
                y - 0.5 * cell_size,
                x + 0.5 * cell_size,
                y + 0.5 * cell_size,
            ],
            dim=-1,
        )
        return grid_cells

    def grid_cells_to_center(self, grid_cells):
        """
        Get center location of each gird cell
        :param grid_cells: grid cells of a feature map
        :return: center points
        """
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return torch.stack([cells_cx, cells_cy], dim=-1)
