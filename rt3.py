@register
class DETRHead(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'use_focal_loss']
    __inject__ = ['loss']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 nhead=8,
                 num_mlp_layers=3,
                 loss='DETRLoss',
                 fpn_dims=[1024, 512, 256],
                 with_mask_head=False,
                 use_focal_loss=False):
        super(DETRHead, self).__init__()
        # add background class
        self.num_classes = num_classes if use_focal_loss else num_classes + 1
        self.hidden_dim = hidden_dim
        self.loss = loss
        self.with_mask_head = with_mask_head
        self.use_focal_loss = use_focal_loss

        self.score_head = nn.Linear(hidden_dim, self.num_classes) ####
        self.bbox_head = MLP(hidden_dim,
                             hidden_dim,
                             output_dim=4,
                             num_layers=num_mlp_layers)
        if self.with_mask_head:
            self.bbox_attention = MultiHeadAttentionMap(hidden_dim, hidden_dim,
                                                        nhead)
            self.mask_head = MaskHeadFPNConv(hidden_dim + nhead, fpn_dims,
                                             hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.score_head)  ####

    @classmethod
    def from_config(cls, cfg, hidden_dim, nhead, input_shape):

        return {
            'hidden_dim': hidden_dim,
            'nhead': nhead,
            'fpn_dims': [i.channels for i in input_shape[::-1]][1:]
        }

    @staticmethod
    def get_gt_mask_from_polygons(gt_poly, pad_mask):
        out_gt_mask = []
        for polygons, padding in zip(gt_poly, pad_mask):
            height, width = int(padding[:, 0].sum()), int(padding[0, :].sum())
            masks = []
            for obj_poly in polygons:
                rles = mask_util.frPyObjects(obj_poly, height, width)
                rle = mask_util.merge(rles)
                masks.append(
                    paddle.to_tensor(mask_util.decode(rle)).astype('float32'))
            masks = paddle.stack(masks)
            masks_pad = paddle.zeros(
                [masks.shape[0], pad_mask.shape[1], pad_mask.shape[2]])
            masks_pad[:, :height, :width] = masks
            out_gt_mask.append(masks_pad)
        return out_gt_mask

    def forward(self, out_transformer, body_feats, inputs=None):
        r"""
        Args:
            out_transformer (Tuple): (feats: [num_levels, batch_size,
                                                num_queries, hidden_dim],
                            memory: [batch_size, hidden_dim, h, w],
                            src_proj: [batch_size, h*w, hidden_dim],
                            src_mask: [batch_size, 1, 1, h, w])
            body_feats (List(Tensor)): list[[B, C, H, W]]
            inputs (dict): dict(inputs)
        """
        feats, memory, src_proj, src_mask = out_transformer
        outputs_logit = self.score_head(feats)      ####
        outputs_bbox = F.sigmoid(self.bbox_head(feats))
        outputs_seg = None
        if self.with_mask_head:
            bbox_attention_map = self.bbox_attention(feats[-1], memory,
                                                     src_mask)
            fpn_feats = [a for a in body_feats[::-1]][1:]
            outputs_seg = self.mask_head(src_proj, bbox_attention_map,
                                         fpn_feats)
            outputs_seg = outputs_seg.reshape([
                feats.shape[1], feats.shape[2], outputs_seg.shape[-2],
                outputs_seg.shape[-1]
            ])

        if self.training:
            assert inputs is not None
            assert 'gt_bbox' in inputs and 'gt_class' in inputs
            gt_mask = self.get_gt_mask_from_polygons(
                inputs['gt_poly'],
                inputs['pad_mask']) if 'gt_poly' in inputs else None
            return self.loss(
                outputs_bbox,
                outputs_logit,      ####
                inputs['gt_bbox'],
                inputs['gt_class'],
                masks=outputs_seg,
                gt_mask=gt_mask)
        else:
            return (outputs_bbox[-1], outputs_logit[-1], outputs_seg)   ####

