import torch
import torch.nn as nn

from src.models.asl import AsymmetricLossOptimized
from src.models.gnn_model import GNNGL_PPI
from src.train.pipeline import PipelineStep


class BuildGNNGLModelTrainStep(PipelineStep):
    def process(self, context):
        args = context.args

        context.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(context.device)
        context.graph.to(context.device)

        context.model = GNNGL_PPI(
            context.graph,
            gin_in_feature=256,
            num_layers=1,
            hidden=512,
            use_jk=False,
            train_eps=True,
            feature_fusion=None,
            class_num=7,
            fusion_strategy=args.fusion_strategy,
            feature_source=args.feature_source,
        ).to(context.device)

        context.optimizer = torch.optim.Adam(context.model.parameters(), lr=0.001, weight_decay=5e-4)

        context.scheduler = None
        if args.use_lr_scheduler:
            context.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                context.optimizer,
                mode='min',
                factor=0.5,
                patience=20,
                verbose=True,
            )

        context.loss_fn = nn.BCEWithLogitsLoss().to(context.device)
        context.loss_asl = AsymmetricLossOptimized(
            gamma_neg=1,
            gamma_pos=0,
            clip=0.05,
            disable_torch_grad_focal_loss=True,
        ).to(context.device)
        if args.loss_type == 'bce':
            context.loss_asl = context.loss_fn
        return context
