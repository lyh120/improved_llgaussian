#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from argparse import ArgumentParser, Namespace

from utils.model_format import MODEL_FORMAT_VERSION

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            if key == "model_format_version":
                # The file format is an invariant, not a user-selectable mode.
                parser.set_defaults(
                    model_format_version=value if not fill_none else None
                )
                continue
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.model_format_version = MODEL_FORMAT_VERSION
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4
        self.kernel_size = 0.1


        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = 1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.use_3D_filter = False
        # Scaffold anchor initialization only.  ``prune_ratio=1`` keeps the
        # complete input point cloud; smaller values enable the baseline's
        # distance-aware stochastic pruning controlled by ``beta``.
        self.prune_ratio = 1.0
        self.beta = 1.0
        self.enhancement_prior_path = ""
        self.depth_prior_path = ""
        self.enhancement_prior_backend = "cidnet"
        self.cidnet_root = "submodules/HVI-CIDNet"
        self.cidnet_weights = "submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth"
        self.stablesr_root = "submodules/StableSR"
        self.stablesr_python = ""
        self.stablesr_config = "configs/stableSRNew/v2-finetune_text_T_512.yaml"
        self.stablesr_checkpoint = "checkpoints/stablesr_turbo.ckpt"
        self.stablesr_vqgan_checkpoint = "checkpoints/vqgan_cfw_00011.ckpt"
        # StableSR illumination is explicit and reproducible. It is never
        # inferred from image brightness through a hidden 0.45/mean rule.
        self.stablesr_input_gain = 15.0
        self.depth_anything_root = "submodules/Depth-Anything-V2"
        self.depth_anything_encoder = "vitl"
        self.depth_anything_checkpoint = "checkpoints/depth_anything_v2_vitl.pth"
        self.depth_anything_input_size = 518
        
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        # self.feature_lr = 0.075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000

        self.explicit_appearance_lr_init = 0.008
        self.explicit_appearance_lr_final = 0.00005
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.pose_lr_init = 0.00002
        self.pose_lr_final = 0.0000002
        # self.pose_lr_init = 0.0
        # self.pose_lr_final = 0.0
        self.pose_lr_delay_mult = 0.01
        self.pose_lr_max_steps = 30_000

        self.lambda_dssim = 0.2
        self.lambda_scaling = 0.01
        self.lambda_reflectance_reconstruction = 1.0
        self.lambda_illumination = 1.0
        self.lambda_enhanced = 1.0
        self.lambda_depth = 1.0
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 2_000
        self.update_interval = 100
        self.update_until = 15_000
        # Warmup starts from the complete input point cloud.  Its separate,
        # small densification budget improves coverage without compounding the
        # main-stage anchor growth.
        self.warmup_start_stat = 200
        self.warmup_iterations = 2_000
        self.warmup_geometry_lr_scale = 0.1
        self.warmup_update_from = 1_200
        self.warmup_update_until = 1_900
        self.warmup_update_interval = 200
        self.warmup_max_new_anchors = 128
        self.warmup_level_caps = "64,40,24"
        self.warmup_densify_grad_threshold = 0.00025
        self.warmup_success_threshold = 0.8
        # Keep anchor growth bounded and reproducible.  These values are
        # intentionally conservative for the short 8k training schedule.
        self.max_anchors = 30_000
        self.max_new_anchors_per_update = 256
        self.densify_level_caps = "128,80,48"
        self.anchor_prune_grace_iters = 1_200
        self.prune_from_iter = 0
        self.max_pruned_anchors_per_update = 0
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002 

        super().__init__(parser, "Optimization Parameters")


def _has_adjust_event(
    stage_start: int,
    start_stat: int,
    update_from: int,
    update_until: int,
    update_interval: int,
    iterations: int,
) -> bool:
    """Return whether a stage can reach an anchor-adjustment iteration."""
    if update_interval <= 0:
        raise ValueError("Scaffold update intervals must be positive")
    first_candidate = max(stage_start, start_stat + 1, update_from + 1)
    last_candidate = min(iterations, update_until - 1)
    first_event = (
        (first_candidate + update_interval - 1) // update_interval
    ) * update_interval
    return first_event <= last_candidate


def _growth_budget_enabled(max_new_anchors: int, level_caps) -> bool:
    if max_new_anchors <= 0:
        return False
    if isinstance(level_caps, str):
        level_caps = [int(value) for value in level_caps.split(",") if value.strip()]
    return not level_caps or any(int(value) > 0 for value in level_caps)


def validate_training_schedule(args) -> None:
    """Reject active anchor growth that reaches the end without refinement."""
    errors = []
    warmup_enabled = bool(getattr(args, "warmup", False))
    # Both stages use their own local 1..N counter even though checkpoints
    # retain a monotonic global iteration.
    main_stage_start = 1
    main_adjust_event = _has_adjust_event(
        main_stage_start,
        args.start_stat,
        args.update_from,
        args.update_until,
        args.update_interval,
        args.iterations,
    )
    main_growth_event = main_adjust_event and _growth_budget_enabled(
        args.max_new_anchors_per_update,
        args.densify_level_caps,
    )
    if main_growth_event and args.update_until >= args.iterations:
        errors.append(
            f"update_until ({args.update_until}) must be smaller than iterations "
            f"({args.iterations}) so inherited appearance has a refinement tail"
        )

    if args.max_pruned_anchors_per_update > 0:
        pruning_event = _has_adjust_event(
            max(1, args.prune_from_iter),
            args.start_stat,
            args.update_from,
            args.update_until,
            args.update_interval,
            args.iterations,
        )
        if not main_adjust_event or not pruning_event:
            errors.append(
                "training-time pruning is enabled but no main-stage anchor "
                "adjustment can occur at or after prune_from_iter "
                f"({args.prune_from_iter}); require prune_from_iter < "
                f"min(update_until, iterations) and a matching update interval"
            )

    if warmup_enabled:
        warmup_growth_event = _has_adjust_event(
            1,
            args.warmup_start_stat,
            args.warmup_update_from,
            args.warmup_update_until,
            args.warmup_update_interval,
            args.warmup_iterations,
        ) and _growth_budget_enabled(
            args.warmup_max_new_anchors,
            args.warmup_level_caps,
        )
        if warmup_growth_event and args.warmup_update_until >= args.warmup_iterations:
            errors.append(
                f"warmup_update_until ({args.warmup_update_until}) must be smaller "
                f"than warmup_iterations ({args.warmup_iterations}) so warmup growth has a "
                "refinement tail"
            )
    if errors:
        raise ValueError("Invalid Scaffold densification schedule:\n- " + "\n- ".join(errors))


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    config_loaded = False
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
            config_loaded = True
    except (TypeError, FileNotFoundError):
        print("Config file not found at")
    args_cfgfile = eval(cfgfile_string)
    cfg_dict = vars(args_cfgfile).copy()
    if config_loaded and cfg_dict.get("model_format_version") != MODEL_FORMAT_VERSION:
        raise RuntimeError(
            "This configuration predates MODEL_FORMAT_VERSION=2 and cannot be loaded. "
            "A command-line value cannot upgrade an old cfg_args file."
        )

    merged_dict = cfg_dict
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    version = merged_dict.get("model_format_version")
    if merged_dict and version != MODEL_FORMAT_VERSION:
        raise RuntimeError(
            "This configuration predates MODEL_FORMAT_VERSION=2 and cannot be loaded. "
            "Start a new explicit-appearance experiment."
        )
    return Namespace(**merged_dict)
