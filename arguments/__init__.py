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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
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
        self.sh_degree = 3
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

        self.appearance_residual_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        # self.add_color_dist = False
        self.add_reflectance_dist = False
        self.add_illumination_dist = False
        self.add_residual_dist = False
        self.use_residual = False
        self.use_dual_transient = False
        self.use_3D_filter = False
        self.use_sg_illumination = True
        self.illumination_mode = "sg"
        self.sg_lobes = 4
        self.sg_lambda_min = 1.0
        self.sg_energy_reg = 1e-4
        self.sg_smooth_reg = 5e-5
        self.reflectance_consistency_reg = 2e-5
        self.reflectance_smooth_reg = 0.0
        self.reflectance_edge_reg = 2e-4
        self.reflectance_edge_uplift_reg = 3e-3
        self.reflectance_contrast_reg = 2e-3
        self.reflectance_highfreq_reg = 3e-3
        self.highlight_reflectance_reg = 1e-3
        self.residual_chroma_reg = 5e-4
        self.noise_residual_reg = 1.0
        self.artifact_residual_reg = 0.35
        self.noise_zero_mean_reg = 0.05
        self.noise_highfreq_reg = 0.05
        self.noise_dark_weight_reg = 0.05
        self.artifact_highlight_reg = 0.25
        self.reflectance_detail_reg = 1e-6
        self.reflectance_decoder_reg = 2e-5
        self.enhancement_reflectance_reg = 0.06
        self.enhancement_degree_reg = 0.2
        self.enhancement_degree_global_reg = 0.05
        self.enhancement_smooth_reg = 4e-4
        self.enhancement_diff_start_iter = 2500
        self.enhancement_color_reg = 0.06
        self.enhancement_color_std_reg = 0.02
        self.enhancement_green_bias_reg = 0.05
        self.enhancement_prior = "cidnet"
        self.cidnet_conda_env = "CIDNet"
        self.cidnet_root = "./submodules/HVI-CIDNet"
        self.cidnet_weights = "./submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth"
        self.cidnet_refresh_interval = 2000
        self.cidnet_mlp_steps = 100
        self.cidnet_target_exposure = 0.5
        self.cidnet_refresh_reg = 0.5
        self.cidnet_color_reg = 0.2
        self.cidnet_param_reg = 0.1
        self.cidnet_mv_reg = 0.5
        self.cidnet_gamma_init = 1.6
        self.cidnet_alpha_init = 1.6
        self.cidnet_force_refresh = False
        self.residual_hardmask_percentile = 0.8
        self.residual_higherror_percentile = 0.8
        self.residual_highlight_percentile = 0.9
        self.b0_spatial_smooth_reg = 0.0
        self.prune_ratio = 0.05
        self.beta = 1.0
        
        
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

        self.mlp_color_lr_init = 0.008 
        self.mlp_color_lr_final = 0.00005  
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000


        self.mlp_enhance_lr_init = 0.04
        self.mlp_enhance_lr_final = 0.00025

        # self.mlp_color_lr_init = 0.08
        # self.mlp_color_lr_final = 0.0005
        # self.mlp_color_lr_delay_mult = 0.01
        # self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.005
        self.appearance_lr_delay_mult = 0.01    
        self.appearance_lr_max_steps = 30_000

        self.pose_lr_init = 0.00002
        self.pose_lr_final = 0.0000002
        # self.pose_lr_init = 0.0
        # self.pose_lr_final = 0.0
        self.pose_lr_delay_mult = 0.01
        self.pose_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.3
        self.b0_lr = 0.001
        self.reflectance_offset_lr = 0.008
        self.reflectance_decoder_lr = 0.002
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        self.enhancement_from = 10_000
        self.residual_start_iter = 3_000
        self.residual_ramp_iters = 2_500
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002 

        super().__init__(parser, "Optimization Parameters")


def _backfill_model_compatibility(merged_dict):
    """Fill newly introduced model arguments and map legacy names."""
    legacy_to_new = {
        "num_sg": "sg_lobes",
        "use_sg": "use_sg_illumination",
    }
    for legacy_key, new_key in legacy_to_new.items():
        if new_key not in merged_dict and legacy_key in merged_dict:
            merged_dict[new_key] = merged_dict[legacy_key]

    defaults = {
        "use_sg_illumination": True,
        "illumination_mode": "sg",
        "use_dual_transient": False,
        "sg_lobes": 4,
        "sg_lambda_min": 1.0,
        "sg_energy_reg": 1e-4,
        "sg_smooth_reg": 5e-5,
        "reflectance_consistency_reg": 2e-5,
        "reflectance_smooth_reg": 0.0,
        "reflectance_edge_reg": 2e-4,
        "reflectance_edge_uplift_reg": 3e-3,
        "reflectance_contrast_reg": 2e-3,
        "reflectance_highfreq_reg": 3e-3,
        "highlight_reflectance_reg": 1e-3,
        "residual_chroma_reg": 5e-4,
        "noise_residual_reg": 1.0,
        "artifact_residual_reg": 0.35,
        "noise_zero_mean_reg": 0.05,
        "noise_highfreq_reg": 0.05,
        "noise_dark_weight_reg": 0.05,
        "artifact_highlight_reg": 0.25,
        "reflectance_detail_reg": 1e-6,
        "reflectance_decoder_reg": 2e-5,
        "enhancement_reflectance_reg": 0.06,
        "enhancement_degree_reg": 0.2,
        "enhancement_degree_global_reg": 0.05,
        "enhancement_smooth_reg": 4e-4,
        "enhancement_diff_start_iter": 2500,
        "enhancement_color_reg": 0.06,
        "enhancement_color_std_reg": 0.02,
        "enhancement_green_bias_reg": 0.05,
        "enhancement_prior": "cidnet",
        "cidnet_conda_env": "CIDNet",
        "cidnet_root": "./submodules/HVI-CIDNet",
        "cidnet_weights": "./submodules/HVI-CIDNet/weights/LOLv2_real/w_perc.pth",
        "cidnet_refresh_interval": 2000,
        "cidnet_mlp_steps": 100,
        "cidnet_target_exposure": 0.5,
        "cidnet_refresh_reg": 0.5,
        "cidnet_color_reg": 0.2,
        "cidnet_param_reg": 0.1,
        "cidnet_mv_reg": 0.5,
        "cidnet_gamma_init": 1.6,
        "cidnet_alpha_init": 1.6,
        "cidnet_force_refresh": False,
        "b0_lr": 0.001,
        "reflectance_offset_lr": 0.008,
        "reflectance_decoder_lr": 0.002,
        "residual_hardmask_percentile": 0.8,
        "residual_higherror_percentile": 0.8,
        "residual_highlight_percentile": 0.9,
        "b0_spatial_smooth_reg": 0.0,
        "residual_start_iter": 3_000,
        "residual_ramp_iters": 2_500,
    }
    for key, value in defaults.items():
        merged_dict.setdefault(key, value)

    return merged_dict

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    merged_dict = _backfill_model_compatibility(merged_dict)
    return Namespace(**merged_dict)
