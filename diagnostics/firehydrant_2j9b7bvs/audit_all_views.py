import os,sys,ast,json,contextlib,io,gc
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch,numpy as np,cv2
from PIL import Image
from argparse import ArgumentParser
from types import SimpleNamespace
root=Path('/home/liuyuhao/ll_further/LL-Gaussian-sg');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'submodules/Depth-Anything-V2'));os.chdir(root)
from arguments import ModelParams,OptimizationParams,PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render,prefilter_voxel
from utils.reflectance_utils import reflectance_config
from utils.pose_utils import get_camera_from_tensor,get_tensor_from_camera
from utils.composition_utils import compose_decomposed_render
from utils.loss_utils import l1_plus_loss,ssim,L_Illu,L_Smooth,L_Reflectance_Edge_Uplift,L_Reflectance_LocalContrast,L_Reflectance_HighFreq
from utils.depth_utils import relative_depth_supervision,normalize_relative_depth
from depth_anything_v2.dpt import DepthAnythingV2
experiment=root/'experiments/llrs2_firehydrant_v7_1_stablesr_6k_20261022_203503';out=experiment/'diagnostics_training_audit';out.mkdir(exist_ok=True)
meta=json.loads((root/'wandb/run-20261022_203509-2j9b7bvs/files/wandb-metadata.json').read_text())
parser=ArgumentParser(allow_abbrev=False);mp=ModelParams(parser);op=OptimizationParams(parser);pp=PipelineParams(parser)
args,_=parser.parse_known_args(meta['args']);dataset=mp.extract(args);opt=op.extract(args);pipe=pp.extract(args);dataset.model_path=str(experiment)
names=['feat_dim','n_offsets','voxel_size','update_depth','update_init_factor','update_hierachy_factor','use_feat_bank','appearance_residual_dim','ratio','add_opacity_dist','add_cov_dist','add_reflectance_dist','add_illumination_dist','add_residual_dist','use_3D_filter','use_sg_illumination','use_asg_illumination','illumination_mode','reflectance_mode','sg_lobes','sg_lambda_min','asg_lobes','asg_lambda_min']
kwargs={k:getattr(dataset,k) for k in names};kwargs.update(use_residual=False,use_dual_transient=False,reflectance_settings=reflectance_config(dataset))

reports=[]
for iteration in [2000,4000,6000]:
 model=GaussianModel(**kwargs)
 with contextlib.redirect_stdout(io.StringIO()):scene=Scene(dataset,model,None,load_iteration=iteration,shuffle=False)
 model.eval();bg=torch.zeros(3,device='cuda');metrics=[]
 with torch.no_grad():
  for view in scene.getTrainCameras():
   pose=model.get_RT(view.uid);mask=prefilter_voxel(view,model,pipe,bg,kernel_size=dataset.kernel_size,camera_pose=pose)
   p=render(view,model,pipe,bg,kernel_size=dataset.kernel_size,visible_mask=mask,camera_pose=pose)
   image=compose_decomposed_render(p);gt=view.original_image
   weighted=l1_plus_loss(image,gt,phi=.5/255).abs().mean();ss=ssim(image,gt);brightss=ssim((image*70).clamp(0,1),(gt*70).clamp(0,1))
   metrics.append(dict(view=view.image_name,weighted_l1=float(weighted),ssim=float(ss),brightened_ssim=float(brightss),psnr=float(-10*torch.log10((image-gt).square().mean())),photo_loss=float(.7*weighted+.3*(1-ss))))
 summary={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0] if k!='view'};summary.update(iteration=iteration,views=len(metrics),anchors=len(model.get_anchor),pose_requires_grad=model.P.requires_grad)
 reports.append(dict(summary=summary,per_view=metrics));print(json.dumps(summary),flush=True)
 (out/'all_training_views.json').write_text(json.dumps(reports,indent=2))
 del model,scene,p;gc.collect();torch.cuda.empty_cache()
