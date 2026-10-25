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
dep=DepthAnythingV2(encoder='vitl',features=256,out_channels=[256,512,1024,1024]);dep.load_state_dict(torch.load(root/'checkpoints/depth_anything_v2_vitl.pth',map_location='cpu',weights_only=False));dep=dep.cuda().eval().requires_grad_(False)
with torch.no_grad():
 low=torch.as_tensor(dep.infer_image(cv2.imread(str(root/'datasets/LLRS2/firehydrant/images/IMG_30.png'))),device='cuda')[None]
 high=torch.as_tensor(dep.infer_image(cv2.imread(str(root/'datasets/LLRS2/firehydrant/high/IMG_30.png'))),device='cuda')[None]
 low=normalize_relative_depth(low);high=normalize_relative_depth(high)
dep.cpu();del dep;torch.cuda.empty_cache()
print('DEPTH_PRIOR_LOWHIGH_CORRELATION',float(torch.corrcoef(torch.stack([low.flatten(),high.flatten()]))[0,1]),flush=True)
reports=[]
for iteration in [2000,6000]:
 model=GaussianModel(**kwargs)
 with contextlib.redirect_stdout(io.StringIO()):scene=Scene(dataset,model,None,load_iteration=iteration,shuffle=False)
 model.train();view=next(v for v in scene.getTrainCameras() if v.image_name=='IMG_30');pose=model.get_RT(view.uid);bg=torch.zeros(3,device='cuda')
 errors=[]
 for c in scene.getTrainCameras():errors.append(float((get_camera_from_tensor(model.get_RT(c.uid))-c.world_view_transform.T).abs().max()))
 test_mismatch=[]
 for c in scene.getTestCameras():test_mismatch.append(float((get_camera_from_tensor(model.get_RT(c.uid))-c.world_view_transform.T).abs().max()))
 mask=prefilter_voxel(view,model,pipe,bg,kernel_size=dataset.kernel_size,camera_pose=pose)
 p=render(view,model,pipe,bg,kernel_size=dataset.kernel_size,visible_mask=mask,camera_pose=pose,retain_grad=True,return_reflectance_aux=True,return_training_depth=True,return_coverage=True)
 image=compose_decomposed_render(p);gt=view.original_image;R=p['render_reflectance'];L=p['render_illumination'];Le=p['render_illumination_enhanced'];aux=p['render_reflectance_aux'];depth=p['render_depth_for_loss'];coverage=p['render_depth_coverage']
 losses={}
 losses['photometric']=(1-opt.lambda_dssim)*l1_plus_loss(image,gt,phi=.5/255).abs().mean()+opt.lambda_dssim*(1-ssim(image,gt))
 losses['depth']=dataset.depth_loss_weight*relative_depth_supervision(depth,low,coverage,128,.5)
 losses['illumination']=L_Illu(gt,L)+L_Smooth(L,gt,kernel_size=9 if iteration==2000 else 5)*(dataset.warmup_illumination_smooth_reg if iteration==2000 else dataset.illumination_smooth_reg)
 losses['scale_reg']=.01*p['scaling'].prod(dim=1).mean()
 losses['R_sharpen']=dataset.reflectance_edge_uplift_reg*L_Reflectance_Edge_Uplift(aux,gt)+dataset.reflectance_contrast_reg*L_Reflectance_LocalContrast(aux,gt)+dataset.reflectance_highfreq_reg*L_Reflectance_HighFreq(aux,gt)
 if iteration==6000:
  prior=torch.tensor(np.asarray(Image.open(root/'datasets/LLRS2/firehydrant/diffusion_prior_70'/f'{view.uid}.png').convert('RGB')).copy(),device='cuda').permute(2,0,1).float()/255
  losses['enhancement_guidance']=.75*(Le*R.detach()-prior).abs().mean()
  losses['enhancement_degree']=dataset.enhancement_degree_reg*(Le.mean(0)-(L.mean(0).detach()*70).clamp(0,1)).abs().mean()+dataset.enhancement_degree_global_reg*(Le.mean()-L.mean().detach()*70).abs()
 groups={'offset':[model._offset],'scaling':[model._scaling],'features':[model._anchor_feat],'B0':[model._base_log_reflectance],'ASG_bias':[model._illum_asg_bias],'mlp_cov':list(model.mlp_cov.parameters()),'mlp_opacity':list(model.mlp_opacity.parameters()),'screen':[p['viewspace_points']]}
 params=[q for values in groups.values() for q in values]; gradtable={};refgrads=None;offsetidx=0
 for name,loss in losses.items():
  grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True);entries={};at=0
  for group,values in groups.items():
   chunk=grads[at:at+len(values)];at+=len(values);norm=sum(float(g.detach().square().sum()) for g in chunk if g is not None)**.5;entries[group]=norm
  if name=='photometric':refgrads=grads
  if name!='photometric':
   dot=norma=normb=0.
   for i,g in enumerate(grads):
    if i>=2 or g is None or refgrads[i] is None:continue
    a=g.detach();b=refgrads[i].detach();dot+=float((a*b).sum());norma+=float(a.square().sum());normb+=float(b.square().sum())
   entries['geometry_cosine_with_photo']=dot/(max(norma*normb,1e-40)**.5)
  gradtable[name]={'loss':float(loss.detach()),'norms':entries}
 report={'iteration':iteration,'pose_requires_grad':model.P.requires_grad,'max_train_pose_matrix_delta_from_colmap':max(errors),'max_test_eval_wrong_pose_matrix_delta':max(test_mismatch),'gradient_table':gradtable,'depth_quantiles':torch.quantile(depth.detach(),torch.tensor([0.,.01,.5,.99,1.],device='cuda')).cpu().tolist()}
 reports.append(report);print(json.dumps(report),flush=True)
 (out/'gradient_audit.json').write_text(json.dumps(reports,indent=2))
 del p,losses,params,grads,refgrads,model,scene;gc.collect();torch.cuda.empty_cache()
