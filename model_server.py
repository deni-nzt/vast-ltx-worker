# LTX 2.3 model server for Vast Serverless. Loads the 22B transformer ONCE (resident),
# serves POST /generate (image_b64 -> video_b64) + GET /health.
#
# PIPELINE MODES (env LTX_PIPELINE):
#   one_stage (default) — TI2VidOneStagePipeline, 8-step distilled at the REQUEST's height/width.
#   two_stage           — DistilledPipeline (official distilled two-stage): stage 1 at half res,
#                         2x latent upsample, distilled refine at LTX_TARGET_HEIGHT x LTX_TARGET_WIDTH
#                         (default 1024x1536 — the request's dims are IGNORED; the operator picks the
#                         tier via env). Same dev checkpoint + distilled LoRA (= the distilled model),
#                         plus the spatial upscaler. ONE DiffusionStage serves both stages, so the
#                         resident cache stays a single ~44GB transformer — no VRAM doubling.
#
# Checkpoint / quantization / offload are env-driven (LTX_CKPT / LTX_QUANT / LTX_OFFLOAD) so the SAME
# image runs bf16-on-80GB or fp8-cast-on-cheaper GPUs without a rebuild.
import os, time, base64, tempfile, threading, json, urllib.request, traceback
import torch
from contextlib import contextmanager
from fastapi import FastAPI
from pydantic import BaseModel

# ---- THE RESIDENT FIX: cache the transformer build, never free it ----
from ltx_pipelines.utils.blocks import DiffusionStage
_TCACHE={}; _LOAD=[0.0]
@contextmanager
def _resident_tctx(self, **kwargs):
    k=id(self)
    if k not in _TCACHE:
        t=time.time(); _TCACHE[k]=self._build_transformer(**kwargs); _LOAD[0]=time.time()-t
        print(f"[worker] MODEL_LOAD {_LOAD[0]:.1f}s (resident)",flush=True)
    yield _TCACHE[k]   # reuse, no teardown
DiffusionStage._transformer_ctx=_resident_tctx

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.utils.args import default_1_stage_arg_parser, default_2_stage_distilled_arg_parser
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.distilled import DistilledPipeline

CKPT=os.environ.get("LTX_CKPT","/workspace/models/checkpoints/ltx-2.3-22b-dev.safetensors"); GEMMA=os.environ.get("LTX_GEMMA","/workspace/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"); LORA=os.environ.get("LTX_LORA","/workspace/models/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
UPSCALER=os.environ.get("LTX_UPSCALER","/workspace/models/upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
QUANT=os.environ.get("LTX_QUANT","").strip()      # ""=bf16 resident (80GB); "fp8-cast"=lossless fp8 (fits 48GB+; offload for less)
OFFLOAD=os.environ.get("LTX_OFFLOAD","none").strip()  # "none" default; other modes push the text encoder/blocks to CPU for cheaper GPUs
MODE=os.environ.get("LTX_PIPELINE","one_stage").strip()
TARGET_H=int(os.environ.get("LTX_TARGET_HEIGHT","1024")); TARGET_W=int(os.environ.get("LTX_TARGET_WIDTH","1536"))

params=detect_params(CKPT)
print(f"[worker] building pipeline mode={MODE}...",flush=True)
if MODE=="two_stage":
    parser=default_2_stage_distilled_arg_parser(params=params)
    def mkargs2(p,img,h,w,nf):
        a=["--distilled-checkpoint-path",CKPT,"--gemma-root",GEMMA,"--spatial-upsampler-path",UPSCALER,"--lora",LORA,"1.0","--offload",OFFLOAD,"--prompt",p,"--image",img,"0","1.0","--output-path","/tmp/_unused.mp4","--height",str(h),"--width",str(w),"--num-frames",str(nf),"--frame-rate","25"]
        if QUANT: a+=["--quantization",QUANT]
        return parser.parse_args(a)
    _a=mkargs2("x","/workspace/test.png",TARGET_H,TARGET_W,97)
    PIPELINE=DistilledPipeline(distilled_checkpoint_path=_a.distilled_checkpoint_path,gemma_root=_a.gemma_root,spatial_upsampler_path=_a.spatial_upsampler_path,loras=tuple(_a.lora) if _a.lora else (),quantization=_a.quantization,compilation_config=_a.compile,offload_mode=_a.offload_mode)
else:
    parser=default_1_stage_arg_parser(params=params)
    def mkargs(p,n,img,h,w,nf,st,out):
        a=["--checkpoint-path",CKPT,"--gemma-root",GEMMA,"--lora",LORA,"1.0","--offload",OFFLOAD,"--prompt",p,"--negative-prompt",n,"--image",img,"0","1.0","--output-path",out,"--num-inference-steps",str(st),"--height",str(h),"--width",str(w),"--num-frames",str(nf),"--frame-rate","25"]
        if QUANT: a+=["--quantization",QUANT]
        return parser.parse_args(a)
    _a=mkargs("x","","/workspace/test.png",512,768,97,8,"/tmp/_.mp4")
    PIPELINE=TI2VidOneStagePipeline(checkpoint_path=_a.checkpoint_path,gemma_root=_a.gemma_root,loras=tuple(_a.lora) if _a.lora else (),quantization=_a.quantization,compilation_config=_a.compile,offload_mode=_a.offload_mode)
print("[worker] PIPELINE_READY",flush=True)
def vgp(a): return MultiModalGuiderParams(cfg_scale=a.video_cfg_guidance_scale,stg_scale=a.video_stg_guidance_scale,rescale_scale=a.video_rescale_scale,modality_scale=a.a2v_guidance_scale,skip_step=a.video_skip_step,stg_blocks=a.video_stg_blocks)
def agp(a): return MultiModalGuiderParams(cfg_scale=a.audio_cfg_guidance_scale,stg_scale=a.audio_stg_guidance_scale,rescale_scale=a.audio_rescale_scale,modality_scale=a.v2a_guidance_scale,skip_step=a.audio_skip_step,stg_blocks=a.audio_stg_blocks)

# Vast Serverless + the PyWorker own worker lifecycle (idle workers are stopped by the autoscaler),
# so there is NO self-stop here. LAST/BUSY are kept only for /health + logging.
LAST=[time.time()]; BUSY=[False]

app=FastAPI(); STATS=[]
class Req(BaseModel):
    prompt: str
    image_b64: str
    negative_prompt: str="worst quality, blurry, morphing, distortion"
    height: int=512
    width: int=768
    num_frames: int=97
    steps: int=8
@app.get("/health")
def health(): return {"ok":True,"mode":MODE,"resident":bool(_TCACHE),"model_load_s":round(_LOAD[0],1),"clips_done":len(STATS),"batch_avg_s":round(sum(STATS)/len(STATS),1) if STATS else 0}
@app.post("/generate")
def generate(r: Req):
    BUSY[0]=True; LAST[0]=time.time(); t0=time.time(); img=tempfile.mktemp(suffix=".png"); out=tempfile.mktemp(suffix=".mp4")
    open(img,"wb").write(base64.b64decode(r.image_b64))
    try:
        warm=bool(_TCACHE); ts=time.time()
        if MODE=="two_stage":
            # Final res is the OPERATOR's tier (env), not the request's; the pipeline halves internally
            # for stage 1. Distilled path = fixed sigmas, no CFG (negative_prompt/steps are inert).
            a=mkargs2(r.prompt,img,TARGET_H,TARGET_W,r.num_frames)
            tiling=TilingConfig.default()
            with torch.inference_mode():
                video,audio=PIPELINE(prompt=a.prompt,seed=a.seed,height=TARGET_H,width=TARGET_W,num_frames=a.num_frames,frame_rate=a.frame_rate,images=a.images,tiling_config=tiling)
                gen=time.time()-ts; te=time.time()
                encode_video(video=video,fps=a.frame_rate,audio=audio,output_path=out,video_chunks_number=get_video_chunks_number(a.num_frames,tiling)); enc=time.time()-te
        else:
            a=mkargs(r.prompt,r.negative_prompt,img,r.height,r.width,r.num_frames,r.steps,out)
            with torch.inference_mode():
                video,audio=PIPELINE(prompt=a.prompt,negative_prompt=a.negative_prompt,seed=a.seed,height=a.height,width=a.width,num_frames=a.num_frames,frame_rate=a.frame_rate,num_inference_steps=a.num_inference_steps,video_guider_params=vgp(a),audio_guider_params=agp(a),images=a.images,max_batch_size=a.max_batch_size)
                gen=time.time()-ts; te=time.time()
                encode_video(video=video,fps=a.frame_rate,audio=audio,output_path=out,video_chunks_number=1); enc=time.time()-te
        total=time.time()-t0; STATS.append(total); LAST[0]=time.time()
        print(f"[worker] CLIP mode={MODE} warm={warm} model_load={_LOAD[0]:.1f} sample+decode={gen:.1f} encode={enc:.1f} total={total:.1f} avg={sum(STATS)/len(STATS):.1f}",flush=True)
        return {"video_b64":base64.b64encode(open(out,"rb").read()).decode(),"timing":{"model_load_s":round(_LOAD[0],1),"sample_decode_s":round(gen,1),"encode_s":round(enc,1),"total_s":round(total,1)},"batch_avg_s":round(sum(STATS)/len(STATS),1),"warm":warm,"mode":MODE}
    except Exception as e:
        return {"error":str(e)[:200],"tb":traceback.format_exc()[-500:]}
    finally:
        BUSY[0]=False; LAST[0]=time.time()
        for f in (img,out):
            try: os.remove(f)
            except: pass
