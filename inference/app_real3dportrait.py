import os, sys
sys.path.append('./')
import argparse
import gradio as gr
from inference.real3d_infer import GeneFace2Infer
from utils.commons.hparams import hparams

class Inferer(GeneFace2Infer):
    def infer_once_args(self, *args, **kargs):
        assert len(kargs) == 0
        keys = [
            'src_image_name',
            'drv_audio_name',
            'drv_pose_name',
            'bg_image_name',
            'blink_mode',
            'temperature',
            'mouth_amp',
            'out_mode',
            'map_to_init_pose',
            'low_memory_usage',
            'hold_eye_opened',
            'a2m_ckpt',
            'head_ckpt',
            'torso_ckpt',
            'min_face_area_percent',
        ]
        inp = {}
        out_name = None
        info = ""
        
        try: # try to catch errors and jump to return 
            for key_index in range(len(keys)):
                key = keys[key_index]
                inp[key] = args[key_index]
                if '_name' in key:
                    inp[key] = inp[key] if inp[key] is not None else ''
            
            if inp['src_image_name'] == '':
                info = "Input Error: Source image is REQUIRED!"
                raise ValueError
            if inp['drv_audio_name'] == '' and inp['drv_pose_name'] == '':
                info = "Input Error: At least one of driving audio or video is REQUIRED!"
                raise ValueError


            if inp['drv_audio_name'] == '' and inp['drv_pose_name'] != '':
                inp['drv_audio_name'] = inp['drv_pose_name']
                print("No audio input, we use driving pose video for video driving")
                
            if inp['drv_pose_name'] == '':
                inp['drv_pose_name'] = 'static'    
            
            reload_flag = False
            if inp['a2m_ckpt'] != self.audio2secc_dir:
                print("Changes of a2m_ckpt detected, reloading model")
                reload_flag = True
            if inp['head_ckpt'] != self.head_model_dir:
                print("Changes of head_ckpt detected, reloading model")
                reload_flag = True
            if inp['torso_ckpt'] != self.torso_model_dir:
                print("Changes of torso_ckpt detected, reloading model")
                reload_flag = True

            inp['out_name'] = ''
            inp['seed'] = 42
            
            print(f"infer inputs : {inp}")
                
            try:
                if reload_flag:
                    self.__init__(inp['a2m_ckpt'], inp['head_ckpt'], inp['torso_ckpt'], inp=inp, device=self.device)
            except Exception as e:
                content = f"{e}"
                info = f"Reload ERROR: {content}"
                raise ValueError
            try:
                out_name = self.infer_once(inp)
            except Exception as e:
                content = f"{e}"
                info = f"Inference ERROR: {content}"
                raise ValueError
        except Exception as e:
            if info == "": # unexpected errors
                content = f"{e}"
                info = f"WebUI ERROR: {content}"
        
        # output part
        if len(info) > 0 : # there is errors    
            print(info)
            info_gr = gr.update(visible=True, value=info)
        else: # no errors
            info_gr = gr.update(visible=False, value=info)
        if out_name is not None and len(out_name) > 0 and os.path.exists(out_name): # good output
            print(f"Succefully generated in {out_name}")
            video_gr = gr.update(visible=True, value=out_name)
        else:
            print(f"Failed to generate")
            video_gr = gr.update(visible=True, value=out_name)
            
        return video_gr, info_gr

def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
    
def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)

def real3dportrait_demo(
    audio2secc_dir,
    head_model_dir,
    torso_model_dir, 
    device          = 'cuda',
    warpfn          = None,
    ):

    sep_line = "-" * 40

    infer_obj = Inferer(
        audio2secc_dir=audio2secc_dir, 
        head_model_dir=head_model_dir,
        torso_model_dir=torso_model_dir,
        device=device,
    )

    print(sep_line)
    print("Model loading is finished.")
    print(sep_line)
    with gr.Blocks(analytics_enabled=False) as real3dportrait_interface:
        gr.Markdown("\
            <div align='center'> <h2> Real3D-Portrait: One-shot Realistic 3D Talking Portrait Synthesis (ICLR 2024 Spotlight) </span> </h2> \
            <a style='font-size:18px;color: #a0a0a0' href='https://arxiv.org/pdf/2401.08503.pdf'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            <a style='font-size:18px;color: #a0a0a0' href='https://real3dportrait.github.io/'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
            <a style='font-size:18px;color: #a0a0a0' href='https://github.com/yerfor/Real3DPortrait/'> Github </div>")
        
        sources = None
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            src_image_name = gr.Image(label="Source image (required)", sources=sources, type="filepath", value="data/raw/examples/Macron.png")
                with gr.Tabs(elem_id="driven_audio"):
                    with gr.TabItem('Upload audio'):
                        with gr.Column(variant='panel'):
                            drv_audio_name = gr.Audio(label="Input audio (required for audio-driven)", sources=sources, type="filepath", value="data/raw/examples/Obama_5s.wav")
                with gr.Tabs(elem_id="driven_pose"):
                    with gr.TabItem('Upload video'):
                        with gr.Column(variant='panel'):
                            drv_pose_name = gr.Video(label="Driven Pose (required for video-driven, optional for audio-driven)", sources=sources, value="data/raw/examples/May_5s.mp4")
                with gr.Tabs(elem_id="bg_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            bg_image_name = gr.Image(label="Background image (optional)", sources=sources, type="filepath", value="data/raw/examples/bg.png")

                             
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('General Settings'):
                        with gr.Column(variant='panel'):

                            blink_mode = gr.Radio(['none', 'period'], value='period', label='blink mode', info="whether to blink periodly") #        
                            min_face_area_percent = gr.Slider(minimum=0.15, maximum=0.5, step=0.01, label="min_face_area_percent",  value=0.2, info='The minimum face area percent in the output frame, to prevent bad cases caused by a too small face.',)
                            temperature = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="temperature",  value=0.2, info='audio to secc temperature',)
                            mouth_amp = gr.Slider(minimum=0.0, maximum=1.0, step=0.025, label="mouth amplitude",  value=0.45, info='higher -> mouth will open wider, default to be 0.4',)
                            out_mode = gr.Radio(['final', 'concat_debug'], value='concat_debug', label='output layout', info="final: only final output ; concat_debug: final output concated with internel features") 
                            low_memory_usage = gr.Checkbox(label="Low Memory Usage Mode: save memory at the expense of lower inference speed. Useful when running a low audio (minutes-long).", value=False)
                            map_to_init_pose = gr.Checkbox(label="Whether to map pose of first frame to initial pose", value=True)
                            hold_eye_opened  = gr.Checkbox(label="Whether to maintain eyes always open")
                                
                            submit = gr.Button('Generate', elem_id="generate", variant='primary')
                        
                    with gr.Tabs(elem_id="genearted_video"):
                        info_box = gr.Textbox(label="Error", interactive=False, visible=False)
                        gen_video = gr.Video(label="Generated video", format="mp4", visible=True)
            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="checkbox"):
                    with gr.TabItem('Checkpoints'):
                        with gr.Column(variant='panel'):
                            ckpt_info_box = gr.Textbox(value="Please select \"ckpt\" under the checkpoint folder ", interactive=False, visible=True, show_label=False)
                            audio2secc_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=audio2secc_dir, file_count='single', label='audio2secc model ckpt path or directory')
                            head_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=head_model_dir, file_count='single', label='head model ckpt path or directory (will be ignored if torso model is set)')
                            torso_model_dir = gr.FileExplorer(glob="checkpoints/**/*.ckpt", value=torso_model_dir, file_count='single', label='torso model ckpt path or directory')
                            # audio2secc_dir = gr.Textbox(audio2secc_dir, max_lines=1, label='audio2secc model ckpt path or directory (will be ignored if torso model is set)')
                            # head_model_dir = gr.Textbox(head_model_dir, max_lines=1, label='head model ckpt path or directory (will be ignored if torso model is set)')
                            # torso_model_dir = gr.Textbox(torso_model_dir, max_lines=1, label='torso model ckpt path or directory')


        fn = infer_obj.infer_once_args
        if warpfn:
            fn = warpfn(fn)
        submit.click(
                    fn=fn, 
                    inputs=[
                        src_image_name, 
                        drv_audio_name,
                        drv_pose_name,
                        bg_image_name,
                        blink_mode,
                        temperature,
                        mouth_amp,
                        out_mode,
                        map_to_init_pose,
                        low_memory_usage,
                        hold_eye_opened,
                        audio2secc_dir,
                        head_model_dir,
                        torso_model_dir,
                        min_face_area_percent,
                    ], 
                    outputs=[
                        gen_video,
                        info_box,
                    ],
                    )

    print(sep_line)
    print("Gradio page is constructed.")
    print(sep_line)

    return real3dportrait_interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", type=str, default='checkpoints/240210_real3dportrait_orig/audio2secc_vae/model_ckpt_steps_400000.ckpt')
    parser.add_argument("--head_ckpt", type=str, default='')
    parser.add_argument("--torso_ckpt", type=str, default='checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig/model_ckpt_steps_100000.ckpt') 
    parser.add_argument("--port", type=int, default=None) 
    parser.add_argument("--server", type=str, default='127.0.0.1')
    parser.add_argument("--share", action='store_true', dest='share', help='share srever to Internet')

    args = parser.parse_args()
    demo = real3dportrait_demo(
        audio2secc_dir=args.a2m_ckpt,
        head_model_dir=args.head_ckpt,
        torso_model_dir=args.torso_ckpt,
        device='cuda:0',
        warpfn=None,
    )
    demo.queue()
    demo.launch(share=args.share, server_name=args.server, server_port=args.port)
