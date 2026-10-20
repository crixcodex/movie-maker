"""Real FFmpeg acceptance; ComfyUI generation is mocked, H3 is never called.

Run on the existing media host: python -m unittest test_media_acceptance -v
These tests use only private temporary folders, never live checkpoints/outputs.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import app


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg/ffprobe required')
class MediaAcceptance(unittest.TestCase):
    def source(self, target: Path, seconds: int) -> None:
        app.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
            f'testsrc2=size=320x180:rate=24:duration={seconds}',
            '-f','lavfi','-i',f'sine=frequency=440:sample_rate=32000:duration={seconds}',
            '-c:v','libx264','-threads','2','-preset','ultrafast','-c:a','aac','-ac','2',str(target)])

    def test_variable_duration_native_audio_real_mux(self):
        with tempfile.TemporaryDirectory(prefix='studio-variable-test-') as temporary:
            root=Path(temporary)
            media={'width':320,'height':180,'fps':24,'postproduction':app.postproduction_config()}
            shots=[]
            for index,seconds in enumerate((4,11),1):
                source=root/f'raw_{index:04d}.mp4';self.source(source,seconds)
                shot={'shot_id':index,'duration_seconds':seconds,'seed':1,'visual_prompt':'Native audio source',
                      'native_audio':True,'status':'rendered'}
                details=app.normalize(source,root/f'shot_{index:04d}.mp4',media,shot)
                self.assertEqual(details['audio_plan']['source'],'provider_native')
                shot.update(status='completed',**details);shots.append(shot)
            studio=app.Studio(root,acquire_lock=False)
            studio.store.data={'job_id':'synthetic-media','folder':str(root),'media':media,'shots':shots,'master_shot_ids':[]}
            studio.assemble()
            metadata=app.validate_finished_media(root/'current_movie_stream.mp4',15,media)
            self.assertEqual(next(s for s in metadata['streams'] if s['codec_type']=='audio')['sample_rate'],'48000')
            app.run(['ffmpeg','-v','error','-xerror','-i',str(root/'current_movie_stream.mp4'),'-f','null','-'])

    def test_orchestrated_pipeline_plan_to_verified_local_master(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix='studio-pipeline-test-') as temporary:
            root=Path(temporary);source=root/'comfy-source.mp4';self.source(source,5)
            reference=root/'reference.png';Image.new('RGB',(256,256),'navy').save(reference)
            client=Mock();client.upload.return_value='approved-anchor.png'
            client.reconcile.return_value=None;client.submit.return_value='test-comfy-id'
            client.wait.return_value=source
            config={'video_node':'9','output_dir':str(root),'consistency':{}}
            studio=app.Studio(root,acquire_lock=False)
            story=json.dumps({'plot':'Mara watches the fog over the harbor.',
                'characters':[{'id':'mara','name':'Mara','appearance':'dark short hair','wardrobe':'navy coat'}],
                'render_settings':{'width':320,'height':180}})
            with patch.dict(os.environ,{'LLM_MODEL':'','MINIMAX_ENABLED':'0'}), \
                 patch.object(app,'workflow_config',return_value=({},config)), \
                 patch.object(app,'ComfyClient',return_value=client), \
                 patch.object(studio,'launch'),patch.object(app.os,'system') as shell:
                studio.start(story,str(reference),5/60,'None','','',shutdown=False,
                             orchestration={'enabled':True,'policy':'local'})
                studio.pipeline()
                shell.assert_not_called()
            state=studio.store.data
            self.assertEqual(state['status'],'complete','\n'.join(studio.lines))
            self.assertTrue(state['movie_ready']);self.assertEqual(state['shot_count'],1)
            self.assertTrue(state['quality_report']['decoded_without_errors'])
            self.assertEqual(state['quality_report']['export_dimensions'],[320,134])
            self.assertEqual(state['quality_report']['visual_identity'],'not_measured')
            shot=state['shots'][0]
            self.assertEqual(shot['reference_manifest']['selected'][0]['character_id'],'mara')
            self.assertIn('Locked continuity constraints',shot['compiled_prompt'])
            self.assertEqual(shot['status'],'completed')
            self.assertTrue((Path(state['folder'])/'production_plan.json').is_file())
            self.assertTrue((Path(state['folder'])/'quality_report.json').is_file())
            self.assertEqual(app.digest(Path(state['final_file'])),state['final_sha256'])
            client.submit.assert_called_once()


if __name__=='__main__':unittest.main(verbosity=2)
