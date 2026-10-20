"""Cinematic acceptance tests. Provider responses here are mocks, not generation."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import app
from cinematic_engine import (CinematicBrain, MiniMaxH3, ProviderError, ReferenceBudget,
                              choose_model, dependency_report, registry)


class CinematicTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ,{"MINIMAX_ENABLED":"0","MINIMAX_API_KEY":"","LLM_MODEL":""})
        self.env.start();self.addCleanup(self.env.stop)
        self.caps = registry({"video_node":"58"})
        self.h3 = replace(self.caps[0],availability="configured_unverified")

    def test_simple_atmosphere_uses_one_take_not_mandatory_angles(self):
        plan=CinematicBrain(self.caps).plan({"plot":"Fog rests over a quiet lake."},5)
        self.assertEqual(len(plan["shots"]),1)
        self.assertEqual(plan["scenes"][0]["coverage"],["continuous"])
        self.assertEqual(plan["shots"][0]["model_route"]["backend"],"comfyui")

    def test_long_scene_respects_real_duration_and_total_budget(self):
        caps=[self.h3,self.caps[-1]]
        plan=CinematicBrain(caps).plan({"plot":"Fog rests over a quiet lake."},60)
        self.assertEqual(len(plan["shots"]),4)
        self.assertEqual({s["duration_seconds"] for s in plan["shots"]},{15})
        self.assertEqual(len({s["logical_take_id"] for s in plan["shots"]}),1)
        self.assertEqual(sum(s["duration_seconds"] for s in plan["shots"]),60)

    def test_dialogue_coverage_and_new_angles_do_not_chain_composition(self):
        p=CinematicBrain([self.h3]).plan({"plot":"Mara asks Ivo a question in their conversation.",
            "characters":[{"id":"mara","name":"Mara"},{"id":"ivo","name":"Ivo"}]},24)
        self.assertEqual([s["framing"] for s in p["shots"]],["master","over_the_shoulder","reverse_angle","reaction"])
        self.assertTrue(all(s["continuity_policy"]=="new_composition" for s in p["shots"]))
        self.assertEqual(p["shots"][0]["characters_present"],["mara","ivo"])

    def test_action_sequence_has_ordered_coverage(self):
        plan=CinematicBrain([self.h3]).plan({"plot":"Mara runs then hands over a compass."},18)
        self.assertEqual([s["framing"] for s in plan["shots"]],["wide_action","medium_action","reaction"])

    def test_authored_duration_and_story_are_preserved(self):
        source={"plot":"A traveler arrives then leaves.","storyboard":[{"scene_id":"one","shots":[
            {"visual_prompt":"Arrives","duration_seconds":10}, {"visual_prompt":"Leaves","duration_seconds":5}]}]}
        plan=CinematicBrain(self.caps).plan(source,15)
        self.assertEqual([s["visual_prompt"] for s in plan["shots"]],["Arrives","Arrives","Leaves"])
        with self.assertRaises(ValueError):CinematicBrain(self.caps).plan(source,10)

    def test_capability_policy_never_calls_unavailable_h3(self):
        self.assertEqual(choose_model(self.caps)[0].backend,"comfyui")
        with self.assertRaises(ValueError): choose_model(self.caps,"h3")
        self.assertEqual(choose_model([self.h3]+self.caps,"auto")[0].id,"MiniMax-H3")
        self.assertEqual(choose_model([self.h3]+self.caps,"local")[0].backend,"comfyui")

    def test_reference_priority_preserves_critical_character(self):
        cap=replace(self.h3,images=2,combined=2)
        assets=[{"id":str(i),"kind":"image","role":"reference_image","approved":True,"priority":i} for i in range(10)]
        assets[0].update(critical=True,character_id="mara")
        result=ReferenceBudget(cap).select(assets,"reference",["mara"])
        self.assertEqual([a["id"] for a in result["selected"]],["0","9"])
        self.assertEqual(len(result["omitted"]),8)
        for a in assets:a["critical"]=True
        with self.assertRaises(ValueError):ReferenceBudget(cap).select(assets,"reference")

    def test_reference_modes_cannot_mix_and_planning_is_not_submitted(self):
        first={"id":"first","kind":"image","role":"first_frame","approved":True,"critical":True}
        ref={"id":"cast","kind":"image","role":"reference_image","approved":True}
        with self.assertRaises(ValueError):ReferenceBudget(self.h3).select([first,ref],"endpoints")
        plan={"id":"internal","kind":"image","role":"planning","approved":True}
        self.assertEqual(ReferenceBudget(self.h3).select([plan],"reference")["selected"],[])

    def test_reference_duration_budget_and_duplicate_detection(self):
        assets=[{"id":str(i),"kind":"video","role":"reference_video","approved":True,
                 "critical":True,"duration_seconds":8} for i in range(2)]
        with self.assertRaises(ValueError):ReferenceBudget(self.h3).select(assets,"reference")
        assets[1]["id"]=assets[0]["id"]
        self.assertEqual(len(ReferenceBudget(self.h3).select(assets,"reference")["selected"]),1)

    def test_provider_post_ambiguity_is_not_retried(self):
        session=Mock();session.request.side_effect=TimeoutError()
        client=MiniMaxH3(self.h3,session)
        with patch.dict(os.environ,{"MINIMAX_API_KEY":"unit-test-only"}):
            with self.assertRaises(ProviderError) as caught:client.create({"model":"MiniMax-H3"})
        self.assertTrue(caught.exception.ambiguous)
        self.assertEqual(session.request.call_count,1)

    def test_provider_auth_error_is_permanent_and_secret_is_not_logged(self):
        session=Mock();session.request.return_value.status_code=401
        with patch.dict(os.environ,{"MINIMAX_API_KEY":"unit-test-only"}):
            with self.assertRaises(ProviderError) as caught:MiniMaxH3(self.h3,session).verify_access()
        self.assertEqual(str(caught.exception),"authentication_failed")
        self.assertEqual(session.request.call_count,1)

    def test_provider_query_contract_and_success(self):
        session=Mock();response=session.request.return_value;response.status_code=200
        response.json.return_value={"task":{"id":"123","status":"succeeded","duration":5}}
        with patch.dict(os.environ,{"MINIMAX_API_KEY":"unit-test-only"}):
            task=MiniMaxH3(self.h3,session).wait("123",lambda _:None)
        self.assertEqual(task["status"],"succeeded")
        self.assertTrue(session.request.call_args.args[1].endswith('/v2/query/video_generation/123'))

    def test_request_does_not_send_unsupported_seed_or_audio_switch(self):
        c=MiniMaxH3(self.h3,Mock())
        p=c.payload({"visual_prompt":"Clouds move","duration_seconds":7},
                    {"mode":"text","selected":[]},[])
        self.assertEqual(p["model"],"MiniMax-H3")
        self.assertNotIn("seed",p);self.assertNotIn("audio",p)
        with self.assertRaises(ValueError):c.payload({"duration_seconds":3},{"mode":"text","selected":[]},[])

    def test_variable_duration_ledger_uses_actual_shot_seconds(self):
        self.assertEqual(app.movie_seconds([{"duration_seconds":4},{"duration_seconds":11},{}]),20)
        with self.assertRaises(ValueError):app.shot_seconds({"duration_seconds":float('nan')})

    def test_regeneration_archives_target_and_preserves_neighbor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'runs'/'job';folder.mkdir(parents=True)
            first=folder/'shot_0001.mp4';second=folder/'shot_0002.mp4'
            first.write_bytes(b'first');second.write_bytes(b'second')
            studio=app.Studio(root,acquire_lock=False)
            studio.store.data={"job_id":"job","folder":str(folder),"status":"paused", "cloud":{"provider":"None"},
                "shots":[{"shot_id":1,"shot_uid":"a","status":"completed","approval":"approved","visual_prompt":"Old",
                          "duration_seconds":5,"scene_id":"one","camera_movement":"steady","lighting_cue":"day","seed":1},
                         {"shot_id":2,"shot_uid":"b","status":"completed","dependency_ids":["a"]}]}
            with self.assertRaises(ValueError):studio.revise_shot(1,"New",False)
            with patch.object(studio,'launch'):
                studio.revise_shot(1,"New",True)
            self.assertEqual(second.read_bytes(),b'second')
            self.assertEqual(studio.store.data['shots'][1]['status'],'completed')
            self.assertTrue(studio.store.data['shots'][1]['dependency_review_required'])
            self.assertEqual(next((folder/'versions').glob('*/shot_0001.mp4')).read_bytes(),b'first')
            self.assertFalse(studio.store.data['shutdown'])

    def test_authored_shot_cast_excludes_offscreen_characters(self):
        source={"plot":"Mara and Ivo meet.","characters":[{"id":"m","name":"Mara"},{"id":"i","name":"Ivo"}],
            "storyboard":[{"scene_id":"one","shots":[{"visual_prompt":"Mara waits alone","duration_seconds":5},
                {"visual_prompt":"Ivo approaches","duration_seconds":5}]}]}
        p=CinematicBrain(self.caps).plan(source,10)
        self.assertEqual([s['characters_present'] for s in p['shots']],[["m"],["i"]])

    def test_dependency_cycle_or_unbounded_text_is_rejected(self):
        p=CinematicBrain(self.caps).plan({"plot":"A quiet lake."},10)
        p['shots'][0]['dependency_ids']=[p['shots'][1]['shot_uid']]
        with self.assertRaises(ValueError):CinematicBrain.validate(p)
        p['shots'][0]['dependency_ids']=[];p['shots'][0]['lighting_cue']={"bad":"object"}
        with self.assertRaises(ValueError):CinematicBrain.validate(p)

    def test_changed_approved_reference_fails_before_paid_post(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'anchor.png';Image.new('RGB',(256,256)).save(p)
            asset={"id":"a","path":str(p),"kind":"image","role":"reference_image","approved":True,"sha256":"wrong"}
            with self.assertRaises(ValueError):MiniMaxH3(self.h3,Mock()).payload(
                {"visual_prompt":"Walks","duration_seconds":5},{"mode":"reference","selected":[asset]},[Path(tmp)])

    def test_dependency_conflict_is_reported_without_automatic_upgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'requirements.txt';manifest.write_text('requests>=999\n')
            with patch('cinematic_engine.subprocess.run') as pip:
                result=dependency_report(manifest,install=True)
            self.assertFalse(result['ready']);pip.assert_not_called()
            self.assertEqual(result['packages'][0]['state'],'version_conflict')

    def test_missing_dependency_install_failure_is_bounded(self):
        import importlib.metadata
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'requirements.txt';manifest.write_text('requests>=2\n')
            with patch('cinematic_engine.importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError()), \
                 patch('cinematic_engine.subprocess.run',side_effect=subprocess.CalledProcessError(1,'pip')) as pip:
                result=dependency_report(manifest,install=True)
            self.assertFalse(result['ready']);self.assertEqual(pip.call_count,1)
            self.assertEqual(result['packages'][0]['state'],'installation_failed')

    def test_h3_auxiliary_stage_does_not_replay_ambiguous_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            studio=app.Studio(Path(tmp),acquire_lock=False)
            studio.store.data={'shots':[]}
            shot={};provider=Mock();provider.create.side_effect=ProviderError('submission_unknown',ambiguous=True)
            with self.assertRaises(ProviderError):studio.h3_stage(shot,provider,'context_ir','/v2/h3_context_ir',{})
            with self.assertRaises(ProviderError):studio.h3_stage(shot,provider,'context_ir','/v2/h3_context_ir',{})
            self.assertEqual(provider.create.call_count,1)

    def test_narrative_refinement_preserves_ids_duration_and_dialogue(self):
        response=Mock();response.json.return_value={'choices':[{'message':{'content':json.dumps({'shots':[
            {'shot_id':1,'visual_prompt':'Mara turns toward the hatch','dialogue':'We have arrived.','characters_present':['m']}]})}}]}
        with patch.dict(os.environ,{'LLM_MODEL':'local-test'}),patch('requests.post',return_value=response):
            plan=CinematicBrain(self.caps).plan({'plot':'Mara arrives.','characters':[{'id':'m','name':'Mara'}]},5)
        self.assertEqual(plan['planner'],'structural_coverage_with_configured_llm_direction')
        self.assertEqual(plan['shots'][0]['dialogue'],'We have arrived.')
        self.assertEqual(plan['shots'][0]['duration_seconds'],5)

    def test_invalid_llm_result_retains_complete_structural_plan(self):
        response=Mock();response.json.return_value={'choices':[{'message':{'content':'{"shots":[]}'}}]}
        with patch.dict(os.environ,{'LLM_MODEL':'local-test','LLM_FAILURE_POLICY':'structural'}),patch('requests.post',return_value=response):
            plan=CinematicBrain(self.caps).plan({'plot':'A quiet lake.'},10)
        self.assertEqual(plan['planner'],'deterministic_structural_director')
        self.assertEqual(len(plan['shots']),2)

    def test_rate_limit_budget_and_invalid_post_body(self):
        session=Mock();session.request.return_value.status_code=429
        with patch.dict(os.environ,{'MINIMAX_API_KEY':'test'}),patch('cinematic_engine.time.sleep'):
            with self.assertRaises(ProviderError):MiniMaxH3(self.h3,session).create({})
        self.assertEqual(session.request.call_count,3)
        session.reset_mock();session.request.return_value.status_code=200
        session.request.return_value.json.side_effect=ValueError()
        with patch.dict(os.environ,{'MINIMAX_API_KEY':'test'}):
            with self.assertRaises(ProviderError) as caught:MiniMaxH3(self.h3,session).create({})
        self.assertTrue(caught.exception.ambiguous);self.assertEqual(session.request.call_count,1)

    def test_endpoint_payload_uses_separate_approved_images(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);assets=[]
            for role,color in [('first_frame','red'),('last_frame','blue')]:
                path=root/(role+'.png');Image.new('RGB',(256,256),color).save(path)
                assets.append({'id':role,'path':str(path),'kind':'image','role':role,'approved':True,'critical':True,'sha256':app.digest(path)})
            payload=MiniMaxH3(self.h3,Mock()).payload({'duration_seconds':5,'visual_prompt':'Crosses the room'},
                {'mode':'endpoints','selected':assets},[root])
            self.assertEqual([a['role'] for a in payload['content'][1:]],['first_frame','last_frame'])
            self.assertEqual(payload['ratio'],'adaptive')
            self.assertNotEqual(payload['content'][1]['image_url'],payload['content'][2]['image_url'])

    def test_optional_missing_dependency_preserves_local_generation(self):
        import importlib.metadata
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'requirements.txt';p.write_text('google-auth-oauthlib>=1.2,<2\n')
            with patch('cinematic_engine.importlib.metadata.version',side_effect=importlib.metadata.PackageNotFoundError()):
                report=dependency_report(p)
            self.assertTrue(report['ready']);self.assertFalse(report['packages'][0]['essential'])

    def test_h3_max_has_own_duration_resolution_and_explicit_route(self):
        maximum=replace(self.caps[1],availability='configured_unverified')
        selected,_=choose_model([self.h3,maximum],'h3-max')
        self.assertEqual(selected.id,'MiniMax-H3-Max')
        self.assertNotIn(4,selected.durations);self.assertNotIn('2K',selected.resolutions)
        with self.assertRaises(ValueError):choose_model(self.caps,'h3-max')


if __name__=='__main__':unittest.main(verbosity=2)
