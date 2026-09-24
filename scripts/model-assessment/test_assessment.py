"""Run with: python -m unittest discover -s scripts/model-assessment -v"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sleap_io as sio

from assess_centroids import crowded_flags, load_predictions, match_points, metrics


class AssessmentTests(unittest.TestCase):
    def test_crowded_flags_use_neighbour_skeleton_not_just_points(self):
        skeleton=sio.Skeleton(['a','b','c'])
        # Animal 0 is a long horizontal line at y=0; animal 1 is a short one 10 px
        # above it between x=20 and 30; animal 2 is far away. Animal 1's target is
        # 27 px from animal 0's nearest node but 10 px from its segment, so it is
        # crowded only if segment distance is used. Animal 0's target (50,0) is
        # 22 px from animal 1's nearest point, so it is not crowded at 15 px.
        animals=[sio.Instance.from_numpy(np.array([[0.,0.],[50.,0.],[100.,0.]]),skeleton),
                 sio.Instance.from_numpy(np.array([[20.,10.],[25.,10.],[30.,10.]]),skeleton),
                 sio.Instance.from_numpy(np.array([[0.,500.],[50.,500.],[100.,500.]]),skeleton)]
        targets=np.array([[50.,0.],[25.,10.],[50.,500.]])
        self.assertEqual(crowded_flags(animals,targets,15).tolist(),[False,True,False])
        self.assertEqual(crowded_flags(animals,targets,25).tolist(),[True,True,False])
        self.assertEqual(crowded_flags(animals,targets,9).tolist(),[False,False,False])

    def test_global_assignment_avoids_greedy_loss(self):
        # GT 0 can take either prediction; GT 1 can only take prediction 0.
        pairs=match_points([[0,0],[3,0]],[[1,0],[-2,0]],2)
        self.assertEqual({(i,j) for i,j,_ in pairs},{(0,1),(1,0)})

    def test_invalid_edge_cannot_displace_valid_match(self):
        pairs=match_points([[0,0],[100,0]],[[0,0],[200,0]],15)
        self.assertEqual([(i,j) for i,j,_ in pairs],[(0,0)])

    def test_duplicates_and_threshold_boundary(self):
        self.assertEqual(len(match_points([[0,0]],[[15,0],[0,0]],15)),1)
        self.assertEqual(len(match_points([[0,0]],[[15,0]],15)),1)
        self.assertEqual(match_points([[0,0]],[[15.001,0]],15),[])

    def test_empty_predictions_and_gt(self):
        self.assertEqual(match_points([[0,0]],[]),[])
        self.assertEqual(match_points([],[[0,0]]),[])
        self.assertIsNone(metrics([{'tp':0,'fp':0,'fn':2,'distance_sum':0}])['precision_pct'])

    def test_repeats_do_not_inflate_counts(self):
        row={'tp':3,'fp':1,'fn':1,'distance_sum':6}
        result=metrics([row]*10,10)
        self.assertEqual(result['tp_mean'],3)
        self.assertEqual(result['recall_pct'],75)
        self.assertEqual(result['mean_error_px'],2)

    def test_missing_prediction_frame_counts_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'predictions.slp'
            skeleton=sio.Skeleton(['centroid'])
            video=sio.Video('missing.mp4',backend=None,backend_metadata={'shape':[2,100,100,1]})
            instance=sio.PredictedInstance.from_numpy(np.array([[10.,20.]]),skeleton,point_scores=np.array([.8]),score=.8)
            labels=sio.Labels([sio.LabeledFrame(video,0,[instance])],videos=[video],skeletons=[skeleton])
            labels.save(str(path))
            predictions,missing=load_predictions(path,2)
            self.assertEqual(missing,[1])
            self.assertEqual(len(predictions[1][0]),0)


if __name__=='__main__':
    unittest.main()
