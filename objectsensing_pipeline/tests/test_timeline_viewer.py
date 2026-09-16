import json
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import timeline_viewer  # noqa: E402
import viz  # noqa: E402

pytestmark = pytest.mark.smoke


def test_timeline_viewer_installs_shared_form_primitives():
    source = open(timeline_viewer.__file__, encoding="utf-8").read()

    assert "installSharedPrimitives();" in source


def test_timeline_viewer_embeds_trace_and_ply():
    tmp = tempfile.mkdtemp()
    summary_path = os.path.join(tmp, "office.json")
    progress_path = os.path.join(tmp, "office_progress.json")
    ply_path = os.path.join(tmp, "office_viz.ply")
    html_path = os.path.join(tmp, "office_viz.html")

    summary = [{
        "model": "chair_demo",
        "category": "chair",
        "status": "selected",
        "coverage": 0.8,
        "cov_reverse": 0.7,
    }]
    progress = {
        "database_summary": {"models": 2, "categories": {"chair": 1, "table": 1}},
        "run_config": {"coverage": 0.45, "fusion_backend": "cpu"},
        "fusion_trace": {
            "backend": "cpu",
            "total_frames": 2,
            "frames": [
                {"frame_index": 1, "points": [[0, 0, 0]], "colors": [[255, 0, 0]], "camera_position": [0, 0, 0]},
                {"frame_index": 2, "points": [[1, 0, 0]], "colors": [[0, 255, 0]], "camera_position": [0.1, 0, 0]},
            ],
        },
        "execution_trace": [{
            "frame_count": 20,
            "full": {
                "scan": {
                    "points": 3,
                    "keypoints": [{"position": [0.0, 0.5, 0.0], "response": 0.9}],
                },
                "preselection": {
                    "pool": ["chair_demo", "table_demo"],
                    "top_k": ["chair_demo", "table_demo"],
                },
                "matching": [{
                    "model": "chair_demo",
                    "category": "chair",
                    "status": "candidate",
                    "model_preview": [[0, 0, 0], [0, 1, 0]],
                    "correspondences": 12,
                    "constellations": 3,
                    "correspondence_preview": [{
                        "model_keypoint": 0,
                        "scan_keypoint": 0,
                        "model_position": [0, 0, 0],
                        "scan_position": [0, 0.5, 0],
                        "theta_deg": 0,
                        "scale": 1,
                        "translation": [0, 0.5, 0],
                    }],
                    "constellation_preview": [{
                        "quality": 2.0,
                        "inliers": 1,
                        "theta_deg": 0,
                        "scale": 1,
                        "translation": [0, 0.5, 0],
                        "inlier_pairs": [{
                            "model_keypoint": 0,
                            "scan_keypoint": 0,
                            "model_position": [0, 0, 0],
                            "scan_position": [0, 0.5, 0],
                        }],
                    }],
                }],
                "checkpoint_selection": [{
                    "model": "chair_demo", "status": "selected",
                }],
            },
        }],
        "final_revalidation": [{
            "model": "chair_demo", "status": "kept", "coverage": 0.8,
        }],
        "final_selection": [
            {
                "model": "chair_demo",
                "status": "selected",
                "translation": [0.0, 0.5, 0.0],
                "theta_deg": 12.0,
                "scale": 1.1,
            },
            {
                "model": "table_demo",
                "status": "rejected_min_symmetric_score",
                "translation": [1.0, 0.0, 0.0],
                "theta_deg": 0.0,
                "scale": 1.0,
            },
        ],
    }
    with open(summary_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream)
    with open(progress_path, "w", encoding="utf-8") as stream:
        json.dump(progress, stream)
    viz.write_points_ply(
        ply_path,
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.full((3, 3), 150, dtype=np.uint8),
    )

    output = timeline_viewer.build_timeline_html(
        summary_path, ply_path, html_path, progress_json=progress_path,
    )

    assert output == html_path
    html = open(html_path, encoding="utf-8").read()
    assert "window.OBJECTSENSING_RUN=" in html
    assert '"scan_point_count":3' in html
    assert "chair_demo" in html
    assert "table_demo" in html
    assert "rejected_min_symmetric_score" in html
    assert "Top-k" in html
    assert "fusion-frame" in html
    assert "constellation" in html
    assert "major-next" in html
    assert 'id="scope-final"' in html
    assert 'id="clear-final-selection"' in html
    assert 'id="show-all-final"' in html
    assert 'id="focus-final-selection"' in html
    assert 'id="candidate-search"' in html
    assert 'id="candidate-category"' in html
    assert 'id="candidate-status"' in html
    assert 'id="viewer-tooltip"' in html
    assert 'href="/" aria-label="Back to Run Console"' in html
    assert 'id="stage-tabs"' not in html
    assert 'id="current-step-title"' in html
    assert 'id="timeline-chapters"' in html
    assert 'id="viewer-event-track"' in html
    assert 'id="viewer-inspector-resizer"' in html
    assert 'id="viewer-timeline-resizer"' in html
    assert 'id="toggle-viewer-inspector"' in html
    assert 'aria-label="Resize inspector"' in html
    assert 'aria-label="Resize timeline"' in html
    assert "function bindViewerResizers()" in html
    assert "objectsensing-viewer-inspector-width" in html
    assert "objectsensing-viewer-timeline-height" in html
    assert 'id="candidate-timeline-navigation"' in html
    assert 'id="inspector-scroll"' in html
    assert "buildKeypointConstellationContext" in html
    assert "candidateRelationForScanIndices" in html
    assert "selectedConstellationContext" in html
    assert 'id="clear-relation-filter"' in html
    assert 'Candidates linked to C' in html
    assert "candidateDetailHeading" in html
    assert "tracked context" in html
    assert "applyCandidateFilters" in html
    assert "createCandidateFilterController" in html
    assert "candidateTooltip" in html
    assert "candidateTimelineGroups" in html
    assert "renderCandidateTimelineNavigation" in html
    assert "candidate-related" in html
    assert "selectedFinalKeys" in html
    assert "activateTimelineScope" in html
    assert "toggle-all-constellations" in html
    assert "syncAllConstellations" in html
    assert "constellationPositions" in html
    assert "aria-label=\"3D layer legend\"" in html
    assert "syncFinalCandidatePreviews" in html
    assert "function finalModelKeypoints(detail)" in html
    assert "model_keypoint_preview" in html
    assert "final-model-keypoints" in html
    assert "markers.material.sizeAttenuation=false" in html
    assert "object.add(markers)" in html
    assert "height: 108px" in html
    assert "function focusFinalSelection()" in html
    assert 'container.dataset.sceneReference="scan-rgbd"' in html
    assert "fitCameraToBox(scanBounds" in html
    assert "scanPointCount" in html
    assert "SCENE3D_THEME.pointSize.candidate" in html
    assert "createSceneViewport" in html
    assert "bindResizableDimension" in html
    assert 'bindMultiSelectListInteractions({container:"#candidate-list"' in html
    assert 'createTooltipController({selector:"#viewer-tooltip"' in html
    assert "bindHoverPreviewPanels" in html
    assert "inspector-collapsed" in html
    assert "candidate-thumb" in html
    assert "--ui-accent: #65c7d6" in html
    assert "contain: inline-size" in html
    assert "overflow-x: hidden" in html
    assert ".inspector-scroll" in html
    assert "overflow-y: auto" in html
    assert ".relation-context" in html
    assert ".candidate-detail-heading" in html
    assert "bindResizableAccordions" in html
    assert html.count('<details class="ui-accordion"') == 4
    assert 'data-panel="candidates"' in html
    assert 'property:"--accordion-height"' in html
    assert "contain: inline-size" in html
    assert "__OBJECTSENSING_SHARED_CSS__" not in html
    assert "__OBJECTSENSING_VIEWER_CSS__" not in html
    assert "__OBJECTSENSING_LUCIDE__" not in html
    assert "__OBJECTSENSING_TIMELINE_COMPONENTS__" not in html
    assert "__OBJECTSENSING_PAYLOAD__" not in html
