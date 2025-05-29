import pytest
from app.services.services.clipExtractor import calculate_virality_score, VIRALITY_CONFIG

# Mock the logger if calculate_virality_score uses it (currently it doesn't directly, but good practice if it might)
@pytest.fixture
def mock_logger(mocker):
    return mocker.patch('app.services.services.clipExtractor.log')

@pytest.fixture
def default_analysis_data():
    return {
        "text_analysis": {
            'sentiment': {'label': 'neutral', 'score': 0.5},
            'emotions': [],
            'keywords': [],
            'summary': "Neutral summary."
        },
        "visual_analysis": {
            'key_objects': [],
            'facial_expressions': [],
            'frame_dimensions': {'width': 1920, 'height': 1080}, # Default typical HD
            'visual_intensity_score': 0.5,
            'notes': "Default visual analysis."
        },
        "audio_rms": 0.1,
        "motion_score": 0.1
    }

def test_calculate_virality_score_baseline(default_analysis_data):
    score = calculate_virality_score(
        default_analysis_data["text_analysis"],
        default_analysis_data["visual_analysis"],
        default_analysis_data["audio_rms"],
        default_analysis_data["motion_score"],
        VIRALITY_CONFIG
    )
    # This baseline depends heavily on default weights and lack of specific triggers
    # For example, neutral sentiment, no strong emotions, no keywords, default visual intensity
    # Expected score might be low. Let's check it's a float.
    assert isinstance(score, float)
    # A more specific expected value would require manual calculation based on VIRALITY_CONFIG:
    # E.g. visual_intensity (0.5 * 1.0) + audio_rms (0.1 * 0.5) + motion_score (0.1 * 0.3)
    # + vertical_cropability (0 if no objects)
    # = 0.5 + 0.05 + 0.03 = 0.58
    assert score == pytest.approx(0.58, abs=0.01)


def test_positive_sentiment_increases_score(default_analysis_data):
    default_analysis_data["text_analysis"]['sentiment'] = {'label': 'positive', 'score': 0.9}
    score = calculate_virality_score(
        default_analysis_data["text_analysis"],
        default_analysis_data["visual_analysis"],
        default_analysis_data["audio_rms"],
        default_analysis_data["motion_score"],
        VIRALITY_CONFIG
    )
    # Expected: 0.9 * 1.2 (positive_sentiment_weight) + previous_visual/audio/motion (0.58)
    # = 1.08 + 0.5 = 1.58 (visual intensity is 0.5 * 1.0 = 0.5)
    # Visual (0.5*1.0) + Audio (0.1*0.5) + Motion (0.1*0.3) = 0.5 + 0.05 + 0.03 = 0.58
    # Sentiment (0.9 * 1.2) = 1.08
    # Total = 0.58 + 1.08 = 1.66
    assert score > 0.58 
    assert score == pytest.approx(1.66, abs=0.01)


def test_keyword_match_increases_score(default_analysis_data):
    default_analysis_data["text_analysis"]['keywords'] = ['challenge', 'hack'] # Matches VIRALITY_CONFIG
    score = calculate_virality_score(
        default_analysis_data["text_analysis"],
        default_analysis_data["visual_analysis"],
        default_analysis_data["audio_rms"],
        default_analysis_data["motion_score"],
        VIRALITY_CONFIG
    )
    # Expected: Baseline (0.58) + 2 * 1.5 (keyword_match_weight for 2 keywords) = 0.58 + 3.0 = 3.58
    assert score > 0.58
    assert score == pytest.approx(0.58 + 2 * VIRALITY_CONFIG['weights']['keyword_match'], abs=0.01)


def test_cropability_score_person_centered(default_analysis_data):
    default_analysis_data["visual_analysis"]['key_objects'] = [
        {'class': 'person', 'confidence': 0.8, 'bbox': [860, 100, 1060, 980]} # 200px wide, centered in 1920
    ]
    default_analysis_data["visual_analysis"]['frame_dimensions'] = {'width': 1920, 'height': 1080}
    
    # Expected crop width for 1920x1080 (16:9) -> 9:16 crop: 1080 * (9/16) = 607.5
    # Subject center_x = (860+1060)/2 = 960.
    # Crop_x1 = 960 - 607.5/2 = 960 - 303.75 = 656.25
    # Crop_x2 = 960 + 303.75 = 1263.75
    # Overlap: max(860, 656.25) to min(1060, 1263.75) => 860 to 1060. Overlap width = 200.
    # Subject width = 200. Coverage = 200/200 = 1.0.
    # Cropability score contribution: 1.0 * 1.5 (weight) = 1.5
    # Expected total: Baseline (0.58) + 1.5 = 2.08
    
    score = calculate_virality_score(
        default_analysis_data["text_analysis"],
        default_analysis_data["visual_analysis"],
        default_analysis_data["audio_rms"],
        default_analysis_data["motion_score"],
        VIRALITY_CONFIG
    )
    assert score == pytest.approx(0.58 + 1.5, abs=0.01)

def test_cropability_score_person_off_center(default_analysis_data):
    default_analysis_data["visual_analysis"]['key_objects'] = [
        {'class': 'person', 'confidence': 0.8, 'bbox': [100, 100, 300, 980]} # 200px wide, off-center
    ]
    default_analysis_data["visual_analysis"]['frame_dimensions'] = {'width': 1920, 'height': 1080}
    
    # Subject center_x = (100+300)/2 = 200.
    # Crop_x1 = 200 - 303.75 = -103.75 (so 0)
    # Crop_x2 = 200 + 303.75 = 503.75
    # Overlap: max(100, 0) to min(300, 503.75) => 100 to 300. Overlap width = 200.
    # Subject width = 200. Coverage = 1.0. This is because the subject is small enough to fit.
    # Let's make the subject wider or crop narrower to test partial coverage.
    # Crop width = 1080 * (9/16) = 607.5.
    # If subject is [0, 0, 800, 1080]. Center_x = 400.
    # Crop_x1 = 400 - 303.75 = 96.25. Crop_x2 = 400 + 303.75 = 703.75
    # Overlap: max(0, 96.25) to min(800, 703.75) => 96.25 to 703.75. Overlap width = 607.5
    # Subject width = 800. Coverage = 607.5 / 800 = 0.759375
    # Cropability score: 0.759375 * 1.5 = 1.139
    
    default_analysis_data["visual_analysis"]['key_objects'] = [
        {'class': 'person', 'confidence': 0.8, 'bbox': [0, 0, 800, 1080]} 
    ]
    score = calculate_virality_score(
        default_analysis_data["text_analysis"],
        default_analysis_data["visual_analysis"],
        default_analysis_data["audio_rms"],
        default_analysis_data["motion_score"],
        VIRALITY_CONFIG
    )
    expected_crop_score_contribution = (607.5 / 800.0) * VIRALITY_CONFIG['weights']['vertical_cropability']
    assert score == pytest.approx(0.58 + expected_crop_score_contribution, abs=0.01)


def test_missing_frame_dimensions_for_cropability(default_analysis_data, mock_logger):
    default_analysis_data["visual_analysis"]['key_objects'] = [
        {'class': 'person', 'confidence': 0.8, 'bbox': [860, 100, 1060, 980]}
    ]
    default_analysis_data["visual_analysis"].pop('frame_dimensions') # Remove frame dimensions
    
    score = calculate_virality_score(
        default_analysis_data["text_analysis"],
        default_analysis_data["visual_analysis"],
        default_analysis_data["audio_rms"],
        default_analysis_data["motion_score"],
        VIRALITY_CONFIG
    )
    # Should be baseline as cropability score is 0
    assert score == pytest.approx(0.58, abs=0.01)
    # Check if warning was logged (if direct logging is added to calculate_virality_score)
    # For now, assume it handles it gracefully by not adding to score.
    # If using the global `log` object in `calculate_virality_score`:
    # mock_logger.warning.assert_any_call("Frame dimensions not available for cropability score calculation.")


def test_no_key_objects_for_cropability(default_analysis_data):
    default_analysis_data["visual_analysis"]['key_objects'] = [] # No objects
    score = calculate_virality_score(
        default_analysis_data["text_analysis"],
        default_analysis_data["visual_analysis"],
        default_analysis_data["audio_rms"],
        default_analysis_data["motion_score"],
        VIRALITY_CONFIG
    )
    # Should be baseline as cropability score is 0
    assert score == pytest.approx(0.58, abs=0.01)
