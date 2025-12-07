from audio_features import mel_spectrogram, mfcc, forward_fill, backward_fill, zero_crossing_rate
import torch

# idk how to compute mel_spectrogram or mfcc expected values manually,
# so just testing the fill functions and zcr here

def test_fills():
    t = torch.Tensor([[[1, 4, 0, 0, 2, 0, 7],
                       [6, 4, 13, 3, 9, 10, 8]],

                      [[0, 0, 0, 0, 0, 0, 0],
                       [0, 12, 0, 5, 0, 0, 0]]])

    expected_forward = torch.Tensor([[[1, 4, 4, 4, 2, 2, 7],
                                      [6, 4, 13, 3, 9, 10, 8]],

                                     [[0, 0, 0, 0, 0, 0, 0],
                                      [0, 12, 12, 5, 5, 5, 5]]])
    expected_backward = torch.Tensor([[[1, 4, 2, 2, 2, 7, 7],
                                       [6, 4, 13, 3, 9, 10, 8]],

                                      [[0, 0, 0, 0, 0, 0, 0],
                                       [12, 12, 5, 5, 0, 0, 0]]])

    assert torch.equal(forward_fill(t), expected_forward), "Forward fill did not match expected output."
    assert torch.equal(backward_fill(t), expected_backward), "Backward fill did not match expected output."

    t_nan = t.where(t != 0, float('nan'))

    nan_forward = forward_fill(t_nan, to_fill=float('nan'))
    nan_backward = backward_fill(t_nan, to_fill=float('nan'))

    # nan values cannot be directly compared using torch.equal due to NaN != NaN
    # so we'll replace them with 0s and check them with the old expected values
    alt_forward = nan_forward.where(torch.isnan(nan_forward).logical_not(), 0.0)
    alt_backward = nan_backward.where(torch.isnan(nan_backward).logical_not(), 0.0)

    assert torch.equal(alt_forward, expected_forward), "Forward fill with NaNs did not match expected output."
    assert torch.equal(alt_backward, expected_backward), "Backward fill with NaNs did not match expected output."

def test_zcr():
    # test zero handling options
    a = torch.Tensor([1, 0, -1, 0, -1, 0, 1, -1])

    expected = {
        'forward_fill': torch.Tensor([[0.0, 0.25, 0.0, 0.5, 0.25]]),
        'backward_fill': torch.Tensor([[0.25, 0.25, 0.25, 0.5, 0.25]]),
        'positive': torch.Tensor([[0.0, 0.5, 0.75, 0.5, 0.25]]),
        'negative': torch.Tensor([[0.25, 0.25, 0.0, 0.5, 0.25]]),
        'unsigned': torch.Tensor([[0, 0, 0, 0.25, 0.25]]),
    }

    for zero_handling, exp in expected.items():
        zcr = zero_crossing_rate(a, frame_length=4, hop_length=2, threshold=0.0, zero_handling=zero_handling)
        assert torch.allclose(zcr, exp), f"ZCR with zero_handling='{zero_handling}' did not match expected output."

    # test remaining parameters
    b = torch.Tensor([[[1, -0.3, -0.1, 0.3],
                       [-0.1, 0.1, 0.5, -1]],

                      [[-1, 0.1, -0.1, 1],
                       [0.1, -1, 0.4, 0.3]]])

    zcr_b = zero_crossing_rate(b, frame_length=3, hop_length=1, threshold=0.2, center=False)
    expected_b = torch.Tensor([[[[1/3, 1/3]],
                                [[0, 1/3]]],

                               [[[0, 1/3]],
                                [[1/3, 1/3]]]])
    assert torch.allclose(zcr_b, expected_b), "ZCR with different parameters did not match expected output."

    # test when hop_length does not divide frame_length
    c = torch.Tensor([1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1])

    expected_c = torch.Tensor([[0.2, 0.2, 0.6, 0.6, 0.2]])
    expected_c_no_center = torch.Tensor([[0.4, 0.6, 0.6]])

    zcr_c = zero_crossing_rate(c, frame_length=5, hop_length=3)
    zcr_c_no_center = zero_crossing_rate(c, frame_length=5, hop_length=3, center=False)
    assert torch.allclose(zcr_c, expected_c), "ZCR with hop_length not dividing length did not match expected output."
    assert torch.allclose(zcr_c_no_center, expected_c_no_center), "ZCR with no centering and hop_length not dividing length did not match expected output."
    

test_fills()
test_zcr()

print("all tests passed!")