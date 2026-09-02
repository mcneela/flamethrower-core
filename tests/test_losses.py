import numpy as np
import pytest

from flamethrower.autograd import Tensor
import flamethrower.nn.loss as losses


def test_cross_entropy_is_stable_and_has_the_logits_gradient():
    logits = Tensor(np.array([[1000.0, -1000.0], [0.0, 0.0]]))
    labels = np.array([0, 1])

    value = losses.cross_entropy(labels, logits)
    value.backward()

    np.testing.assert_allclose(value.data, np.log(2) / 2)
    np.testing.assert_allclose(logits.grad, [[0.0, 0.0], [0.25, -0.25]])


def test_cross_entropy_none_reduction_returns_one_loss_per_example():
    logits = Tensor(np.array([[2.0, 1.0], [0.0, 0.0]]))

    value = losses.cross_entropy(np.array([0, 1]), logits, reduction='none')

    expected = [np.log1p(np.exp(-1)), np.log(2)]
    np.testing.assert_allclose(value.data, expected)


def test_binary_cross_entropy_clips_extreme_probabilities():
    predictions = Tensor(np.array([0.0, 1.0, 0.25, 0.75]))
    targets = np.array([0.0, 1.0, 0.0, 1.0])

    value = losses.binary_cross_entropy(targets, predictions, reduction='none')
    np.testing.assert_allclose(
        value.data,
        [-np.log(1 - 1e-7), -np.log(1 - 1e-7), -np.log(0.75), -np.log(0.75)],
    )
    assert np.all(np.isfinite(value.data))


def test_mse_and_l1_support_consistent_reductions_and_gradients():
    predictions = Tensor(np.array([3.0, 4.0]))
    targets = np.array([0.0, 0.0])

    mse = losses.mean_squared_error(targets, predictions)
    np.testing.assert_allclose(mse.data, 12.5)
    mse.backward()
    np.testing.assert_allclose(predictions.grad, [3.0, 4.0])

    predictions = Tensor(np.array([3.0, 4.0]))
    np.testing.assert_allclose(
        losses.mean_squared_error(targets, predictions, reduction='none').data,
        [9.0, 16.0],
    )
    np.testing.assert_allclose(
        losses.mean_squared_error(targets, predictions, reduction='sum').data,
        25.0,
    )
    np.testing.assert_allclose(losses.l1(targets, predictions).data, 7.0)
    np.testing.assert_allclose(
        losses.l1(targets, predictions, reduction='mean').data,
        3.5,
    )


def test_l2_is_the_euclidean_residual_norm():
    predictions = Tensor(np.array([3.0, 4.0]))

    value = losses.l2(np.zeros(2), predictions, eps=0)

    np.testing.assert_allclose(value.data, 5.0)


def test_huber_uses_the_correct_delta_constant_and_mean_reduction():
    predictions = Tensor(np.array([1.0, 3.0, -3.0]))

    value = losses.huber(np.zeros(3), predictions, delta=2)
    value.backward()

    np.testing.assert_allclose(value.data, (0.5 + 4.0 + 4.0) / 3)
    np.testing.assert_allclose(predictions.grad, [1 / 3, 2 / 3, -2 / 3])


def test_modified_huber_treats_zero_prediction_as_a_continuous_score():
    prediction = Tensor(np.array([0.0]))

    value = losses.huber_binary_loss(np.array([1]), prediction)
    value.backward()

    np.testing.assert_allclose(value.data, 1.0)
    np.testing.assert_allclose(prediction.grad, [-2.0])


def test_kl_divergence_handles_zero_probability_entries():
    p = np.array([0.0, 0.5, 0.5])
    q = Tensor(np.array([0.2, 0.25, 0.55]))

    value = losses.kl_divergence(p, q)

    expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.55)
    np.testing.assert_allclose(value.data, expected)
    assert np.isfinite(value.data)


@pytest.mark.parametrize("reduction", ["invalid", "batchmean", None])
def test_losses_reject_unknown_reductions(reduction):
    with pytest.raises(ValueError, match="reduction"):
        losses.mean_squared_error(
            np.array([0.0]),
            Tensor(np.array([1.0])),
            reduction=reduction,
        )


def test_regularizer_is_added_once_after_reduction():
    predictions = Tensor(np.array([1.0, 3.0]))

    value = losses.mean_squared_error(
        np.zeros(2),
        predictions,
        regularizer=lambda: 2.0,
    )

    np.testing.assert_allclose(value.data, 7.0)
