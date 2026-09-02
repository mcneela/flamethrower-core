import numpy as np

from flamethrower.autograd import Tensor
import flamethrower.nn.loss as losses


def test_cross_entropy_is_stable_and_has_the_logits_gradient():
    logits = Tensor(np.array([[1000.0, -1000.0], [0.0, 0.0]]))
    labels = np.array([0, 1])

    value = losses.cross_entropy(labels, logits)
    value.backward()

    np.testing.assert_allclose(value.data, np.log(2) / 2)
    np.testing.assert_allclose(logits.grad, [[0.0, 0.0], [0.25, -0.25]])


def test_binary_cross_entropy_clips_extreme_probabilities():
    # A prediction of exactly 0 or 1 used to push log()'s argument slightly
    # past its safe range (via a "+eps inside every log call" approach),
    # which could even make the loss go negative for a perfect prediction -
    # impossible for a true loss. Clipping the prediction itself keeps this
    # finite and non-negative; a clipped prediction's local gradient is
    # exactly zero, the same saturation behavior as torch.clamp.
    predictions = Tensor(np.array([0.0, 1.0, 0.25, 0.75]))
    targets = np.array([0.0, 1.0, 0.0, 1.0])

    value = losses.binary_cross_entropy(targets, predictions)
    value.backward()

    np.testing.assert_allclose(value.data, 0.1438460362508906)
    np.testing.assert_allclose(predictions.grad, [0.0, 0.0, 1 / 3, -1 / 3])
    assert np.isfinite(value.data) and value.data >= 0


def test_mse_and_l1_have_the_expected_values_and_gradients():
    predictions = Tensor(np.array([3.0, 4.0]))
    targets = np.array([0.0, 0.0])

    mse = losses.mean_squared_error(targets, predictions)
    np.testing.assert_allclose(mse.data, 12.5)
    mse.backward()
    np.testing.assert_allclose(predictions.grad, [3.0, 4.0])

    np.testing.assert_allclose(losses.l1(targets, predictions).data, 7.0)


def test_l2_is_the_euclidean_residual_norm():
    predictions = Tensor(np.array([3.0, 4.0]))

    value = losses.l2(np.zeros(2), predictions, eps=0)

    np.testing.assert_allclose(value.data, 5.0)


def test_huber_uses_the_correct_delta_constant():
    # The standard Huber formula's linear branch is delta*(|r| - 0.5*delta),
    # i.e. delta*|r| - 0.5*delta**2. Subtracting plain delta/2 instead only
    # happened to be correct at the default delta=1; here delta=2 exposes
    # the difference (0.5*delta**2 == 2, not delta/2 == 1).
    predictions = Tensor(np.array([1.0, 3.0, -3.0]))

    value = losses.huber(np.zeros(3), predictions, delta=2)
    value.backward()

    np.testing.assert_allclose(value.data, [0.5, 4.0, 4.0])
    np.testing.assert_allclose(predictions.grad, [1.0, 2.0, -2.0])


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


def test_regularizer_is_added_once_after_reduction():
    predictions = Tensor(np.array([1.0, 3.0]))

    value = losses.mean_squared_error(
        np.zeros(2),
        predictions,
        regularizer=lambda: 2.0,
    )

    np.testing.assert_allclose(value.data, 7.0)
