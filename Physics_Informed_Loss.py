import tensorflow as tf
import keras
from keras import ops


# ---------------------------------------------------------------------------
# Standalone loss for model.compile(loss=...)
# ---------------------------------------------------------------------------
def weighted_mse_loss(y_true, y_pred):
    """Flood-focused loss — v1 weights + false alarm & sparsity penalties.

    V1 used 10x wet weighting and worked well for localization.
    V2 used 100x and caused diffuse wash (water everywhere).
    V3 keeps v1 base weights, adds targeted false-alarm + sparsity penalties.
    """
    squared_error = ops.square(y_true - y_pred)
    wet_mask = ops.cast(y_true > 0.0, "float32")
    dry_mask = 1.0 - wet_mask

    # --- Component 1: Weighted MSE (v1 level: 10x on wet pixels) ---
    weights = ops.where(y_true > 0.0, 10.0, 1.0)
    mse_loss = ops.mean(weights * squared_error)

    # --- Component 2: False alarm — punish predicting water on dry ground ---
    # This prevents the diffuse wash problem from v2
    false_alarm = dry_mask * ops.relu(y_pred)
    false_alarm_loss = ops.mean(ops.square(false_alarm)) * 20.0

    # --- Component 3: Sparsity — predicted wet fraction should match true ---
    pred_wet_frac = ops.mean(ops.cast(y_pred > 0.01, "float32"))
    true_wet_frac = ops.mean(wet_mask)
    sparsity_loss = ops.square(pred_wet_frac - true_wet_frac) * 50.0

    return mse_loss + false_alarm_loss + sparsity_loss


# ---------------------------------------------------------------------------
# Helper: image gradients using keras.ops
# ---------------------------------------------------------------------------
def _image_gradients(image):
    """Compute spatial gradients (finite differences) for (batch, H, W, C) tensor.

    Returns (grad_y, grad_x) — each same shape as input, with last row/col repeated.
    """
    grad_y = image[:, 1:, :, :] - image[:, :-1, :, :]
    grad_y = ops.concatenate([grad_y, grad_y[:, -1:, :, :]], axis=1)

    grad_x = image[:, :, 1:, :] - image[:, :, :-1, :]
    grad_x = ops.concatenate([grad_x, grad_x[:, :, -1:, :]], axis=2)

    return grad_y, grad_x


# ---------------------------------------------------------------------------
# Physics loss functions — operate on real tensors (called in train_step)
# ---------------------------------------------------------------------------
def compute_gravity_loss(dem, last_obs_depth, predictions):
    """Penalize water flowing uphill (toward higher water surface elevation).

    Args:
        dem:            (batch, 128, 128) — normalized DEM (channel 0)
        last_obs_depth: (batch, 128, 128) — most recent observed depth (channel 9)
        predictions:    (batch, 128, 128, 4) — predicted depths at t+15..t+60
    """
    h_sequence = [
        last_obs_depth,
        predictions[:, :, :, 0],
        predictions[:, :, :, 1],
        predictions[:, :, :, 2],
        predictions[:, :, :, 3],
    ]

    total_violation = 0.0

    for k in range(4):
        h_k = h_sequence[k]
        h_k1 = h_sequence[k + 1]

        wse_k = dem + h_k
        delta_h = h_k1 - h_k

        wse_4d = ops.expand_dims(wse_k, axis=-1)
        dh_4d = ops.expand_dims(delta_h, axis=-1)

        grad_wse_y, grad_wse_x = _image_gradients(wse_4d)
        grad_dh_y, grad_dh_x = _image_gradients(dh_4d)

        # Dot product: positive ⇒ depth increases align with increasing WSE (uphill)
        dot = grad_dh_x * grad_wse_x + grad_dh_y * grad_wse_y

        # Only penalize uphill flow, weighted by depth-change magnitude
        violation = ops.relu(dot) * (ops.abs(dh_4d) + 1e-6)

        total_violation = total_violation + ops.mean(violation)

    return total_violation / 4.0


def compute_continuity_loss(dem, last_obs_depth, predictions, rain_signal,
                            epsilon_base=0.01, rain_scale=0.5):
    """Penalize local mass imbalance — water shouldn't appear from nowhere.

    Uses rainfall intensity to dynamically relax the conservation threshold.

    Args:
        dem:            (batch, 128, 128) — normalized DEM (channel 0)
        last_obs_depth: (batch, 128, 128) — most recent observed depth (channel 9)
        predictions:    (batch, 128, 128, 4) — predicted depths at t+15..t+60
        rain_signal:    (batch, 12, 8) — rainfall/tide from input channel 10
    """
    mean_rain = ops.mean(rain_signal, axis=[1, 2])  # (batch,)
    epsilon = epsilon_base + rain_scale * ops.reshape(mean_rain, [-1, 1, 1, 1])

    h_sequence = [
        last_obs_depth,
        predictions[:, :, :, 0],
        predictions[:, :, :, 1],
        predictions[:, :, :, 2],
        predictions[:, :, :, 3],
    ]

    total_violation = 0.0

    for k in range(4):
        delta_h = h_sequence[k + 1] - h_sequence[k]
        dh_4d = ops.expand_dims(delta_h, axis=-1)  # (batch, 128, 128, 1)

        # Local average depth change over 5×5 neighborhood
        local_avg = ops.average_pool(dh_4d, pool_size=5, strides=1, padding='same')

        # Water creation beyond what rainfall explains (strong penalty)
        creation_penalty = ops.square(ops.relu(local_avg - epsilon))

        # Excessive drainage beyond infiltration threshold (mild penalty)
        drainage_penalty = ops.square(ops.relu(-local_avg - epsilon))

        total_violation = total_violation + ops.mean(creation_penalty + 0.1 * drainage_penalty)

    return total_violation / 4.0


# ---------------------------------------------------------------------------
# Custom Model subclass — computes physics losses in train_step
# ---------------------------------------------------------------------------
class FloodModel(keras.Model):
    """Wraps any flood forecasting model and adds physics losses during training.

    The physics losses need access to model inputs (DEM, water depth history,
    rainfall). Keras 3 Functional models don't support add_loss(), so we
    compute them here in train_step where we have real tensors.
    """

    def __init__(self, *args, gravity_target=1.0, continuity_target=0.5,
                 warmup_epochs=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.gravity_target = gravity_target
        self.continuity_target = continuity_target
        self.warmup_epochs = warmup_epochs
        self.current_epoch = tf.Variable(0.0, trainable=False)

        # Metrics for tracking
        self.gravity_metric = keras.metrics.Mean(name="gravity_loss")
        self.continuity_metric = keras.metrics.Mean(name="continuity_loss")

    def _physics_weight(self):
        """Ramp from 0 → 1 over warmup_epochs."""
        return ops.minimum(1.0, (self.current_epoch + 1.0) / self.warmup_epochs)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            # Standard compiled loss (weighted MSE)
            data_loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

            # Extract physics-relevant channels from input
            dem = x[:, :, :, 0]
            last_obs = x[:, :, :, 9]
            rain_signal = x[:, :12, :8, 10]

            # Compute physics losses
            g_loss = compute_gravity_loss(dem, last_obs, y_pred)
            c_loss = compute_continuity_loss(dem, last_obs, y_pred, rain_signal)

            # Apply warm-up ramping
            ramp = self._physics_weight()
            total_loss = (data_loss
                          + self.gravity_target * ramp * g_loss
                          + self.continuity_target * ramp * c_loss)

        # Backprop
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            elif metric.name != "gravity_loss" and metric.name != "continuity_loss":
                metric.update_state(y, y_pred)
        self.gravity_metric.update_state(g_loss)
        self.continuity_metric.update_state(c_loss)

        # Return all metrics
        results = {m.name: m.result() for m in self.metrics}
        results["gravity_loss"] = self.gravity_metric.result()
        results["continuity_loss"] = self.continuity_metric.result()
        return results

    @property
    def metrics(self):
        metrics = super().metrics
        return metrics

    def reset_metrics(self):
        super().reset_metrics()
        self.gravity_metric.reset_state()
        self.continuity_metric.reset_state()


# ---------------------------------------------------------------------------
# Callback to update epoch counter on the model
# ---------------------------------------------------------------------------
class PhysicsLossWarmup(keras.callbacks.Callback):
    """Updates the model's epoch counter for physics loss warm-up scheduling."""

    def on_epoch_begin(self, epoch, logs=None):
        self.model.current_epoch.assign(float(epoch))
        ramp = min(1.0, (epoch + 1) / self.model.warmup_epochs)
        if epoch < self.model.warmup_epochs:
            print(f"  [PhysicsWarmup] epoch {epoch+1}: ramp={ramp:.2f} "
                  f"(gravity={self.model.gravity_target * ramp:.3f}, "
                  f"continuity={self.model.continuity_target * ramp:.3f})")


# ---------------------------------------------------------------------------
# Original combined loss (kept for backward compatibility)
# ---------------------------------------------------------------------------
class Physics_Informed_Loss(keras.losses.Loss):
    def __init__(self, temporal_weight=10.0, terrain_weight=5.0, dx=2.5, dt=900.0,
                 name="physics_informed_loss"):
        super().__init__(name=name)
        self.temporal_weight = temporal_weight
        self.terrain_weight = terrain_weight
        self.dx = dx
        self.dt = dt

    def call(self, y_true, y_pred):
        squared_error = ops.square(y_true - y_pred)
        weights = ops.where(y_true > 0.0, 10.0, 1.0)
        data_loss = ops.mean(weights * squared_error)

        d1 = y_pred[:, :, :, 1] - y_pred[:, :, :, 0]
        d2 = y_pred[:, :, :, 2] - y_pred[:, :, :, 1]
        d3 = y_pred[:, :, :, 3] - y_pred[:, :, :, 2]
        accel1 = d2 - d1
        accel2 = d3 - d2
        temporal_loss = ops.mean(ops.square(accel1) + ops.square(accel2))

        spatial_loss = 0.0
        for ch in range(4):
            depth = y_pred[:, :, :, ch:ch+1]
            grad_y, grad_x = _image_gradients(depth)
            grad_yy, _ = _image_gradients(grad_y)
            _, grad_xx = _image_gradients(grad_x)
            laplacian = (grad_xx + grad_yy) / (self.dx ** 2)
            spatial_loss = spatial_loss + ops.mean(ops.square(laplacian))
        spatial_loss = spatial_loss / 4.0

        return data_loss + self.temporal_weight * temporal_loss + self.terrain_weight * spatial_loss
