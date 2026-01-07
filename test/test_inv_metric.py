"""
Test suite for InvMetric class
"""
import numpy as np
import pytest
from cerebro.utils import InvMetric


class TestInvMetric:
    """Test cases for InvMetric"""
    
    @pytest.fixture
    def inv_metric(self):
        """Create an InvMetric instance with default parameters"""
        return InvMetric(epsilon=1e-8, leverage=1.0, fee=0.01)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input data for testing"""
        batch_size = 4
        sequence_length = 60
        num_features = 4
        
        # Create realistic log-transformed data
        sources = np.random.randn(batch_size, sequence_length, num_features) * 0.01
        sources[:, :, -1] = np.cumsum(np.random.randn(batch_size, sequence_length) * 0.01, axis=1)
        
        inputs = {"sources": sources}
        inv = [np.array([1.0, 0.5, -0.5, 0.8])]  # Batch of inverse positions
        target = np.random.randn(batch_size, 1, 1) * 0.01  # Log-returns
        
        return inputs, inv, target
    
    def test_initialization(self):
        """Test InvMetric initialization with different parameters"""
        # Default parameters
        metric1 = InvMetric()
        assert metric1.epsilon == 1e-8
        assert metric1.leverage == 1.0
        assert metric1.fee == 0.0001  # 0.01 / 100
        
        # Custom parameters
        metric2 = InvMetric(epsilon=1e-6, leverage=2.0, fee=0.05)
        assert metric2.epsilon == 1e-6
        assert metric2.leverage == 2.0
        assert metric2.fee == 0.0005  # 0.05 / 100
    
    def test_forward_basic(self, inv_metric, sample_data):
        """Test basic forward pass"""
        inputs, inv, target = sample_data
        result = inv_metric.forward(inputs, inv, target)
        
        # Check output structure
        assert isinstance(result, dict)
        assert "pnl" in result
        assert "log_pnl" in result
        assert "mean_log_pnl" in result
    
    def test_forward_output_shapes(self, inv_metric, sample_data):
        """Test that output shapes are correct"""
        inputs, inv, target = sample_data
        batch_size = inputs["sources"].shape[0]
        
        result = inv_metric.forward(inputs, inv, target)
        
        # Check shapes
        assert result["pnl"].shape[0] == batch_size
        assert result["log_pnl"].shape[0] == batch_size
        assert isinstance(result["mean_log_pnl"], (float, np.ndarray, np.floating))
    
    def test_call_method(self, inv_metric, sample_data):
        """Test __call__ method delegates to forward"""
        inputs, inv, target = sample_data
        
        result_call = inv_metric(inputs, inv, target)
        result_forward = inv_metric.forward(inputs, inv, target)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(result_call["pnl"], result_forward["pnl"])
        np.testing.assert_array_almost_equal(result_call["log_pnl"], result_forward["log_pnl"])
    
    def test_positive_returns_positive_pnl(self, inv_metric):
        """Test that positive returns with long position produce positive PnL"""
        # Positive returns (exp(0.05) ≈ 1.051)
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([1.0])]  # Long position
        target = np.array([[[0.05]]])  # Positive return
        
        result = inv_metric.forward(inputs, inv, target)
        
        # Long position with positive return should have PnL > 1
        assert result["pnl"][0, 0] > 1.0
    
    def test_negative_returns_negative_pnl(self, inv_metric):
        """Test that negative returns with long position produce negative PnL"""
        # Negative returns (exp(-0.05) ≈ 0.951)
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([1.0])]  # Long position
        target = np.array([[[-0.05]]])  # Negative return
        
        result = inv_metric.forward(inputs, inv, target)
        
        # Long position with negative return should have PnL < 1
        assert result["pnl"][0, 0] < 1.0
    
    def test_short_position_hedges_loss(self, inv_metric):
        """Test that short position hedges long position losses"""
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        target = np.array([[[-0.05]]])  # Negative return
        
        # Long position
        inv_long = [np.array([1.0])]
        result_long = inv_metric.forward(inputs, inv_long, target)
        
        # Short position
        inv_short = [np.array([-1.0])]
        result_short = inv_metric.forward(inputs, inv_short, target)
        
        # Short should benefit from negative return
        assert result_short["pnl"][0, 0] > result_long["pnl"][0, 0]
    
    def test_fee_impact(self):
        """Test that fees reduce PnL"""
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([1.0])]
        target = np.array([[[0.0]]])  # No return
        
        # No fee
        metric_no_fee = InvMetric(leverage=1.0, fee=0.0)
        result_no_fee = metric_no_fee.forward(inputs, inv, target)
        
        # With fee
        metric_with_fee = InvMetric(leverage=1.0, fee=0.01)
        result_with_fee = metric_with_fee.forward(inputs, inv, target)
        
        # With fee should have lower PnL
        assert result_with_fee["pnl"][0, 0] < result_no_fee["pnl"][0, 0]
    
    def test_leverage_impact(self):
        """Test that leverage amplifies returns"""
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([1.0])]
        target = np.array([[[0.05]]])
        
        # Lower leverage
        metric_1x = InvMetric(leverage=1.0, fee=0.0)
        result_1x = metric_1x.forward(inputs, inv, target)
        
        # Higher leverage
        metric_2x = InvMetric(leverage=2.0, fee=0.0)
        result_2x = metric_2x.forward(inputs, inv, target)
        
        # Higher leverage should have higher PnL for positive returns
        assert result_2x["pnl"][0, 0] > result_1x["pnl"][0, 0]
    
    def test_log_pnl_clipping(self, inv_metric):
        """Test that log_pnl clips to avoid log(0)"""
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([1.0])]
        # Extreme negative return that could cause PnL < 0
        target = np.array([[[-10.0]]])
        
        result = inv_metric.forward(inputs, inv, target)
        
        # log_pnl should be valid (not nan or -inf)
        assert np.isfinite(result["log_pnl"]).all()
        assert not np.isnan(result["log_pnl"]).any()
    
    def test_mean_log_pnl_scaling(self, inv_metric, sample_data):
        """Test that mean_log_pnl is scaled by 24 * 364"""
        inputs, inv, target = sample_data
        result = inv_metric.forward(inputs, inv, target)
        
        # Verify scaling factor
        expected_mean = result["log_pnl"].mean() * 24 * 364
        np.testing.assert_almost_equal(result["mean_log_pnl"], expected_mean)
    
    def test_batch_processing(self):
        """Test that metric works correctly with batch data"""
        batch_size = 8
        inputs = {"sources": np.random.randn(batch_size, 60, 4) * 0.01}
        inv = [np.random.uniform(-1, 1, batch_size)]
        target = np.random.randn(batch_size, 1, 1) * 0.02
        
        metric = InvMetric()
        result = metric.forward(inputs, inv, target)
        
        assert result["pnl"].shape[0] == batch_size
        assert result["log_pnl"].shape[0] == batch_size
    
    def test_zero_position(self, inv_metric):
        """Test that zero position (inv=0) results in PnL=1"""
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([0.0])]
        target = np.array([[[0.5]]])  # Large return
        
        result = inv_metric.forward(inputs, inv, target)
        
        # Zero position should have PnL ≈ 1 - fee
        # fee = 0.01/100 * |0| = 0, so PnL = 1
        assert abs(result["pnl"][0, 0] - (1.0 - inv_metric.fee * 0.0)) < 1e-10
    
    def test_epsilon_parameter(self):
        """Test that epsilon prevents division by zero in extreme cases"""
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([1.0])]
        target = np.array([[[0.0]]])
        
        # With small epsilon
        metric_small_eps = InvMetric(epsilon=1e-10)
        result_small = metric_small_eps.forward(inputs, inv, target)
        
        # With larger epsilon
        metric_large_eps = InvMetric(epsilon=1e-2)
        result_large = metric_large_eps.forward(inputs, inv, target)
        
        # Both should produce finite results
        assert np.isfinite(result_small["pnl"]).all()
        assert np.isfinite(result_large["pnl"]).all()


class TestInvMetricIntegration:
    """Integration tests for InvMetric with realistic scenarios"""
    
    def test_bull_market_long_position(self):
        """Test long position in bull market"""
        metric = InvMetric(leverage=1.0, fee=0.01)
        
        # Simulating bull market (consistent positive returns)
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([1.0])]
        
        returns = [0.01, 0.02, 0.015]  # Positive returns
        pnl_values = []
        
        for ret in returns:
            target = np.array([[[ret]]])
            result = metric.forward(inputs, inv, target)
            pnl_values.append(result["pnl"][0, 0])
        
        # All should be > 1
        assert all(p > 1.0 for p in pnl_values)
        # Should be increasing
        assert all(pnl_values[i] <= pnl_values[i+1] or abs(pnl_values[i] - pnl_values[i+1]) < 0.01 
                   for i in range(len(pnl_values)-1))
    
    def test_bear_market_short_position(self):
        """Test short position in bear market"""
        metric = InvMetric(leverage=1.0, fee=0.01)
        
        inputs = {"sources": np.array([[[0.0, 0.0, 0.0, 0.0]]])}
        inv = [np.array([-1.0])]  # Short position
        
        returns = [-0.01, -0.02, -0.015]  # Negative returns
        pnl_values = []
        
        for ret in returns:
            target = np.array([[[ret]]])
            result = metric.forward(inputs, inv, target)
            pnl_values.append(result["pnl"][0, 0])
        
        # All should be > 1 (profiting from negative returns)
        assert all(p > 1.0 for p in pnl_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
