from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment for Day-1 dataset parquet.
    
    State = [dataset_state_vector, balance_norm, position, unrealized_pnl_norm, equity_norm]
    Actions: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
    Reward = log(equity_t / equity_{t-1}) - risk_penalty * abs(position) * volatility_proxy
    """
    
    HOLD = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        initial_balance: float = 10000.0,
        taker_fee: float = 0.0004,
        slippage_bps: float = 1.0,
        risk_penalty: float = 0.1,
        position_fraction: float = 1.0,
        cooldown_steps: int = 10,
        hold_threshold: float = 0.1,
        cost_penalty: float = 0.01,
        position_change_penalty: float = 0.005,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        if len(self.df) < 2:
            raise ValueError("DataFrame must have at least 2 rows")
        
        # Validate required columns
        required_cols = ["timestamp_ms", "close", "next_log_return", "state"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Extract state vector length from first row
        first_state = self.df["state"].iloc[0]
        if isinstance(first_state, (list, tuple)):
            self.state_len = len(first_state)
        elif isinstance(first_state, np.ndarray):
            self.state_len = len(first_state)
        else:
            raise ValueError(f"state column must contain list/array, got {type(first_state)}")
        
        # Observation space: state_vector + 4 account features
        obs_dim = self.state_len + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Parameters
        self.initial_balance = float(initial_balance)
        self.taker_fee = float(taker_fee)
        self.slippage_bps = float(slippage_bps)
        self.risk_penalty = float(risk_penalty)
        self.position_fraction = float(position_fraction)
        self.cooldown_steps = int(cooldown_steps)
        self.hold_threshold = float(hold_threshold)
        self.cost_penalty = float(cost_penalty)
        self.position_change_penalty = float(position_change_penalty)
        
        # State variables (reset in reset())
        self.current_idx: int = 0
        self.balance: float = 0.0
        self.position: int = 0  # -1=short, 0=flat, 1=long
        self.entry_price: float = 0.0
        self.qty: float = 0.0
        self.equity: float = 0.0
        self.prev_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.cooldown_remaining: int = 0
        self.prev_position: int = 0
        
        # Statistics
        self.trade_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0
        self.realized_pnl: float = 0.0
        self.total_fees_paid: float = 0.0
        
        # Seed
        self.seed(seed)
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_idx = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.qty = 0.0
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
        self.peak_equity = self.initial_balance
        
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.realized_pnl = 0.0
        self.cooldown_remaining = 0
        self.prev_position = 0
        self.total_fees_paid = 0.0
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info
    
    def step(self, action: int, action_probs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE)
            action_probs: Optional action probabilities from policy (for hold threshold)
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        if self.current_idx >= len(self.df):
            # Already terminated
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info
        
        # Get current market data
        row = self.df.iloc[self.current_idx]
        price = float(row["close"])
        next_lr = float(row["next_log_return"])
        volatility_proxy = abs(next_lr)
        
        # Store previous state for reward calculation
        self.prev_equity = self.equity
        self.prev_position = self.position
        
        # Apply cooldown: block trades if cooldown is active
        if self.cooldown_remaining > 0:
            if action in (self.LONG, self.SHORT, self.CLOSE):
                action = self.HOLD  # Force HOLD during cooldown
            self.cooldown_remaining -= 1
        
        # Apply hold threshold: if predicted advantage is tiny, force HOLD
        if action_probs is not None and action != self.HOLD:
            # Check if action confidence is below threshold
            action_prob = float(action_probs[action])
            max_prob = float(np.max(action_probs))
            confidence = action_prob / (max_prob + 1e-9)
            
            if confidence < self.hold_threshold:
                action = self.HOLD  # Force HOLD if confidence too low
        
        # Execute action
        position_changed = False
        if action == self.LONG:
            if self.position != 1:
                self._execute_long(price)
                position_changed = True
        elif action == self.SHORT:
            if self.position != -1:
                self._execute_short(price)
                position_changed = True
        elif action == self.CLOSE:
            if self.position != 0:
                self._execute_close(price)
                position_changed = True
        # HOLD: do nothing
        
        # Update equity (mark-to-market)
        self._update_equity(price)
        
        # Compute reward (includes cost penalty and position change penalty)
        reward = self._compute_reward(volatility_proxy, position_changed)
        
        # Advance to next timestep
        self.current_idx += 1
        
        # Check termination
        terminated = self.current_idx >= len(self.df)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, float(reward), terminated, False, info
    
    def _execute_long(self, price: float) -> None:
        """Execute LONG action."""
        if self.position == 1:
            # Already long, do nothing
            return
        
        if self.position == -1:
            # Close short first
            self._execute_close(price)
        
        # Open long position: buy qty at execution_price
        execution_price = price * (1.0 + self.slippage_bps / 10000.0)
        available = self.balance * self.position_fraction
        fee = available * self.taker_fee
        cost_after_fee = available - fee
        
        if cost_after_fee <= 0:
            # Not enough balance
            return
        
        self.qty = cost_after_fee / execution_price
        self.balance -= available  # Deduct the full allocated amount
        self.position = 1
        self.entry_price = execution_price
        self.trade_count += 1
        self.total_fees_paid += fee
        self.cooldown_remaining = self.cooldown_steps  # Start cooldown
    
    def _execute_short(self, price: float) -> None:
        """Execute SHORT action."""
        if self.position == -1:
            # Already short, do nothing
            return
        
        if self.position == 1:
            # Close long first
            self._execute_close(price)
        
        # Open short position: sell qty at execution_price
        # We receive entry_price * qty as proceeds, pay opening fee
        execution_price = price * (1.0 - self.slippage_bps / 10000.0)
        available = self.balance * self.position_fraction
        opening_fee = available * self.taker_fee
        
        if opening_fee > self.balance:
            # Not enough balance for fee
            return
        
        self.qty = available / execution_price
        entry_notional = execution_price * self.qty
        self.balance = self.balance - opening_fee + entry_notional  # Pay fee, receive proceeds
        self.position = -1
        self.entry_price = execution_price
        self.trade_count += 1
        self.total_fees_paid += opening_fee
        self.cooldown_remaining = self.cooldown_steps  # Start cooldown
        self.cooldown_remaining = self.cooldown_steps  # Start cooldown
    
    def _execute_close(self, price: float) -> None:
        """Execute CLOSE action."""
        if self.position == 0:
            # No position to close
            return
        
        # Close position
        if self.position == 1:
            # Close long: sell qty at execution_price
            execution_price = price * (1.0 - self.slippage_bps / 10000.0)
            notional = execution_price * self.qty
            fee = notional * self.taker_fee
            pnl = notional - (self.entry_price * self.qty) - fee
            self.balance += notional - fee
        else:
            # Close short: buy back qty at execution_price
            execution_price = price * (1.0 + self.slippage_bps / 10000.0)
            closing_notional = execution_price * self.qty
            closing_fee = closing_notional * self.taker_fee
            cost_to_close = closing_notional + closing_fee
            
            # PnL = proceeds_received - cost_to_close
            # We received entry_price * qty when we opened (minus opening fee)
            proceeds_received = self.entry_price * self.qty
            pnl = proceeds_received - cost_to_close
            
            if cost_to_close <= self.balance:
                self.balance -= cost_to_close
            else:
                # Not enough balance to close (liquidation scenario)
                pnl = self.balance - cost_to_close
                self.balance = 0.0
        
        self.realized_pnl += pnl
        
        if pnl > 0:
            self.win_count += 1
        elif pnl < 0:
            self.loss_count += 1
        
        # Track fees (already included in closing_fee calculations above)
        if self.position == 1:
            closing_fee = notional * self.taker_fee
        else:
            closing_fee = closing_notional * self.taker_fee
        self.total_fees_paid += closing_fee
        
        self.position = 0
        self.qty = 0.0
        self.entry_price = 0.0
        self.cooldown_remaining = self.cooldown_steps  # Start cooldown after closing
    
    def _update_equity(self, price: float) -> None:
        """Update equity based on current position."""
        if self.position == 0:
            self.equity = self.balance
        elif self.position == 1:
            # Long position: equity = balance + current_value of position
            current_value = price * self.qty
            self.equity = self.balance + current_value
        else:
            # Short position: equity = balance - (current_value - entry_value)
            # We owe current_value to buy back, but received entry_value when we opened
            entry_value = self.entry_price * self.qty
            current_value = price * self.qty
            unrealized_pnl = entry_value - current_value
            self.equity = self.balance + unrealized_pnl
        
        # Ensure equity doesn't go negative (clamp to small positive value)
        self.equity = max(self.equity, 1e-6)
        self.peak_equity = max(self.peak_equity, self.equity)
    
    def _compute_reward(self, volatility_proxy: float, position_changed: bool) -> float:
        """
        Compute reward: log return - risk penalty - cost penalty - position change penalty.
        
        Reward = Δ log(equity) - risk_penalty - cost_penalty - position_change_penalty
        """
        if self.prev_equity <= 0:
            return -1.0  # Penalty for zero/negative equity
        
        # Log return component
        log_return = np.log(self.equity / self.prev_equity)
        
        # Risk penalty (proportional to position size and volatility)
        risk_term = self.risk_penalty * abs(self.position) * volatility_proxy
        
        # Cost penalty (proportional to fees paid this step)
        # Estimate fees as fraction of equity (normalized)
        cost_term = 0.0
        if self.total_fees_paid > 0 and self.equity > 0:
            # Normalize fees relative to equity
            cost_term = self.cost_penalty * (self.total_fees_paid / max(self.equity, 1e-6))
        
        # Position change penalty (discourage unnecessary turnover)
        position_change_term = 0.0
        if position_changed:
            position_change_term = self.position_change_penalty
        
        # Drawdown penalty (optional, already captured in log_return but can add explicit term)
        drawdown_term = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if drawdown > 0.1:  # Only penalize significant drawdowns
                drawdown_term = self.risk_penalty * drawdown * 0.5
        
        reward = log_return - risk_term - cost_term - position_change_term - drawdown_term
        
        return float(reward)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        if self.current_idx >= len(self.df):
            # Return last valid observation
            self.current_idx = len(self.df) - 1
        
        row = self.df.iloc[self.current_idx]
        state_vec = row["state"]
        
        # Convert state to numpy array if needed
        if isinstance(state_vec, (list, tuple)):
            state_arr = np.array(state_vec, dtype=np.float32)
        elif isinstance(state_vec, np.ndarray):
            state_arr = state_vec.astype(np.float32)
        else:
            raise ValueError(f"Invalid state type: {type(state_vec)}")
        
        # Clean NaN/inf values in state vector
        state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize account features
        balance_norm = self.balance / max(self.initial_balance, 1e-12)
        position_norm = float(self.position)  # -1, 0, or 1
        
        unrealized_pnl = 0.0
        if self.position != 0:
            price = float(row["close"])
            if self.position == 1:
                unrealized_pnl = (price - self.entry_price) * self.qty
            else:
                unrealized_pnl = (self.entry_price - price) * self.qty
        
        unrealized_pnl_norm = unrealized_pnl / max(self.initial_balance, 1e-12)
        equity_norm = self.equity / max(self.initial_balance, 1e-12)
        
        # Clip to reasonable ranges to prevent overflow
        balance_norm = np.clip(balance_norm, -10.0, 10.0)
        unrealized_pnl_norm = np.clip(unrealized_pnl_norm, -10.0, 10.0)
        equity_norm = np.clip(equity_norm, 0.0, 10.0)
        
        # Concatenate: [state_vector, balance_norm, position, unrealized_pnl_norm, equity_norm]
        account_features = np.array([balance_norm, position_norm, unrealized_pnl_norm, equity_norm], dtype=np.float32)
        account_features = np.nan_to_num(account_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        obs = np.concatenate([state_arr, account_features])
        
        # Final safety check
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        row = self.df.iloc[min(self.current_idx, len(self.df) - 1)]
        return {
            "current_idx": int(self.current_idx),
            "timestamp_ms": int(row["timestamp_ms"]),
            "price": float(row["close"]),
            "balance": float(self.balance),
            "position": int(self.position),
            "qty": float(self.qty),
            "entry_price": float(self.entry_price),
            "equity": float(self.equity),
            "peak_equity": float(self.peak_equity),
            "drawdown": float((self.peak_equity - self.equity) / self.peak_equity) if self.peak_equity > 0 else 0.0,
            "trade_count": int(self.trade_count),
            "win_count": int(self.win_count),
            "loss_count": int(self.loss_count),
            "realized_pnl": float(self.realized_pnl),
        }


def make_env_from_dataframe(
    df: pd.DataFrame,
    **kwargs: Any,
) -> TradingEnv:
    """Helper function to create environment from DataFrame."""
    return TradingEnv(df, **kwargs)
