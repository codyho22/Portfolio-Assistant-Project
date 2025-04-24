import yfinance as yf
import numpy as np
import pandas as pd

class PortfolioEnv:
    def __init__(self, tickers, shares, purchase_prices):
        self.tickers = tickers
        self.shares = np.array(shares)
        self.purchase_prices = np.array(purchase_prices)
        self.start_date = "2023-01-01"
        self.end_date = "2025-01-01"

        raw_data = yf.download(tickers, start=self.start_date, end=self.end_date)

        if isinstance(raw_data.columns, pd.MultiIndex):
            self.prices = raw_data["Close"].ffill().dropna()
        else:
            self.prices = raw_data[["Close"]].copy()
            self.prices.columns = pd.Index([tickers])
            self.prices = self.prices.ffill().dropna()

        self.returns = self.prices.pct_change().fillna(0)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = 0
        self.asset_holdings = self.shares.copy()
        self.portfolio_values = [self._get_portfolio_value()]
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        return self.returns.iloc[self.current_step].values.astype(np.float32)

    def _get_portfolio_value(self):
        prices = self.prices.iloc[self.current_step].values
        return self.cash + np.dot(self.asset_holdings, prices)

    def step(self, action_idx):
        actions = self._decode_action(action_idx)
        prices = self.prices.iloc[self.current_step].values

        for i, act in enumerate(actions):
            if act == 0:  # Sell
                self.cash += self.asset_holdings[i] * prices[i]
                self.asset_holdings[i] = 0
            elif act == 2:  # Buy
                num_shares = self.cash // prices[i]
                self.asset_holdings[i] += num_shares
                self.cash -= num_shares * prices[i]

        self.current_step += 1
        self.done = self.current_step >= len(self.prices) - 1

        portfolio_value = self._get_portfolio_value()
        self.portfolio_values.append(portfolio_value)

        if len(self.portfolio_values) > 1:
            returns = pd.Series(self.portfolio_values).pct_change().dropna()
            if returns.std() > 0:
                reward = returns.mean() / (returns.std() + 1e-8)
            else:
                reward = 0
        else:
            reward = 0

        return self._get_observation(), reward, self.done, {}

    def _decode_action(self, idx):
        base = 3
        action_list = []
        for _ in range(len(self.tickers)):
            action_list.append(idx % base)
            idx //= base
        return action_list[::-1]

    def render(self):
        print(f"Step: {self.current_step}, Portfolio Value: ${self._get_portfolio_value():.2f}")