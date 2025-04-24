from env import PortfolioEnv
from agent import REINFORCEAgent
import numpy as np

def train(env, agent, episodes=10):
    for ep in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []

        while True:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            if done:
                break

        agent.update_policy(rewards, log_probs)
        total_return = env.portfolio_values[-1] / env.portfolio_values[0] - 1
        print(f"Episode {ep+1}: Total Return = {total_return:.4f}")

def suggest_action(env, agent):
    state = env._get_observation()
    action_idx, _ = agent.select_action(state)
    action_list = env._decode_action(action_idx)

    print("\nSuggested Action:")
    for ticker, action in zip(env.tickers, action_list):
        if action == 0:
            act_str = "SELL"
        elif action == 1:
            act_str = "HOLD"
        else:
            act_str = "BUY"
        print(f"{ticker}: {act_str}")


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT"]
    shares = [10, 5]
    purchase_prices = [150, 250]

    env = PortfolioEnv(tickers, shares, purchase_prices)
    state_dim = env._get_observation().shape[0]
    action_dim = 3 ** len(tickers)

    agent = REINFORCEAgent(state_dim, action_dim)
    print("Training RL agent...")
    train(env, agent, episodes=50)

    print("\n--- Generating suggestion based on current portfolio state ---")
    suggest_action(env, agent)

