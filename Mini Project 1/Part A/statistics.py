import numpy as np
import matplotlib as mpl

gp_elbo = np.array([-89.7416305419922, -91.58768662109375, -89.50657092285157, -90.94428303222656, -89.42571530761718, -89.98001264648437])

print(f"Gaussian Prior 6 runs: Mean Avg ELBO: {gp_elbo.mean():.2f}, Std of Avg ELBO {gp_elbo.std():.2f}")

mob_elbo = np.array([-87.40810258789062, -88.46548905029297, -86.1971771118164, -86.84063795166016, -89.45142404785156, -89.31198603515625])

print(f"Mixture of Gaussian Prior 6 runs: Mean Avg ELBO: {mob_elbo.mean():.2f}, Std of Avg ELBO {mob_elbo.std():.2f}")

flow_elbo = np.array([-88.2771490966797, -86.70559956054687, -88.57668880615235, -88.16901608886718, -87.33453779296875, -88.0205509765625])

print(f"Flow-based prior 6 runs: Mean Avg ELBO: {flow_elbo.mean():.2f}, Std of Avg ELBO {flow_elbo.std():.2f}")


