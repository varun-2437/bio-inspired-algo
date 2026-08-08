import numpy as np
from mlp_sboa.fitness import fitness_function


def levy_flight(dimension):
    # simple levy step generator
    beta = 1.5
    sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, size=dimension)
    v = np.random.normal(0, 1, size=dimension)
    step = u / (np.abs(v) ** (1 / beta))
    return step


def SBOA(X_train, Y_train, input_dim, hidden_dim, output_dim,
         SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """A straightforward port of the SBOA logic from R to NumPy.
    fitness is a function(params, X_train, Y_train, input_dim, hidden_dim, output_dim)
    """
    # initialize population
    X = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))
    fit = np.full(SearchAgents, np.inf)
    for i in range(SearchAgents):
        fit[i] = fitness(X[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)

    best_so_far = np.zeros(Max_iterations)
    fbest = np.min(fit)
    Bast_P = X[np.argmin(fit)].copy()

    for t in range(1, Max_iterations + 1):
        CF = (1 - t / Max_iterations) ** (2 * t / Max_iterations)
        best = np.min(fit)
        location = np.argmin(fit)
        if t == 1:
            Bast_P = X[location, :].copy()
            fbest = best
        elif best < fbest:
            fbest = best
            Bast_P = X[location, :].copy()

        # update each agent
        for i in range(SearchAgents):
            if t < Max_iterations / 3:
                # random walk inspired
                idx1, idx2 = np.random.randint(0, SearchAgents, size=2)
                R1 = np.random.rand(dimension)
                X1 = X[i, :] + (X[idx1, :] - X[idx2, :]) * R1
                X1 = np.clip(X1, lowerbound, upperbound)
            elif t < 2 * Max_iterations / 3:
                RB = np.random.randn(dimension)
                X1 = Bast_P + np.exp((t / Max_iterations) ** 4) * (RB - 0.5) * (Bast_P - X[i, :])
                X1 = np.clip(X1, lowerbound, upperbound)
            else:
                RL = 0.5 * levy_flight(dimension)
                X1 = Bast_P + CF * X[i, :] * RL
                X1 = np.clip(X1, lowerbound, upperbound)

            f_newP1 = fitness(X1, X_train, Y_train, input_dim, hidden_dim, output_dim)
            if f_newP1 <= fit[i]:
                X[i, :] = X1
                fit[i] = f_newP1

        # escape strategy
        r = np.random.rand()
        k = np.random.randint(0, SearchAgents)
        Xrandom = X[k, :]
        for i in range(SearchAgents):
            if r < 0.5:
                RB = np.random.uniform(-1, 1, size=dimension)
                X2 = Bast_P + (1 - t / Max_iterations) ** 2 * (2 * RB - 1) * X[i, :]
                X2 = np.clip(X2, lowerbound, upperbound)
            else:
                K = 1 + int(round(np.random.rand()))
                R2 = np.random.rand(dimension)
                X2 = X[i, :] + R2 * (Xrandom - K * X[i, :])
                X2 = np.clip(X2, lowerbound, upperbound)

            f_newP2 = fitness(X2, X_train, Y_train, input_dim, hidden_dim, output_dim)
            if f_newP2 <= fit[i]:
                X[i, :] = X2
                fit[i] = f_newP2

        best_so_far[t - 1] = fbest
        print(f"Iteration {t}: Best Cost = {best_so_far[t-1]:.6f}")

    Best_score = fbest
    Best_pos = Bast_P
    return {"Best_score": Best_score, "Best_pos": Best_pos, "SBOA_curve": best_so_far}


def SBOA_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
            lowerbound=-1, upperbound=1):
    # Normalization
    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)
    data_X = X_train.copy()
    data_y = Y_train.copy()

    max_data_X = X_train.max(axis=0)
    min_data_X = X_train.min(axis=0)
    Xn = (X_train - min_data_X) / (max_data_X - min_data_X + 1e-12)

    max_data_y = Y_train.max(axis=0)
    min_data_y = Y_train.min(axis=0)
    Yn = (Y_train - min_data_y) / (max_data_y - min_data_y + 1e-12)

    input_dim = Xn.shape[1]
    output_dim = Yn.shape[1] if Yn.ndim > 1 else 1
    if Yn.ndim == 1:
        Yn = Yn.reshape(-1, 1)

    num_params = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim

    result = SBOA(Xn, Yn, input_dim, hidden_dim, output_dim,
                  SearchAgents, Max_iterations, lowerbound, upperbound, num_params, fitness_function)

    best_params = result["Best_pos"]
    # unpack
    idx = 0
    W1 = best_params[idx: idx + input_dim * hidden_dim].reshape((input_dim, hidden_dim))
    idx += input_dim * hidden_dim
    b1 = best_params[idx: idx + hidden_dim]
    idx += hidden_dim
    W2 = best_params[idx: idx + hidden_dim * output_dim].reshape((hidden_dim, output_dim))
    idx += hidden_dim * output_dim
    b2 = best_params[idx: idx + output_dim]

    weights = {"W1": W1, "W2": W2}
    biases = {"b1": b1, "b2": b2}

    return {"weights": weights, "biases": biases, "sboa_curve": result["SBOA_curve"],
            "normalization_X": (min_data_X, max_data_X), "normalization_y": (min_data_y, max_data_y)}
