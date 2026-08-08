import numpy as np
import math
from mlp_sboa.fitness import fitness_function


def levy_flight(dimension):
    # simple levy step generator
    beta = 1.5
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
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

    return {"weights": weights, "biases": biases, "curve": result["SBOA_curve"], "sboa_curve": result["SBOA_curve"],
            "normalization_X": (min_data_X, max_data_X), "normalization_y": (min_data_y, max_data_y)}


def ABC(X_train, Y_train, input_dim, hidden_dim, output_dim,
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness,
        abandonment_limit_param=0.6, a=1):
    """Artificial Bee Colony (ABC) optimizer for MLP."""
    L = int(round(abandonment_limit_param * dimension * SearchAgents))
    
    pop_position = [np.random.uniform(lowerbound, upperbound, size=dimension) for _ in range(SearchAgents)]
    pop_cost = [fitness(pos, X_train, Y_train, input_dim, hidden_dim, output_dim) for pos in pop_position]
    
    best_sol_pos = pop_position[0].copy()
    best_sol_cost = pop_cost[0]
    for i in range(SearchAgents):
        if pop_cost[i] <= best_sol_cost:
            best_sol_pos = pop_position[i].copy()
            best_sol_cost = pop_cost[i]
            
    C = np.zeros(SearchAgents, dtype=int)
    best_cost = np.zeros(Max_iterations)
    
    for it in range(Max_iterations):
        # Employed Bees
        for i in range(SearchAgents):
            k_choices = [idx for idx in range(SearchAgents) if idx != i]
            k = np.random.choice(k_choices)
            phi = a * np.random.uniform(-1, 1, size=dimension)
            new_pos = pop_position[i] + phi * (pop_position[i] - pop_position[k])
            new_pos = np.clip(new_pos, lowerbound, upperbound)
            new_cost = fitness(new_pos, X_train, Y_train, input_dim, hidden_dim, output_dim)
            if new_cost <= pop_cost[i]:
                pop_position[i] = new_pos
                pop_cost[i] = new_cost
            else:
                C[i] += 1
                
        # Onlooker Bees
        mean_cost = np.mean(pop_cost)
        fit_vals = np.exp(-np.array(pop_cost) / (mean_cost + 1e-12))
        probs = fit_vals / np.sum(fit_vals)
        cum_probs = np.cumsum(probs)
        
        for m in range(SearchAgents):
            r = np.random.rand()
            i = int(np.searchsorted(cum_probs, r))
            if i >= SearchAgents:
                i = SearchAgents - 1
            k_choices = [idx for idx in range(SearchAgents) if idx != i]
            k = np.random.choice(k_choices)
            phi = a * np.random.uniform(-1, 1, size=dimension)
            new_pos = pop_position[i] + phi * (pop_position[i] - pop_position[k])
            new_pos = np.clip(new_pos, lowerbound, upperbound)
            new_cost = fitness(new_pos, X_train, Y_train, input_dim, hidden_dim, output_dim)
            if new_cost <= pop_cost[i]:
                pop_position[i] = new_pos
                pop_cost[i] = new_cost
            else:
                C[i] += 1
                
        # Scout Bees
        for i in range(SearchAgents):
            if C[i] >= L:
                pop_position[i] = np.random.uniform(lowerbound, upperbound, size=dimension)
                pop_cost[i] = fitness(pop_position[i], X_train, Y_train, input_dim, hidden_dim, output_dim)
                C[i] = 0
                
        # Update Best Solution
        for i in range(SearchAgents):
            if pop_cost[i] <= best_sol_cost:
                best_sol_pos = pop_position[i].copy()
                best_sol_cost = pop_cost[i]
                
        best_cost[it] = best_sol_cost
        print(f"Iteration {it+1}: Best Cost = {best_cost[it]:.6f}")
        
    return {"Best_pos": best_sol_pos, "Best_score": best_sol_cost, "curve": best_cost}


def ABC_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
           lowerbound=-1, upperbound=1):
    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)
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

    result = ABC(Xn, Yn, input_dim, hidden_dim, output_dim,
                 SearchAgents, Max_iterations, lowerbound, upperbound, num_params, fitness_function)

    best_params = result["Best_pos"]
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

    return {"weights": weights, "biases": biases, "curve": result["curve"],
            "normalization_X": (min_data_X, max_data_X), "normalization_y": (min_data_y, max_data_y)}


def GTO(X_train, Y_train, input_dim, hidden_dim, output_dim,
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """Gorilla Troops Optimizer (GTO) for MLP."""
    Silverback = np.zeros(dimension)
    Silverback_Score = float("inf")
    
    X = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))
    convergence_curve = np.zeros(Max_iterations)
    Pop_Fit = np.zeros(SearchAgents)
    
    for i in range(SearchAgents):
        Pop_Fit[i] = fitness(X[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)
        if Pop_Fit[i] < Silverback_Score:
            Silverback_Score = Pop_Fit[i]
            Silverback = X[i, :].copy()
            
    GX = X.copy()
    lb = np.full(dimension, lowerbound)
    ub = np.full(dimension, upperbound)
    
    p = 0.03
    Beta = 3
    w = 0.8
    
    for It in range(1, Max_iterations + 1):
        a = (np.cos(2 * np.random.rand()) + 1) * (1 - It / Max_iterations)
        C = a * (2 * np.random.rand() - 1)
        
        # Exploration
        for i in range(SearchAgents):
            if np.random.rand() < p:
                GX[i, :] = (ub - lb) * np.random.rand(dimension) + lb
            else:
                if np.random.rand() >= 0.5:
                    Z = np.random.uniform(-a, a, size=dimension)
                    H = Z * X[i, :]
                    rand_idx = np.random.randint(0, SearchAgents)
                    GX[i, :] = (np.random.rand() - a) * X[rand_idx, :] + C * H
                else:
                    rand_idx1 = np.random.randint(0, SearchAgents)
                    rand_idx2 = np.random.randint(0, SearchAgents)
                    GX[i, :] = X[i, :] - C * (C * (X[i, :] - GX[rand_idx1, :]) + np.random.rand() * (X[i, :] - GX[rand_idx2, :]))
                    
        GX = np.clip(GX, lowerbound, upperbound)
        
        # Group formation
        for i in range(SearchAgents):
            New_Fit = fitness(GX[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :].copy()
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :].copy()
                
        # Exploitation
        for i in range(SearchAgents):
            if a >= w:
                g = 2 ** C
                col_means = np.mean(GX, axis=0)
                delta = (np.abs(col_means) ** g) ** (1 / g)
                GX[i, :] = C * delta * (X[i, :] - Silverback) + X[i, :]
            else:
                if np.random.rand() >= 0.5:
                    h = np.random.randn(dimension)
                else:
                    h = np.random.randn()
                r1 = np.random.rand()
                GX[i, :] = Silverback - (Silverback * (2 * r1 - 1) - X[i, :] * (2 * r1 - 1)) * (Beta * h)
                
        GX = np.clip(GX, lowerbound, upperbound)
        
        # Group formation
        for i in range(SearchAgents):
            New_Fit = fitness(GX[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)
            if New_Fit < Pop_Fit[i]:
                Pop_Fit[i] = New_Fit
                X[i, :] = GX[i, :].copy()
            if New_Fit < Silverback_Score:
                Silverback_Score = New_Fit
                Silverback = GX[i, :].copy()
                
        convergence_curve[It - 1] = Silverback_Score
        print(f"Iteration {It}: Best Cost = {convergence_curve[It-1]:.6f}")
        
    return {"Best_pos": Silverback, "Best_score": Silverback_Score, "curve": convergence_curve}


def GTO_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
           lowerbound=-1, upperbound=1):
    X_train = np.array(X_train, dtype=float)
    Y_train = np.array(Y_train, dtype=float)
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

    result = GTO(Xn, Yn, input_dim, hidden_dim, output_dim,
                 SearchAgents, Max_iterations, lowerbound, upperbound, num_params, fitness_function)

    best_params = result["Best_pos"]
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

    return {"weights": weights, "biases": biases, "curve": result["curve"],
            "normalization_X": (min_data_X, max_data_X), "normalization_y": (min_data_y, max_data_y)}

