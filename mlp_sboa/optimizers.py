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


def GWO(X_train, Y_train, input_dim, hidden_dim, output_dim,
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """Grey Wolf Optimizer (GWO) for MLP."""
    Alpha_pos = np.zeros(dimension)
    Alpha_score = float("inf")
    Beta_pos = np.zeros(dimension)
    Beta_score = float("inf")
    Delta_pos = np.zeros(dimension)
    Delta_score = float("inf")

    Positions = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))
    Convergence_curve = np.zeros(Max_iterations)

    for l in range(Max_iterations):
        for i in range(SearchAgents):
            Positions[i, :] = np.clip(Positions[i, :], lowerbound, upperbound)
            fit = fitness(Positions[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)

            if fit < Alpha_score:
                Alpha_score = fit
                Alpha_pos = Positions[i, :].copy()
            elif Alpha_score < fit < Beta_score:
                Beta_score = fit
                Beta_pos = Positions[i, :].copy()
            elif fit > Alpha_score and fit > Beta_score and fit < Delta_score:
                Delta_score = fit
                Delta_pos = Positions[i, :].copy()

        a = 2.0 - (l * (2.0 / Max_iterations))

        for i in range(SearchAgents):
            for j in range(dimension):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3.0

        Convergence_curve[l] = Alpha_score
        print(f"Iteration {l+1}: Best Cost = {Convergence_curve[l]:.6f}")

    return {"Best_pos": Alpha_pos, "Best_score": Alpha_score, "curve": Convergence_curve}


def GWO_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
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

    result = GWO(Xn, Yn, input_dim, hidden_dim, output_dim,
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


def MFO(X_train, Y_train, input_dim, hidden_dim, output_dim,
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """Moth-Flame Optimization (MFO) for MLP."""
    Moth_pos = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))
    Convergence_curve = np.zeros(Max_iterations)

    previous_population = None
    previous_fitness = None
    best_flames = None
    best_flame_fitness = None
    Best_flame_score = float("inf")
    Best_flame_pos = None

    for Iteration in range(1, Max_iterations + 1):
        Flame_no = int(round(SearchAgents - Iteration * ((SearchAgents - 1) / Max_iterations)))
        Flame_no = max(1, Flame_no)

        Moth_pos = np.clip(Moth_pos, lowerbound, upperbound)
        Moth_fitness = np.zeros(SearchAgents)
        for i in range(SearchAgents):
            Moth_fitness[i] = fitness(Moth_pos[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)

        if Iteration == 1:
            sorted_indices = np.argsort(Moth_fitness)
            sorted_population = Moth_pos[sorted_indices, :].copy()
            fitness_sorted = Moth_fitness[sorted_indices].copy()
            best_flames = sorted_population.copy()
            best_flame_fitness = fitness_sorted.copy()
        else:
            double_population = np.vstack([previous_population, best_flames])
            double_fitness = np.concatenate([previous_fitness, best_flame_fitness])
            sorted_indices = np.argsort(double_fitness)
            double_sorted_population = double_population[sorted_indices, :]

            fitness_sorted = double_fitness[sorted_indices][:SearchAgents]
            sorted_population = double_sorted_population[:SearchAgents, :]
            best_flames = sorted_population.copy()
            best_flame_fitness = fitness_sorted.copy()

        Best_flame_score = fitness_sorted[0]
        Best_flame_pos = sorted_population[0, :].copy()

        previous_population = Moth_pos.copy()
        previous_fitness = Moth_fitness.copy()

        a = -1.0 + Iteration * (-1.0 / Max_iterations)

        for i in range(SearchAgents):
            for j in range(dimension):
                b = 1.0
                t = (a - 1.0) * np.random.rand() + 1.0
                if i < Flame_no:
                    distance_to_flame = abs(sorted_population[i, j] - Moth_pos[i, j])
                    Moth_pos[i, j] = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + sorted_population[i, j]
                else:
                    distance_to_flame = abs(sorted_population[i, j] - Moth_pos[i, j])
                    Moth_pos[i, j] = distance_to_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + sorted_population[Flame_no - 1, j]

        Convergence_curve[Iteration - 1] = Best_flame_score
        print(f"Iteration {Iteration}: Best Cost = {Convergence_curve[Iteration - 1]:.6f}")

    return {"Best_pos": Best_flame_pos, "Best_score": Best_flame_score, "curve": Convergence_curve}


def MFO_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
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

    result = MFO(Xn, Yn, input_dim, hidden_dim, output_dim,
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


def PSO(X_train, Y_train, input_dim, hidden_dim, output_dim,
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness,
        w=1.0, wdamp=0.99, c1=1.5, c2=2.0):
    """Particle Swarm Optimization (PSO) for MLP."""
    vel_max = 0.1 * (upperbound - lowerbound)
    vel_min = -upperbound

    positions = [np.random.uniform(lowerbound, upperbound, size=dimension) for _ in range(SearchAgents)]
    velocities = [np.zeros(dimension) for _ in range(SearchAgents)]
    costs = [float("inf")] * SearchAgents
    pbest_positions = [pos.copy() for pos in positions]
    pbest_costs = [float("inf")] * SearchAgents

    gbest_position = None
    gbest_cost = float("inf")

    for i in range(SearchAgents):
        c = fitness(positions[i], X_train, Y_train, input_dim, hidden_dim, output_dim)
        costs[i] = c
        pbest_positions[i] = positions[i].copy()
        pbest_costs[i] = c
        if c < gbest_cost:
            gbest_cost = c
            gbest_position = positions[i].copy()

    best_cost_history = np.zeros(Max_iterations)

    for it in range(Max_iterations):
        for i in range(SearchAgents):
            r1 = np.random.rand(dimension)
            r2 = np.random.rand(dimension)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest_positions[i] - positions[i]) +
                             c2 * r2 * (gbest_position - positions[i]))
            velocities[i] = np.clip(velocities[i], vel_min, vel_max)

            positions[i] = positions[i] + velocities[i]

            is_outside = (positions[i] < lowerbound) | (positions[i] > upperbound)
            velocities[i][is_outside] = -velocities[i][is_outside]

            positions[i] = np.clip(positions[i], lowerbound, upperbound)

            c = fitness(positions[i], X_train, Y_train, input_dim, hidden_dim, output_dim)
            costs[i] = c

            if c < pbest_costs[i]:
                pbest_positions[i] = positions[i].copy()
                pbest_costs[i] = c
                if c < gbest_cost:
                    gbest_cost = c
                    gbest_position = positions[i].copy()

        best_cost_history[it] = gbest_cost
        print(f"Iteration {it+1}: Best Cost = {best_cost_history[it]:.6f}")
        w *= wdamp

    return {"Best_pos": gbest_position, "Best_score": gbest_cost, "curve": best_cost_history}


def PSO_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
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

    result = PSO(Xn, Yn, input_dim, hidden_dim, output_dim,
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


def SSA(X_train, Y_train, input_dim, hidden_dim, output_dim,
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """Salp Swarm Algorithm (SSA) for MLP."""
    lb = np.full(dimension, lowerbound)
    ub = np.full(dimension, upperbound)

    SalpPositions = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))
    SalpFitness = np.zeros(SearchAgents)

    for i in range(SearchAgents):
        SalpFitness[i] = fitness(SalpPositions[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)

    sorted_idx = np.argsort(SalpFitness)
    FoodPosition = SalpPositions[sorted_idx[0], :].copy()
    FoodFitness = SalpFitness[sorted_idx[0]]

    Convergence_curve = np.zeros(Max_iterations)

    for l in range(1, Max_iterations + 1):
        c1 = 2.0 * np.exp(-((4.0 * l / Max_iterations) ** 2))

        for i in range(SearchAgents):
            if i < SearchAgents / 2:
                for j in range(dimension):
                    c2 = np.random.rand()
                    c3 = np.random.rand()
                    if c3 < 0.5:
                        SalpPositions[i, j] = FoodPosition[j] + c1 * ((ub[j] - lb[j]) * c2 + lb[j])
                    else:
                        SalpPositions[i, j] = FoodPosition[j] - c1 * ((ub[j] - lb[j]) * c2 + lb[j])
            else:
                point1 = SalpPositions[i - 1, :]
                point2 = SalpPositions[i, :]
                SalpPositions[i, :] = (point2 + point1) / 2.0

        for i in range(SearchAgents):
            SalpPositions[i, :] = np.clip(SalpPositions[i, :], lowerbound, upperbound)
            fit = fitness(SalpPositions[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)

            if fit < FoodFitness:
                FoodFitness = fit
                FoodPosition = SalpPositions[i, :].copy()

        Convergence_curve[l - 1] = FoodFitness
        print(f"Iteration {l}: Best Cost = {Convergence_curve[l - 1]:.6f}")

    return {"Best_pos": FoodPosition, "Best_score": FoodFitness, "curve": Convergence_curve}


def SSA_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
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

    result = SSA(Xn, Yn, input_dim, hidden_dim, output_dim,
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


def WOA(X_train, Y_train, input_dim, hidden_dim, output_dim,
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """Whale Optimization Algorithm (WOA) for MLP."""
    Leader_pos = np.zeros(dimension)
    Leader_score = float("inf")

    Positions = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))
    Convergence_curve = np.zeros(Max_iterations)

    for t in range(Max_iterations):
        for i in range(SearchAgents):
            Positions[i, :] = np.clip(Positions[i, :], lowerbound, upperbound)
            fit = fitness(Positions[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)

            if fit < Leader_score:
                Leader_score = fit
                Leader_pos = Positions[i, :].copy()

        a = 2.0 - t * (2.0 / Max_iterations)
        a2 = -1.0 + t * (-1.0 / Max_iterations)

        for i in range(SearchAgents):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            b = 1.0
            l = (a2 - 1.0) * np.random.rand() + 1.0
            p = np.random.rand()

            for j in range(dimension):
                if p < 0.5:
                    if abs(A) >= 1.0:
                        rand_leader_idx = np.random.randint(0, SearchAgents)
                        X_rand = Positions[rand_leader_idx, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand
                    else:
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader
                else:
                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    Positions[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2.0 * np.pi) + Leader_pos[j]

        Convergence_curve[t] = Leader_score
        print(f"Iteration {t+1}: Best Cost = {Convergence_curve[t]:.6f}")

    return {"Best_pos": Leader_pos, "Best_score": Leader_score, "curve": Convergence_curve}


def WOA_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
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

    result = WOA(Xn, Yn, input_dim, hidden_dim, output_dim,
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


def ZOA(X_train, Y_train, input_dim, hidden_dim, output_dim,
        SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    """Zebra Optimization Algorithm (ZOA) for MLP."""
    X = np.random.uniform(lowerbound, upperbound, size=(SearchAgents, dimension))
    fit = np.zeros(SearchAgents)

    for i in range(SearchAgents):
        fit[i] = fitness(X[i, :], X_train, Y_train, input_dim, hidden_dim, output_dim)

    best_so_far = np.zeros(Max_iterations)
    PZ = X[np.argmin(fit), :].copy()
    fbest = np.min(fit)

    for t in range(1, Max_iterations + 1):
        best = np.min(fit)
        location = np.argmin(fit)
        if best < fbest:
            fbest = best
            PZ = X[location, :].copy()

        # Phase 1: Foraging
        for i in range(SearchAgents):
            I = int(round(1 + np.random.rand()))
            X_newP1 = X[i, :] + np.random.rand(dimension) * (PZ - I * X[i, :])
            X_newP1 = np.clip(X_newP1, lowerbound, upperbound)

            f_newP1 = fitness(X_newP1, X_train, Y_train, input_dim, hidden_dim, output_dim)
            if f_newP1 <= fit[i]:
                X[i, :] = X_newP1.copy()
                fit[i] = f_newP1

        # Phase 2: Defense
        Ps = np.random.rand()
        k = np.random.randint(0, SearchAgents)
        AZ = X[k, :].copy()

        for i in range(SearchAgents):
            if Ps < 0.5:
                R = 0.1
                X_newP2 = X[i, :] + R * (2.0 * np.random.rand(dimension) - 1.0) * (1.0 - t / Max_iterations) * X[i, :]
            else:
                I = int(round(1 + np.random.rand()))
                X_newP2 = X[i, :] + np.random.rand(dimension) * (AZ - I * X[i, :])

            X_newP2 = np.clip(X_newP2, lowerbound, upperbound)
            f_newP2 = fitness(X_newP2, X_train, Y_train, input_dim, hidden_dim, output_dim)
            if f_newP2 <= fit[i]:
                X[i, :] = X_newP2.copy()
                fit[i] = f_newP2

        best_so_far[t - 1] = fbest
        print(f"Iteration {t}: Best Cost = {best_so_far[t - 1]:.6f}")

    return {"Best_pos": PZ, "Best_score": fbest, "curve": best_so_far}


def ZOA_nn(X_train, Y_train, hidden_dim, SearchAgents, Max_iterations,
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

    result = ZOA(Xn, Yn, input_dim, hidden_dim, output_dim,
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




