import random

def hyperparameter_tuning_grid(R, num_iters, lamb, num_features, num_users, num_items):
    best_rmse = float('inf')
    best_params = None

    for rank in num_features:
        for reg in lamb:
            for num_iter in num_iters:
                print(f"Training ALS: features={rank}, lambda={reg}, iterations={num_iter}")

                U, V = als(R, num_users, num_items, num_iter, rank, reg)

                pred = predict(U, V)
                rmse = compute_mask_rmse(val_matrix, pred, val_mask)
                print(f"Validation RMSE = {rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = (rank, reg, num_iter)
    
                print("Best hyperparameters:", best_params)
                print("Best RMSE:", best_rmse)
    return best_params

def hyperparameter_tuning_random(R, num_users, num_items):
    # Define search space
    latent_feature_choices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    iteration_choices = [5, 10, 20, 30, 40, 50]
    lambda_choices = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

    # How many random combinations to try
    max_trials = 60
    early_stopping_rounds = 10
    no_improve_count = 0

    best_rmse = float('inf')
    best_params = None

    for trial in range(max_trials):
        # Randomly sample parameters
        rank = random.choice(latent_feature_choices)
        num_iter = random.choice(iteration_choices)
        reg = random.choice(lambda_choices)
        print(f"Training ALS: features={rank}, lambda={reg}, iterations={num_iter}")

        U, V = als(R, num_users, num_items, num_iter, rank, reg)

        pred = predict(U, V)
        rmse = compute_mask_rmse(val_matrix, pred, val_mask)
        print(f"Trial = {trial+1}, Validation RMSE = {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = (rank, reg, num_iter)
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"No improvement count: {no_improve_count}")

        # Early stopping
        if no_improve_count >= early_stopping_rounds:
            print(f"\nEarly stopping after {trial+1} number of trials with no improvement for {early_stopping_rounds} rounds.")
            break

        print("Best hyperparameters:", best_params)
        print(f"Best RMSE: {best_rmse:.4f}" )
    return best_params