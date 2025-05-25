import random
from ALS import ALS_Cold_Start, ALS_Training, ALS_Recommendation, ALS_Hyperparameter, ALS_Evaluation

filepath = "dataset/" 

def hyperparameter_tuning_grid(R, test_data, num_users, num_items, I, I3):
    best_rmse = float('inf')
    best_params = None

    num_features = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    lamb = [0.001, 0.01, 0.1, 1.0]
    num_iters = [20, 50]
    
    with open(filepath + "hyperparameter_logs.txt", 'a') as file:
        for rank in num_features:
            for reg in lamb:
                for num_iter in num_iters:
                    print(f"Training ALS: features={rank}, lambda={reg}, iterations={num_iter}")
                    file.write(f"Training ALS: features={rank}, lambda={reg}, iterations={num_iter}")

                    #U, V = ALS_Training.als(R, V, num_users, num_items, I, num_iter, rank, reg)
                    U, V = ALS_Training.als2(R, test_data, I, I3, reg, rank, num_iter, num_users, num_items)
                    
                    rmse = ALS_Evaluation.rmse(I3, test_data, U, V)
                    print(f"Validation RMSE = {rmse:.4f}")
                    file.write(f"\tValidation RMSE = {rmse:.4f}\n")

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = (rank, reg, num_iter)
        
                    print("Best hyperparameters:", best_params)
                    print("Best RMSE:", best_rmse)
                    # file.write(f"Best hyperparameters: {best_params}\n")
                    # file.write(f"Best RMSE: {best_rmse}\n\n")
                    file.flush()
        file.write(f"Best hyperparameters: {best_params}\n")
        file.write(f"Best RMSE: {best_rmse}")
    return best_params

def hyperparameter_tuning_random(R, test_data, num_users, num_items):
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

        U, V = ALS_Training.als(R, num_users, num_items, num_iter, rank, reg)

        pred = ALS_Recommendation.predict(U, V)
        rmse = ALS_Evaluation.compute_rmse(test_data, U, V)
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