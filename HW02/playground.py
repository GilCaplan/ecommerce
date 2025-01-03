import numpy as np
from scipy.optimize import minimize

# Read and process user_clip.csv
user_bias = {}
clip_bias = {}
user_clip_dict = {}
with open('user_clip.csv', 'r') as file:
    for line in file:
        if line.strip() == "user_id,clip_id,weight":
            continue
        user_id, clip_id, weight = line.strip().split(',')
        weight = int(weight) if int(weight) > 0 else 0
        user_clip_dict[(int(user_id), int(clip_id))] = weight
        if int(user_id) not in user_bias:
            user_bias[int(user_id)] = []
        user_bias[int(user_id)].append(weight)
        if int(clip_id) not in clip_bias:
            clip_bias[int(clip_id)] = []
        clip_bias[int(clip_id)].append(weight)

# Calculate average rating
r_avg = sum(user_clip_dict.values()) / len(user_clip_dict.values())

# Extract unique users and clips
users = list(user_bias.keys())
clips = list(clip_bias.keys())

# Mapping from user/clip ID to index
user_index = {u: idx for idx, u in enumerate(users)}
clip_index = {i: idx for idx, i in enumerate(clips)}


# Define the loss function
def loss_function(biases):
    b_u = biases[:len(users)]
    b_i = biases[len(users):]

    squared_error_sum = 0
    for (user, clip), r_ui in user_clip_dict.items():
        pred = r_avg + b_u[user_index[user]] + b_i[clip_index[clip]]
        squared_error_sum += (r_ui - pred) ** 2

    regularization_sum = 0.1 * (np.sum(b_u ** 2) + np.sum(b_i ** 2))

    return squared_error_sum + regularization_sum


# Initial biases (concatenated b_u and b_i)
initial_biases = np.zeros(len(users) + len(clips))

# Perform optimization
result = minimize(loss_function, initial_biases, method='L-BFGS-B')

# Extract optimized biases
optimized_biases = result.x
b_u_opt = optimized_biases[:len(users)]
b_i_opt = optimized_biases[len(users):]

# Create mappings from indexes back to user/clip IDs
b_u_dict = {users[i]: b_u_opt[i] for i in range(len(users))}
b_i_dict = {clips[i]: b_i_opt[i] for i in range(len(clips))}

# Read and process test.csv for predictions
test_dict = {}
with open('test.csv', 'r') as file:
    for line in file:
        if line.strip() == ",user_id,clip_id":
            continue
        _, user_id, clip_id = line.strip().split(',')
        user_id, clip_id = int(user_id), int(clip_id)
        b_u = b_u_dict.get(user_id, 0)
        b_i = b_i_dict.get(clip_id, 0)
        r_predict = r_avg + b_u + b_i
        test_dict[(user_id, clip_id)] = r_predict

for (user_id, clip_id), prediction in test_dict.items():
    print(f"User {user_id}, Clip {clip_id}, Predicted Weight: {prediction}")

# Calculate and print the minimized loss function value
minimized_loss = loss_function(optimized_biases)
print(f"Minimized loss: {minimized_loss}")
