import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

# Read and process user_clip.csv
user_clip_dict = {}
user_ids = set()
clip_ids = set()

with open('user_clip.csv', 'r') as file:
    for line in file:
        if line.strip() == "user_id,clip_id,weight":
            continue
        user_id, clip_id, weight = line.strip().split(',')
        user_id, clip_id, weight = int(user_id), int(clip_id), int(weight) if int(weight) > 0 else 0
        user_clip_dict[(user_id, clip_id)] = weight
        user_ids.add(user_id)
        clip_ids.add(clip_id)

user_ids = list(user_ids)
clip_ids = list(clip_ids)

# Mapping from user/clip ID to index
user_index = {u: idx for idx, u in enumerate(user_ids)}
clip_index = {i: idx for idx, i in enumerate(clip_ids)}

# Calculate average rating
r_avg = np.mean(list(user_clip_dict.values()))

# Number of users and clips
num_users = len(user_ids)
num_clips = len(clip_ids)

# Prepare data for sparse matrix construction
rows = []
cols = []
data = []
C = []

# Fill data for user-clip interactions
for idx, ((user, clip), rating) in enumerate(user_clip_dict.items()):
    rows.append(idx)
    cols.append(user_index[user])
    data.append(-1)

    rows.append(idx)
    cols.append(num_users + clip_index[clip])
    data.append(-1)

    C.append(r_avg - rating)

C = np.array(C)

# Add regularization to A (identity matrix multiplied by lambda)
lambda_reg = 0.316227766
reg_rows = []
reg_cols = []
reg_data = []

for i in range(num_users + num_clips):
    reg_rows.append(len(user_clip_dict) + i)
    reg_cols.append(i)
    reg_data.append(lambda_reg)

# Combine user-clip and regularization data
rows.extend(reg_rows)
cols.extend(reg_cols)
data.extend(reg_data)

# Construct the sparse matrix A
A = coo_matrix((data, (rows, cols)), shape=(len(user_clip_dict) + num_users + num_clips, num_users + num_clips))

# Solve the least squares problem A * B = C using sparse matrix solver
B = lsqr(A, np.hstack([C, np.zeros(num_users + num_clips)]))[0]

# Extract user and clip biases
b_u = B[:num_users]
b_i = B[num_users:]

# Create mappings from indexes back to user/clip IDs
b_u_dict = {user_ids[i]: b_u[i] for i in range(num_users)}
b_i_dict = {clip_ids[i]: b_i[i] for i in range(num_clips)}

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
    print(f"{user_id},{clip_id},{prediction}")

# Calculate and print the minimized residual sum of squares
minimized_loss = np.sum((A.dot(B) - np.hstack([C, np.zeros(num_users + num_clips)])) ** 2)
print(f"Minimized loss: {minimized_loss}")
