import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def part_1():
    # We solve least squares of the form ||A * b - c||
    # b is the biases
    # we construct A and c to match the problem, including the regularization

    # Read user_clip.csv
    df = pd.read_csv('user_clip.csv')
    user_clip_dict = {(row['user_id'], row['clip_id']): row['weight'] for index, row in df.iterrows()}
    user_ids = df['user_id'].unique()
    clip_ids = df['clip_id'].unique()

    # Mapping user/clip ID to index
    user_index = {u: idx for idx, u in enumerate(user_ids)}
    clip_index = {i: idx for idx, i in enumerate(clip_ids)}

    # Calculate average rating
    r_avg = np.mean(list(user_clip_dict.values()))

    # Number of users and clips
    num_users = len(user_ids)
    num_clips = len(clip_ids)

    # We make a sparse matrix
    # We need a list of rows, list of cols and list of the data itself
    user_clip_rows = []
    user_clip_cols = []
    user_clip_data = []
    user_clip_c = np.zeros(len(user_clip_dict))

    # Fill lists for user-clip watching
    for idx, ((user, clip), rating) in enumerate(user_clip_dict.items()):
        user_clip_rows.append(idx)
        user_clip_cols.append(user_index[user])
        user_clip_data.append(-1)

        user_clip_rows.append(idx)
        user_clip_cols.append(num_users + clip_index[clip])
        user_clip_data.append(-1)

        user_clip_c[idx] = r_avg - rating

    # Fills list for regularization
    # A has in the bottom an identity matrix multiplied by the square root of lambda
    sqrt_lambda_reg = np.sqrt(0.1)
    reg_rows = list(range(len(user_clip_dict), len(user_clip_dict) + num_users + num_clips))
    reg_cols = list(range(num_users + num_clips))
    reg_data = [sqrt_lambda_reg] * (num_users + num_clips)

    reg_c = np.zeros(num_users + num_clips)

    # Combine user-clip and regularization data, rows and cols
    rows = user_clip_rows + reg_rows
    cols = user_clip_cols + reg_cols
    data = user_clip_data + reg_data

    # Construct the sparse matrix A, and c
    A = coo_matrix((data, (rows, cols)), shape=(len(user_clip_dict) + num_users + num_clips, num_users + num_clips))
    c = np.hstack([user_clip_c, reg_c])

    # Solve the least squares problem ||A * b - c|| using sparse matrices
    b = lsqr(A, c)[0]

    # Take user and clip biases
    b_users = b[:num_users]
    b_clips = b[num_users:]

    # Create mappings from indexes back to user and clip ids
    user_id_to_b_u_dict = {user_ids[i]: b_users[i] for i in range(num_users)}
    clip_id_to_b_i_dict = {clip_ids[i]: b_clips[i] for i in range(num_clips)}

    # Read test.csv for predictions
    test_dict = read_test('test.csv')
    test_predictions = []

    # Make predictions on test
    for item in test_dict:
        user_id = item['user_id']
        clip_id = item['clip_id']

        b_users = user_id_to_b_u_dict.get(user_id, 0)
        b_clips = clip_id_to_b_i_dict.get(clip_id, 0)
        # predictions can't be negative
        r_predict = max(0, r_avg + b_users + b_clips)

        test_predictions.append({'user_id': user_id, 'clip_id': clip_id, 'weight': r_predict})

    save_predictions('337604821_213203573_task1.csv', test_predictions)

    # Calculate f1
    f1_loss = np.sum((A.dot(b) - c) ** 2)
    return f1_loss


def part_2():
    k = 20

    df = pd.read_csv('user_clip.csv')

    users = df['user_id'].unique()
    clips = df['clip_id'].unique()

    user_indexes = {u: idx for idx, u in enumerate(users)}
    clip_indexes = {i: idx for idx, i in enumerate(clips)}

    n_user = len(users)
    n_clips = len(clips)

    R = np.zeros((n_user, n_clips))

    for index, row in df.iterrows():
        user = row['user_id']
        clip = row['clip_id']
        r = row['weight']

        R[user_indexes[user], clip_indexes[clip]] = r

    # make SVD
    u, s, v_T = np.linalg.svd(R, full_matrices=False)

    # make low dimensional reconstructions
    v_k_T = v_T[:k, :]
    s_mat_k = np.diag(s[:k])
    u_k = u[:, :k]

    # make all predictions
    R_hat = np.dot(u_k, np.dot(s_mat_k, v_k_T))
    R_hat[R_hat < 0] = 0

    # the error is 0 if not in the data
    error = (R - R_hat) ** 2
    error[R == 0] = 0

    test_dict = read_test('test.csv')
    test_predictions = []

    # Make predictions on test
    for item in test_dict:
        user_id = item['user_id']
        clip_id = item['clip_id']

        user_index = user_indexes[user_id]
        clip_index = clip_indexes[clip_id]

        # predictions can't be negative
        r_predict = max(0, R_hat[user_index, clip_index])

        test_predictions.append({'user_id': user_id, 'clip_id': clip_id, 'weight': r_predict})

    save_predictions('337604821_213203573_task2.csv', test_predictions)

    return np.sum(error)


def save_predictions(filename, predictions):
    df = pd.DataFrame(predictions)
    df = df.rename(columns={'clip_id': 'song_id'})
    df.to_csv(filename, header=True, index=False, columns=['user_id', 'song_id', 'weight'])


def read_test(filename):
    df = pd.read_csv(filename)
    df = df.drop(['Unnamed: 0'], axis=1)

    return df.to_dict('records')


if __name__ == '__main__':
    read_test('test.csv')
    f1 = part_1()
    print(f'f1 = {f1}')

    f2 = part_2()
    print(f'f2 = {f2}')