import pickle


def save_pickle(data, filepath):

    # save data as *.pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    f.close()

    print(f'{filepath} SAVED')