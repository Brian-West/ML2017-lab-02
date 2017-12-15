# first line: 1
@mem.cache
def get_data(filename):
    data=load_svmlight_file(filename, n_features=123)
    return data[0], data[1]
