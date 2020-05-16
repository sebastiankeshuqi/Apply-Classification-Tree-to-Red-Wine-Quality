def get_data(file_name):
    f = open(file_name,'r').readlines()
    data_name = f[0].split(',')[:-1]
    data = []
    for i in f[1:]:
        d = [float(j) for j in i.split(',')]
        d[-1] = d[-1] > 6
        data.append(d)
    return data_name, data