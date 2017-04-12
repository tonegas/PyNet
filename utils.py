

# def to_one_hot_vect(vect, num_classes):
#     on_hot_vect = []
#     for i,target in enumerate(vect):
#         on_hot_vect.append(np.zeros(num_classes))
#         on_hot_vect[i][target] = 1
#     return on_hot_vect

def to_one_hot_vect(ind, num_classes):
    on_hot_vect = np.zeros(num_classes)
    on_hot_vect[ind] = 1
    return on_hot_vect