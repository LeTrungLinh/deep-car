
def distance_finder(focal_length, object_width_in_frame, real_object_width):
    distance = (real_object_width*focal_length)/object_width_in_frame
    return distance