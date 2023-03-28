class simu_data:
    def __init__(self, num_samples, x_min, x_max, para=None, the_seed = 7):
        if para is None:
            para = {}
        self.num_samples = num_samples
        self.x_min = x_min
        self.x_max = x_max
        self.para = para
        self.the_seed = the_seed