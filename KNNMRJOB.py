from mrjob.job import MRJob
import csv
import numpy as np

class KNNMRJob(MRJob):
    DELIMITER = ","
    minmaxdata = []
    
    #input command line arguments
    def configure_args(self):
        super(KNNMRJob, self).configure_args()
        self.add_passthru_arg("--max_K", type=int, help="Max value of K")
        self.add_file_arg("--test", help='Path of test dataset')        


    #Mapper
    def mapper(self, _, row):
        train_values = row.strip().split(self.DELIMITER)        
        train_nparray = np.array(train_values[1:-1], dtype=float)

        with open(self.options.test, 'r') as test_file:
            test_csv_reader = csv.reader(test_file)
            for test_values in test_csv_reader:
                line_values = np.array(test_values[1:-1], dtype=float)
                #calculate elucidian distance
                sum_vectors = np.sum(np.square(line_values - train_nparray))
                dist = np.sqrt(sum_vectors)
                yield test_values[0], (dist, train_values[-1], test_values[-1])
    

    #Combiner
    def combiner(self, key, values):
        pq = []
        values = list(values)
        for value in values:
            if len(pq) < self.options.max_K:
                pq.append(value)
            else:
                max_value = max(pq, key=lambda x: x[0])
                if max_value[0] < value[0]:
                    pq.remove(max_value)
                    pq.append(value)
        for value in pq:
            yield key, value

    
    #Reducer
    def reducer(self, key, values):
        pq = []
        for value in values:
            if len(pq) < self.options.max_K:
                pq.append(value)
            else:
                max_value = max(pq, key=lambda x: x[0])
                if max_value[0] > value[0]:
                    pq.remove(max_value)
                    pq.append(value)
        for value in pq:
            yield key, value


if __name__ == '__main__':
    KNNMRJob.run()